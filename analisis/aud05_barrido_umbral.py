"""Sensibilidad del margen de anticipacion frente al umbral de inicio.
"""
# -*- coding: utf-8 -*-
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location(
    "p5", "/home/claude/work/05_puerta_intencion.py")
p5 = importlib.util.module_from_spec(spec)
sys.argv = ["05"]
spec.loader.exec_module(p5)
p5.DIR_DATOS = "/mnt/user-data/uploads/outputs/entrenamiento de la red/data"

PRE_MS, POST_MS = 2000, 500
BASE = (-1000, -500)
REJILLA = np.arange(-PRE_MS, POST_MS + 1, p5.HOP_MS)
UMBRALES = [5.0, 7.0, 10.0, 15.0]
PERSISTENCIAS = [50, 100]


def cargar():
    """Caracteristicas, etiquetas y velocidades de cada sesion, una sola vez."""
    import glob, os
    rutas = sorted(glob.glob(os.path.join(p5.DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    n = int(p5.FS * p5.WIN_MS / 1000)
    h = int(p5.FS * p5.HOP_MS / 1000)
    sesiones = []
    for si, r in enumerate(rutas):
        e = pd.read_csv(r, low_memory=False)
        lab = e["label"].fillna("NULL").values
        B = p5.filtrar(e["EMG_ch2_raw_bic"].values)
        T = p5.filtrar(e["EMG_ch3_raw_tri"].values)
        F = np.c_[p5.caracteristicas(p5.ventanas(B, n, h)),
                  p5.caracteristicas(p5.ventanas(T, n, h))]
        F = np.sign(F) * np.log1p(np.abs(F))
        sesiones.append(dict(
            si=si, F=F, fin=np.arange(len(F)) * h + n - 1, lab=lab,
            v_el=np.abs(e["ElbowIMUOmega_dps"].values),
            v_sh=np.abs(e["ShoulderSpd_dps"].values),
            cortes=np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]]), n=n))
        print("   sesion %d procesada, %d ventanas" % (si + 1, len(F)))
    return sesiones


def conjunto(sesiones, umbral, persist):
    """Conjunto de la puerta y trayectorias, con el criterio dado."""
    X, y, g, tray = [], [], [], []
    for s in sesiones:
        F, fin, lab, cortes, n = s["F"], s["fin"], s["lab"], s["cortes"], s["n"]
        for k, i0 in enumerate(cortes):
            if lab[i0] not in p5.GESTOS or k == 0:
                continue
            i1 = cortes[k + 1] if k + 1 < len(cortes) else len(lab)
            v = s["v_el"][i0:i1] if "codo" in lab[i0] else s["v_sh"][i0:i1]
            tm = p5.primer_cruce(v, umbral, persist)
            if tm is None or tm < p5.ANCHO_PRE + n:
                continue
            t0 = i0 + tm
            m = (fin >= t0 - p5.ANCHO_PRE) & (fin < t0)
            j1 = i0 - p5.SEP_REPOSO
            j0 = max(cortes[k - 1] + n, j1 - p5.ANCHO_PRE)
            if j1 - j0 < n or m.sum() == 0:
                continue
            mr = (fin >= j0) & (fin < j1)
            if mr.sum() == 0:
                continue
            ep = s["si"] * 1000 + k
            X.append(F[m]);  y.append(np.ones(m.sum()));  g.append(np.full(m.sum(), ep))
            X.append(F[mr]); y.append(np.zeros(mr.sum())); g.append(np.full(mr.sum(), ep))
            idx = np.searchsorted(fin, t0 + REJILLA)
            if idx[-1] < len(F):
                tray.append((ep, F[idx]))
    if not X:
        return None
    return (np.concatenate(X), np.concatenate(y).astype(int),
            np.concatenate(g), tray)


def divergencia(D, replicas=1000, semilla=20260918):
    ne = D.shape[0]
    umbral_t = stats.t.ppf(0.975, ne - 1)

    def masas(t):
        fuera = np.abs(t) > umbral_t
        salida, i = [], 0
        while i < len(fuera):
            if fuera[i]:
                j = i
                while j + 1 < len(fuera) and fuera[j + 1]:
                    j += 1
                salida.append((i, float(np.abs(t[i:j + 1]).sum())))
                i = j + 1
            else:
                i += 1
        return salida

    def tval(M):
        return M.mean(0) / np.maximum(M.std(0, ddof=1) / np.sqrt(M.shape[0]), 1e-12)

    obs = masas(tval(D))
    if not obs:
        return None, None, None
    i0, masa = max(obs, key=lambda x: x[1])
    rng = np.random.default_rng(semilla)
    nulo = np.empty(replicas)
    for r in range(replicas):
        m = masas(tval(D * rng.choice([-1.0, 1.0], size=(ne, 1))))
        nulo[r] = max([x[1] for x in m]) if m else 0.0
    p = (1.0 + (nulo >= masa).sum()) / (1.0 + replicas)
    return REJILLA[i0], masa, p


def main():
    print("Calculando características por sesión (una sola vez)...")
    sesiones = cargar()
    filas = []
    for persist in PERSISTENCIAS:
        for umbral in UMBRALES:
            r = conjunto(sesiones, umbral, persist)
            if r is None:
                print("  umbral %.0f, persistencia %d ms: sin episodios"
                      % (umbral, persist))
                continue
            X, y, g, tray = r
            eps = np.unique(g)
            prob = np.zeros(len(y))
            modelos = {}
            for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                m = RandomForestClassifier(300, random_state=p5.SEMILLA, n_jobs=-1)
                m.fit(X[tr], y[tr])
                prob[te] = m.predict_proba(X[te])[:, 1]
                for e in np.unique(g[te]):
                    modelos[e] = m
            auc = roc_auc_score(y, prob)
            P = np.array([modelos[ep].predict_proba(F)[:, 1]
                          for ep, F in tray if ep in modelos])
            base = (REJILLA >= BASE[0]) & (REJILLA < BASE[1])
            D = P - P[:, base].mean(1, keepdims=True)
            div, masa, pv = divergencia(D)
            filas.append(dict(umbral=umbral, persistencia=persist,
                              episodios=int(len(eps)),
                              ventanas=int(len(y)), auc=round(float(auc), 4),
                              divergencia_ms=None if div is None else int(div),
                              p=None if pv is None else round(float(pv), 4)))
            print("  umbral %4.0f °/s | persistencia %3d ms | %2d episodios | "
                  "AUC %.4f | divergencia %s | p %s"
                  % (umbral, persist, len(eps), auc,
                     "sin conglomerado" if div is None else "%+d ms" % div,
                     "-" if pv is None else "%.4f" % pv))
    json.dump(filas, open("/home/claude/work/barrido_umbral.json", "w"),
              ensure_ascii=False, indent=1)
    print("\nresumen escrito en barrido_umbral.json")


if __name__ == "__main__":
    main()
