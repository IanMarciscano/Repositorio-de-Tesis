"""Intervalo de confianza del margen de anticipacion, por remuestreo de episodios.
"""
# -*- coding: utf-8 -*-
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

spec = importlib.util.spec_from_file_location(
    "p5", "/home/claude/work/05_puerta_intencion.py")
p5 = importlib.util.module_from_spec(spec)
sys.argv = ["05"]
spec.loader.exec_module(p5)
p5.DIR_DATOS = "/mnt/user-data/uploads/outputs/entrenamiento de la red/data"

PRE_MS, POST_MS = 2000, 500
BASE = (-1000, -500)
REJILLA = np.arange(-PRE_MS, POST_MS + 1, p5.HOP_MS)


def trayectorias():
    """Devuelve la matriz episodios x instantes de caracteristicas y el
    identificador de episodio de cada fila, sobre la rejilla comun."""
    import glob, os
    rutas = sorted(glob.glob(os.path.join(p5.DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    n = int(p5.FS * p5.WIN_MS / 1000)
    h = int(p5.FS * p5.HOP_MS / 1000)
    datos = []
    for si, r in enumerate(rutas):
        e = pd.read_csv(r, low_memory=False)
        lab = e["label"].fillna("NULL").values
        B = p5.filtrar(e["EMG_ch2_raw_bic"].values)
        T = p5.filtrar(e["EMG_ch3_raw_tri"].values)
        F = np.c_[p5.caracteristicas(p5.ventanas(B, n, h)),
                  p5.caracteristicas(p5.ventanas(T, n, h))]
        F = np.sign(F) * np.log1p(np.abs(F))
        fin = np.arange(len(F)) * h + n - 1
        v_el = np.abs(e["ElbowIMUOmega_dps"].values)
        v_sh = np.abs(e["ShoulderSpd_dps"].values)
        cortes = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        for k, i0 in enumerate(cortes):
            if lab[i0] not in p5.GESTOS or k == 0:
                continue
            i1 = cortes[k + 1] if k + 1 < len(cortes) else len(lab)
            v = v_el[i0:i1] if "codo" in lab[i0] else v_sh[i0:i1]
            tm = p5.primer_cruce(v, p5.UMBRAL_VEL, p5.PERSIST_MS)
            if tm is None or tm < p5.ANCHO_PRE + n:
                continue
            t0 = i0 + tm
            idx = np.searchsorted(fin, t0 + REJILLA)
            if idx[0] < 0 or idx[-1] >= len(F):
                continue
            datos.append((si * 1000 + k, F[idx]))
    return datos


def conglomerados(D, replicas=2000, semilla=20260917):
    """Contraste de conglomerados con permutacion de signo.

    D es episodios x instantes, ya restada la linea base. El estadistico por
    instante es el valor t de una muestra; los conglomerados son tramos
    contiguos que superan el umbral y su masa es la suma de los valores t.
    """
    ne = D.shape[0]
    def valores_t(M):
        m = M.mean(0)
        s = M.std(0, ddof=1) / np.sqrt(ne)
        return m / np.maximum(s, 1e-12)
    from scipy import stats
    umbral = stats.t.ppf(0.975, ne - 1)

    def masas(t):
        fuera = np.abs(t) > umbral
        salida, i = [], 0
        while i < len(fuera):
            if fuera[i]:
                j = i
                while j + 1 < len(fuera) and fuera[j + 1]:
                    j += 1
                salida.append((i, j, float(np.abs(t[i:j + 1]).sum())))
                i = j + 1
            else:
                i += 1
        return salida

    obs = masas(valores_t(D))
    rng = np.random.default_rng(semilla)
    nulo = np.zeros(replicas)
    for r in range(replicas):
        signos = rng.choice([-1.0, 1.0], size=(ne, 1))
        m = masas(valores_t(D * signos))
        nulo[r] = max([x[2] for x in m]) if m else 0.0
    return obs, nulo, umbral


def main():
    datos = trayectorias()
    print("episodios con trayectoria completa: %d" % len(datos))
    if len(datos) < 30:
        sys.exit("ABORTADO: muy pocos episodios")

    X, y, g, info = p5.construir()
    prob_modelo = {}
    gkf = GroupKFold(n_splits=5)
    eps_tray = np.array([d[0] for d in datos])
    for tr, te in gkf.split(X, y, g):
        m = RandomForestClassifier(300, random_state=p5.SEMILLA, n_jobs=-1)
        m.fit(X[tr], y[tr])
        for e in np.unique(g[te]):
            prob_modelo[e] = m
    print("modelos fuera de muestra disponibles para %d episodios"
          % len(prob_modelo))

    P, usados = [], []
    for ep, F in datos:
        if ep not in prob_modelo:
            continue
        P.append(prob_modelo[ep].predict_proba(F)[:, 1])
        usados.append(ep)
    P = np.array(P)
    print("trayectorias evaluadas fuera de muestra: %d x %d"
          % P.shape)

    base = (REJILLA >= BASE[0]) & (REJILLA < BASE[1])
    D = P - P[:, base].mean(1, keepdims=True)

    obs, nulo, umbral = conglomerados(D)
    if not obs:
        sys.exit("No se forma ningun conglomerado: el analisis no reproduce "
                 "el resultado del documento.")
    obs_orden = sorted(obs, key=lambda x: -x[2])
    i0, i1, masa = obs_orden[0]
    p = (1.0 + (nulo >= masa).sum()) / (1.0 + len(nulo))
    print("\nconglomerado mayor: de %d a %d ms, masa %.1f, p = %.4f"
          % (REJILLA[i0], REJILLA[i1], masa, p))
    print("maximo de la distribucion nula: %.1f" % nulo.max())
    print("umbral por instante (t de 0,975): %.3f" % umbral)

    divergencia = REJILLA[i0]
    print("\ninstante de divergencia reconstruido: %d ms" % divergencia)
    print("la tesis reporta: -110 ms")

    salida = dict(episodios=int(P.shape[0]), divergencia_ms=int(divergencia),
                  masa=round(float(masa), 1), p=round(float(p), 4),
                  nulo_max=round(float(nulo.max()), 1))

    # intervalo por remuestreo de episodios
    rng = np.random.default_rng(7)
    inicios = []
    from scipy import stats
    for _ in range(2000):
        sel = rng.integers(0, D.shape[0], D.shape[0])
        M = D[sel]
        ne = M.shape[0]
        t = M.mean(0) / np.maximum(M.std(0, ddof=1) / np.sqrt(ne), 1e-12)
        u = stats.t.ppf(0.975, ne - 1)
        fuera = np.abs(t) > u
        if not fuera.any():
            continue
        mejor, i = None, 0
        while i < len(fuera):
            if fuera[i]:
                j = i
                while j + 1 < len(fuera) and fuera[j + 1]:
                    j += 1
                masa_r = float(np.abs(t[i:j + 1]).sum())
                if mejor is None or masa_r > mejor[1]:
                    mejor = (i, masa_r)
                i = j + 1
            else:
                i += 1
        if mejor:
            inicios.append(REJILLA[mejor[0]])
    inicios = np.array(inicios)
    lo, hi = np.percentile(inicios, [2.5, 97.5])
    print("IC 95 %% del instante de divergencia: [%d, %d] ms  (%d replicas)"
          % (lo, hi, len(inicios)))
    print("fraccion de replicas con divergencia anterior al inicio mecanico: "
          "%.3f" % (inicios < 0).mean())
    salida.update(ic=[int(lo), int(hi)], replicas=int(len(inicios)),
                  fraccion_anticipa=round(float((inicios < 0).mean()), 3))
    json.dump(salida, open("/home/claude/work/ic_anticipacion.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
