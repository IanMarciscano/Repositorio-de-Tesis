"""Barrido del ancho de la ventana de transicion de la puerta de intencion.

Evalua seis anchuras con la misma particion agrupada por episodio. Escribe
barrido_ancho_ventana.json.
"""
# -*- coding: utf-8 -*-
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal as sg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DIR_DATOS = sys.argv[1] if len(sys.argv) > 1 else "data"
SEMILLA = 42

# identicos al paso 5
FS = 1000.0
WIN_MS = 300
HOP_MS = 10
SEP_REPOSO = 1000
UMBRAL_VEL = 5.0
PERSIST_MS = 50
BP = (20.0, 450.0)
NOTCH = (60.0, 30.0)
ORDEN = 4
GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro", "levantamiento"]

ANCHOS = [250, 300, 350, 400, 500, 750, 1000]


def filtrar(x):
    nyq = FS / 2
    b1, a1 = sg.butter(ORDEN, [BP[0] / nyq, BP[1] / nyq], btype="bandpass")
    b2, a2 = sg.iirnotch(NOTCH[0] / nyq, NOTCH[1])
    return sg.lfilter(b2, a2, sg.lfilter(b1, a1, x.astype(np.float64)))


def ventanas(x, n, h):
    x = np.ascontiguousarray(x)
    nv = (len(x) - n) // h + 1
    s = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (nv, n), (h * s, s),
                                           writeable=False)


def caracteristicas(V):
    R = np.abs(V)
    fr = np.fft.rfftfreq(V.shape[1], 1 / FS)
    P = np.abs(np.fft.rfft(V, axis=1)) ** 2
    return np.c_[R.sum(1),
                 (R * R).sum(1),
                 np.abs(np.diff(R, axis=1)).sum(1),
                 np.exp(np.log(R + 1.0).mean(1)),
                 (np.diff(np.sign(V), axis=1) != 0).sum(1),
                 (np.diff(np.sign(np.diff(V, axis=1)), axis=1) != 0).sum(1),
                 (P * fr).sum(1) / np.maximum(P.sum(1), 1e-12)]


def primer_cruce(x, umbral, persist):
    s = (x > umbral).astype(int)
    if s.sum() < persist:
        return None
    c = np.convolve(s, np.ones(persist), "valid")
    i = np.flatnonzero(c >= persist)
    return int(i[0]) if len(i) else None


def precalcular(rutas):
    """Filtrado y extraccion, que no dependen del ancho de la ventana de
    transicion y por tanto se hacen una sola vez."""
    n = int(FS * WIN_MS / 1000)
    h = int(FS * HOP_MS / 1000)
    cache = []
    for r in rutas:
        e = pd.read_csv(r, low_memory=False)
        lab = e["label"].fillna("NULL").values
        B = filtrar(e["EMG_ch2_raw_bic"].values)
        T = filtrar(e["EMG_ch3_raw_tri"].values)
        F = np.c_[caracteristicas(ventanas(B, n, h)),
                  caracteristicas(ventanas(T, n, h))]
        F = np.sign(F) * np.log1p(np.abs(F))
        fin = np.arange(len(F)) * h + n - 1
        cache.append(dict(
            lab=lab, F=F, fin=fin,
            v_el=np.abs(e["ElbowIMUOmega_dps"].values),
            v_sh=np.abs(e["ShoulderSpd_dps"].values),
            nombre=os.path.basename(r)))
        print("   leida %s  %d ventanas" % (os.path.basename(r), len(F)))
    return cache, n


def construir(cache, n, ancho_pre):
    X, y, g, por_sesion = [], [], [], []
    for si, c in enumerate(cache):
        lab, F, fin = c["lab"], c["F"], c["fin"]
        cortes = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        n_ep = 0
        for k, i0 in enumerate(cortes):
            if lab[i0] not in GESTOS or k == 0:
                continue
            i1 = cortes[k + 1] if k + 1 < len(cortes) else len(lab)
            v = c["v_el"][i0:i1] if "codo" in lab[i0] else c["v_sh"][i0:i1]
            tm = primer_cruce(v, UMBRAL_VEL, PERSIST_MS)
            if tm is None or tm < ancho_pre + n:
                continue
            t0 = i0 + tm
            m = (fin >= t0 - ancho_pre) & (fin < t0)
            j1 = i0 - SEP_REPOSO
            j0 = max(cortes[k - 1] + n, j1 - ancho_pre)
            if j1 - j0 < n or m.sum() == 0:
                continue
            mr = (fin >= j0) & (fin < j1)
            if mr.sum() == 0:
                continue
            ep = si * 1000 + k
            X.append(F[m]);  y.append(np.ones(m.sum()));   g.append(np.full(m.sum(), ep))
            X.append(F[mr]); y.append(np.zeros(mr.sum())); g.append(np.full(mr.sum(), ep))
            n_ep += 1
        por_sesion.append((c["nombre"], n_ep))
    if not X:
        return None, None, None, por_sesion
    return (np.concatenate(X), np.concatenate(y).astype(int),
            np.concatenate(g), por_sesion)


def evaluar(X, y, g):
    gkf = GroupKFold(n_splits=5)
    prob = np.zeros(len(y))
    aucs = []
    for tr, te in gkf.split(X, y, groups=g):
        m = RandomForestClassifier(300, random_state=SEMILLA, n_jobs=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        prob[te] = p
        aucs.append(roc_auc_score(y[te], p))
    return (float(np.mean(aucs)), float(np.std(aucs)),
            float(roc_auc_score(y, prob)))


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    if not rutas:
        sys.exit("No hay emg_*.csv en " + DIR_DATOS)
    print("Precalculo de caracteristicas")
    cache, n = precalcular(rutas)

    print()
    print("%-8s %10s %10s %10s %10s %10s" %
          ("ancho", "episodios", "ventanas", "AUC media", "desv", "AUC agreg"))
    salida = []
    for ancho in ANCHOS:
        X, y, g, por_sesion = construir(cache, n, ancho)
        if X is None:
            print("%5d ms %10d %10d   %s" % (ancho, 0, 0,
                  "sin episodios utiles: la ventana de reposo no alcanza "
                  "la longitud de una ventana"))
            salida.append(dict(ancho_ms=ancho, n_episodios=0, n_ventanas=0,
                               motivo="ancho_pre menor que la ventana de "
                                      "300 ms; la clase de reposo no cabe"))
            continue
        ne = len(np.unique(g))
        media, desv, agreg = evaluar(X, y, g)
        print("%5d ms %10d %10d %10.4f %10.4f %10.4f"
              % (ancho, ne, len(y), media, desv, agreg))
        salida.append(dict(ancho_ms=ancho, n_episodios=int(ne),
                           n_ventanas=int(len(y)),
                           auc_media=round(media, 4),
                           auc_desv=round(desv, 4),
                           auc_agregada=round(agreg, 4),
                           por_sesion=[{"archivo": a, "episodios": b}
                                       for a, b in por_sesion]))

    ref = [s for s in salida if s["ancho_ms"] == 300][0]
    print()
    print("VALIDACION contra puerta_intencion.json (300 ms)")
    esperado = dict(n_episodios=61, n_ventanas=3660,
                    auc_media=0.6323, auc_agregada=0.6344)
    ok = True
    for k, v in esperado.items():
        got = ref[k]
        bien = (abs(got - v) < 1e-4 if isinstance(v, float) else got == v)
        ok = ok and bien
        print("   %-14s esperado %-8s obtenido %-8s  %s"
              % (k, v, got, "coincide" if bien else "NO COINCIDE"))
    if not ok:
        print()
        print("   AVISO: el barrido NO es comparable con el resultado")
        print("   almacenado. No trasladar estas cifras al documento.")

    with open("barrido_ancho_ventana.json", "w", encoding="utf-8") as fh:
        json.dump(dict(validacion_ok=ok, barrido=salida), fh,
                  indent=1, ensure_ascii=False)
    print()
    print("Escrito barrido_ancho_ventana.json")


if __name__ == "__main__":
    main()
