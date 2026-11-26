"""E7. Simulación: bloques con etiquetas alternadas y características sin información de
Uso: python e7_simulacion.py SALIDA.json [repeticiones]"""
import json, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from common import modelo, auc_seguro, SEMILLA
SAL = sys.argv[1]; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_BLOQ, VENT, DIM, PASO_S = 40, 300, 14, 0.01


def sesion(rng, tau_s, alternadas):
    n = N_BLOQ * VENT
    phi = np.exp(-PASO_S / tau_s)
    z = np.zeros((n, DIM)); e = rng.standard_normal((n, DIM)); z[0] = e[0]
    for t in range(1, n): z[t] = phi * z[t - 1] + np.sqrt(1 - phi * phi) * e[t]
    w = rng.standard_normal((n + 29, DIM)); ruido = np.cumsum(w, 0); ruido = (ruido[30:] - ruido[:-30]) / np.sqrt(30)
    ruido = np.vstack([w[:1], ruido])[:n]
    X = z + ruido
    g = np.repeat(np.arange(N_BLOQ), VENT)
    etq = (np.arange(N_BLOQ) % 2) if alternadas else rng.permutation(np.arange(N_BLOQ) % 2)
    return X, etq[g], g


def evaluar(X, y, g, mod, semilla):
    out = {}
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=semilla).split(X, y):
        p[te] = modelo(mod).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out["aleatoria"] = auc_seguro(y, p)
    P = {0: np.zeros(len(y)), 1: np.zeros(len(y)), 2: np.zeros(len(y))}
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=semilla).split(X, y, g):
        bt = np.unique(g[te])
        for margen in (0, 1, 2):
            lejos = np.min(np.abs(g[tr][:, None] - bt[None, :]), axis=1) > margen
            P[margen][te] = modelo(mod).fit(X[tr][lejos], y[tr][lejos]).predict_proba(X[te])[:, 1]
    out["bloque"], out["purga1"], out["purga2"] = (auc_seguro(y, P[m]) for m in (0, 1, 2))
    return out


res = {}
for alternadas in (True, False):
    for tau in (2.0, 20.0, 200.0):
        for mod in ("HGB", "LDA"):
            clave = f"{'alternadas' if alternadas else 'azar'}|tau{tau:g}|{mod}"; filas = []
            for r in range(REPS):
                rng = np.random.default_rng(SEMILLA + r)
                X, y, g = sesion(rng, tau, alternadas); filas.append(evaluar(X, y, g, mod, SEMILLA + r))
            res[clave] = {k: [f[k] for f in filas] for k in filas[0]}
            print(clave, {k: f"{np.mean(v):.3f}±{np.std(v):.3f}" for k, v in res[clave].items()}, flush=True)
            json.dump(res, open(SAL, "w"), indent=1)
