"""Auditoría A1. Error de Monte Carlo de los IC por remuestreo de bloques: 500 frente a 2000 réplicas."""
import json, sys
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from common import *
D, SAL = sys.argv[1:3]
df = cargar_todo(D, "N1"); X = df[EMG14].to_numpy(); y = df.y.to_numpy(); ses = df.sesion.to_numpy(); g = df.grupo.to_numpy()
p = np.full(len(y), np.nan)
for tr, te in LeaveOneGroupOut().split(X, y, ses):
    p[te] = modelo("LDA").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
r = dict(auc=auc_seguro(y, p))
for n in (500, 2000):
    ic = np.array([ic_bloques(y, p, g, n=n, semilla=s) for s in range(4)])
    r[str(n)] = dict(ic_por_semilla=ic.tolist(), de_limite_inferior=float(ic[:, 0].std()), de_limite_superior=float(ic[:, 1].std()))
json.dump(r, open(SAL, "w"), indent=1); print({k: (v if k == "auc" else (round(v["de_limite_inferior"], 4), round(v["de_limite_superior"], 4))) for k, v in r.items()})
