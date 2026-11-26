"""Auditoría A2. Matrices de confusión de seis clases (E8, refuerzo por gradiente), particiones A y C."""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from common import *
D, SAL = sys.argv[1:3]
CLASES = ["NULL"] + GESTOS; COMPLETAS = [s for s in SESIONES if s != "20260806_193017"]; partes = []
for k in COMPLETAS:
    d = normalizar(cargar_sesion(D, k), "N1")
    d = d[d.etiqueta.isin(CLASES) & (d.t_since_label_s >= VENTANA_S)].copy()
    d["y"] = d.etiqueta.map({c: i for i, c in enumerate(CLASES)}).astype(int); partes.append(d)
df = pd.concat(partes, ignore_index=True); y = df.y.to_numpy(); res = dict(clases=CLASES)
for cj in ("EMG14", "FUS15"):
    X = df[CONJUNTOS[cj]].to_numpy()
    for e in "AC":
        divs = StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(X, y) if e == "A" else LeaveOneGroupOut().split(X, y, df.sesion.to_numpy())
        pred = np.zeros(len(y), int)
        for tr, te in divs:
            pred[te] = HistGradientBoostingClassifier(random_state=SEMILLA, class_weight="balanced").fit(X[tr], y[tr]).predict(X[te])
        cm = confusion_matrix(y, pred, labels=range(6))
        res[f"{cj}|{e}"] = dict(cuentas=cm.tolist(), por_fila=(cm / cm.sum(1, keepdims=True)).round(4).tolist())
        print(cj, e, np.diag(cm / cm.sum(1, keepdims=True)).round(3), flush=True)
        json.dump(res, open(SAL, "w"), indent=1)
