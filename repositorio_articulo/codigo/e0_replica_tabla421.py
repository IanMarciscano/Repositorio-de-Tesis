import sys, json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from common import *

D = sys.argv[1]
df = cargar_todo(D, "N1")
cols = CONJUNTOS["FUS15"]
res = {}
for s in SESIONES:
    d = df[df.sesion == s]
    X, y, g = d[cols].to_numpy(), d.y.to_numpy(), d.bloque.to_numpy()
    out = {}
    for nombre, cv, grupos in [("aleatoria", StratifiedKFold(5, shuffle=True, random_state=SEMILLA), None),
                               ("por_bloque", GroupKFold(5), g)]:
        p = np.zeros(len(y))
        for tr, te in cv.split(X, y, grupos):
            m = modelo("HGB").fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
        out[nombre] = auc_seguro(y, p)
    out["bloques"] = int(len(np.unique(g))); out["ventanas"] = int(len(y)); out["frac_gesto"] = float(y.mean())
    res[s] = out
    print(s, out, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
