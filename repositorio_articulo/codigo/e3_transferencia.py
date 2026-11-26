"""E3. Matriz de transferencia entre sesiones: entrenar en i, probar en j.
Uso: python e3_transferencia.py DIR_DATOS SALIDA.json [NORMALIZACION]"""
import json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from common import *

D, SAL = sys.argv[1], sys.argv[2]
NORM = sys.argv[3] if len(sys.argv) > 3 else "N1"
CONFIG = [("EMG8", "HGB"), ("EMG14", "HGB"), ("EMG14", "LDA"), ("MEC7", "HGB"), ("FUS15", "HGB")]
t0 = time.time(); df = cargar_todo(D, NORM); res = dict(normalizacion=NORM, sesiones=SESIONES, matrices={})
for cj, mod in CONFIG:
    cols = CONJUNTOS[cj]; M = np.zeros((5, 5))
    for i, si in enumerate(SESIONES):
        a = df[df.sesion == si]; Xa, ya, ga = a[cols].to_numpy(), a.y.to_numpy(), a.bloque.to_numpy()
        m = modelo(mod).fit(Xa, ya)
        for j, sj in enumerate(SESIONES):
            if i == j:
                v = []
                for r in range(5):
                    p = np.zeros(len(ya))
                    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA + r).split(Xa, ya, ga):
                        p[te] = modelo(mod).fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
                    v.append(auc_seguro(ya, p))
                M[i, j] = np.mean(v)
            else:
                b = df[df.sesion == sj]
                M[i, j] = auc_seguro(b.y.to_numpy(), m.predict_proba(b[cols].to_numpy())[:, 1])
    res["matrices"][f"{cj}|{mod}"] = M.tolist()
    fuera = M[~np.eye(5, dtype=bool)]
    print(f"[{time.time()-t0:5.0f}s] {cj}|{mod}: diagonal {np.diag(M).mean():.3f}  fuera {fuera.mean():.3f}  "
          f"(0 kg entre sí {np.mean([M[i,j] for i in range(3) for j in range(3) if i!=j]):.3f})", flush=True)
    json.dump(res, open(SAL, "w"), indent=1)
