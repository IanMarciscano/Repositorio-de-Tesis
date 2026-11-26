"""E5b. Sesión excluida (peldaño C) sin la sesión de IMU congelada en el entrenamiento."""
import json, sys, time
import numpy as np
from common import *
D, SAL = sys.argv[1], sys.argv[2]
df = cargar_todo(D, "N1"); S1 = "20260806_182858"; res = {}; t0 = time.time()
for cj in ("MEC7", "FUS15"):
    cols = CONJUNTOS[cj]; fila = {}
    for s in SESIONES:
        if s == S1: continue
        te = df[df.sesion == s]
        con = df[df.sesion != s]; sin = con[con.sesion != S1]
        fila[s] = dict(con_S1=auc_seguro(te.y.to_numpy(), modelo("HGB").fit(con[cols].to_numpy(), con.y.to_numpy()).predict_proba(te[cols].to_numpy())[:, 1]),
                       sin_S1=auc_seguro(te.y.to_numpy(), modelo("HGB").fit(sin[cols].to_numpy(), sin.y.to_numpy()).predict_proba(te[cols].to_numpy())[:, 1]))
        print(f"[{time.time()-t0:5.0f}s]", cj, s, fila[s], flush=True)
    res[cj] = fila; json.dump(res, open(SAL, "w"), indent=1)
