"""E6. Solapamiento con N igualado: entrenamiento fijo (una ventana cada 600 ms por
Uso: python e6_solapamiento_n_fijo.py DIR_DATOS SALIDA.json"""
import json, sys
import numpy as np
from common import *
D, SAL = sys.argv[1], sys.argv[2]
df = cargar_todo(D, "N1")
df["rel"] = df["idx"] - df.groupby(["sesion", "bloque"])["idx"].transform("min")
DESPL = [10, 30, 50, 100, 150, 300]
CONFIG = [("EMG14", "HGB"), ("EMG14", "LDA"), ("MEC7", "HGB"), ("FUS15", "HGB")]
res = {}
for cj, mod in CONFIG:
    cols = CONJUNTOS[cj]; fila = {str(d): {} for d in DESPL}
    for s in SESIONES:
        d = df[df.sesion == s]
        for fase in range(0, 60, 10):
            tr = d[(d.rel % 60) == fase]
            m = modelo(mod).fit(tr[cols].to_numpy(), tr.y.to_numpy())
            for k in DESPL:
                te = d[(d.rel % 60) == (fase + k // 10) % 60]
                fila[str(k)].setdefault(s, []).append(auc_seguro(te.y.to_numpy(), m.predict_proba(te[cols].to_numpy())[:, 1]))
        fila["n_entrenamiento_" + s] = int(((d.rel % 60) == 0).sum())
    res[f"{cj}|{mod}"] = fila
    print(cj, mod, {k: round(float(np.mean([np.mean(v) for kk, v in fila[str(k)].items()])), 3) for k in DESPL}, flush=True)
json.dump(res, open(SAL, "w"), indent=1)
