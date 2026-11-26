"""E2. Descomposición de la fuga dentro de sesión.
Uso: python e2_fuga.py DIR_DATOS SALIDA.json [NORMALIZACION]"""
import json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold
from common import *

D, SAL = sys.argv[1], sys.argv[2]
NORM = sys.argv[3] if len(sys.argv) > 3 else "N1"
PASOS = [10, 30, 50, 100, 150, 300]
CONFIG = [("EMG14", "HGB"), ("EMG14", "LDA"), ("MEC7", "HGB"), ("FUS15", "HGB")]
CONFIG_PURGA = [("EMG14", "HGB"), ("EMG14", "LDA"), ("FUS15", "HGB")]
PERMS = {"EMG14|HGB": 20, "FUS15|HGB": 10}
N_PERM = 10
t0 = time.time()
df = cargar_todo(D, NORM)
res = dict(normalizacion=NORM, barrido_paso={}, division_intra_bloque={}, permutacion={}, purga_adyacentes={}, vecino_temporal={})


def cv_auc(X, y, g, esquema, mod, semilla=SEMILLA, n=5):
    if esquema == "A":
        divs = StratifiedKFold(n, shuffle=True, random_state=semilla).split(X, y)
    else:
        divs = StratifiedGroupKFold(n, shuffle=True, random_state=semilla).split(X, y, g)
    p = np.zeros(len(y))
    for tr, te in divs:
        p[te] = modelo(mod).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return auc_seguro(y, p)


for cj, mod in CONFIG:
    cols = CONJUNTOS[cj]; clave = f"{cj}|{mod}"; res["barrido_paso"][clave] = {}
    for paso in PASOS:
        k = paso // PASO_MS; fila = {}
        for s in SESIONES:
            d = df[(df.sesion == s) & (df.idx % k == 0)]
            X, y, g = d[cols].to_numpy(), d.y.to_numpy(), d.bloque.to_numpy()
            fila[s] = dict(n=int(len(y)), A=cv_auc(X, y, g, "A", mod),
                           B=[cv_auc(X, y, g, "B", mod, SEMILLA + i) for i in range(3)])
        res["barrido_paso"][clave][str(paso)] = fila
        print(f"[{time.time()-t0:5.0f}s] paso {paso:3d} ms {clave:10s} A {np.mean([v['A'] for v in fila.values()]):.4f}  "
              f"B {np.mean([np.mean(v['B']) for v in fila.values()]):.3f}", flush=True)
    json.dump(res, open(SAL, "w"), indent=1)

for cj, mod in CONFIG:
    cols = CONJUNTOS[cj]; fila = {}
    for s in SESIONES:
        d = df[df.sesion == s]
        rango = d.groupby("bloque")["idx"].transform(lambda v: (v - v.min()) / max(1, (v.max() - v.min())))
        corte_idx = d.groupby("bloque")["idx"].transform(lambda v: v.min() + 0.6 * (v.max() - v.min()))
        tr = (d.idx <= corte_idx).to_numpy(); te = (d.idx > corte_idx + 30).to_numpy()
        X, y = d[cols].to_numpy(), d.y.to_numpy()
        p = modelo(mod).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fila[s] = auc_seguro(y[te], p)
    res["division_intra_bloque"][f"{cj}|{mod}"] = fila
    print(f"[{time.time()-t0:5.0f}s] intra-bloque {cj}|{mod}: {np.mean(list(fila.values())):.4f}", flush=True)
json.dump(res, open(SAL, "w"), indent=1)

rng = np.random.default_rng(SEMILLA)
for cj, mod in [("EMG14", "HGB"), ("FUS15", "HGB")]:
    cols = CONJUNTOS[cj]; fila = {}
    for s in SESIONES:
        d = df[df.sesion == s]; X, g = d[cols].to_numpy(), d.bloque.to_numpy()
        bloques = np.unique(g); etq = np.array([d.y[d.bloque == b].iloc[0] for b in bloques])
        A, B = [], []
        for _ in range(PERMS[f"{cj}|{mod}"]):
            mapa = dict(zip(bloques, rng.permutation(etq))); yp = np.array([mapa[b] for b in g])
            A.append(cv_auc(X, yp, g, "A", mod)); B.append(cv_auc(X, yp, g, "B", mod))
        fila[s] = dict(A=A, B=B)
    res["permutacion"][f"{cj}|{mod}"] = fila
    print(f"[{time.time()-t0:5.0f}s] permutación {cj}: A {np.mean([np.mean(v['A']) for v in fila.values()]):.4f}  "
          f"B {np.mean([np.mean(v['B']) for v in fila.values()]):.3f}", flush=True)
    json.dump(res, open(SAL, "w"), indent=1)

for cj, mod in CONFIG_PURGA:
    cols = CONJUNTOS[cj]; fila = {}
    for s in SESIONES:
        d = df[df.sesion == s]; X, y, g = d[cols].to_numpy(), d.y.to_numpy(), d.bloque.to_numpy()
        centro = d.groupby("bloque")["idx"].mean()
        out = {0: [], 1: [], 2: []}; vec = []
        for i in range(2):
            p = {0: np.zeros(len(y)), 1: np.zeros(len(y)), 2: np.zeros(len(y))}; pv = np.zeros(len(y))
            for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=SEMILLA + i).split(X, y, g):
                bt = np.unique(g[te])
                for margen in (0, 1, 2):
                    lejos = np.min(np.abs(g[tr][:, None] - bt[None, :]), axis=1) > margen
                    if len(np.unique(y[tr][lejos])) < 2:
                        p[margen][te] = 0.5; continue
                    p[margen][te] = modelo(mod).fit(X[tr][lejos], y[tr][lejos]).predict_proba(X[te])[:, 1]
                if cj == "EMG14" and mod == "HGB":
                    btr = np.unique(g[tr]); ytr = np.array([y[tr][g[tr] == b][0] for b in btr])
                    for b in bt:
                        cerca = btr[np.argmin(np.abs(centro[btr].to_numpy() - centro[b]))]
                        pv[g == b] = ytr[btr == cerca][0]
            for margen in (0, 1, 2): out[margen].append(auc_seguro(y, p[margen]))
            if cj == "EMG14" and mod == "HGB": vec.append(auc_seguro(y, pv))
        fila[s] = {str(k): v for k, v in out.items()}
        if vec: res["vecino_temporal"][s] = vec
    res["purga_adyacentes"][f"{cj}|{mod}"] = fila
    print(f"[{time.time()-t0:5.0f}s] purga {cj}|{mod}: " + "  ".join(
        f"±{m}: {np.mean([np.mean(v[str(m)]) for v in fila.values()]):.3f}" for m in (0, 1, 2)), flush=True)
    json.dump(res, open(SAL, "w"), indent=1)
print("hecho en %.0f s" % (time.time() - t0))
