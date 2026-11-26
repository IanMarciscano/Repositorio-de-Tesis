"""Auditoría A4. ¿Los bloques vecinos son independientes? Sensibilidad de los IC de E1."""
import json, sys, time
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from common import *
D, SAL = sys.argv[1:3]; NREP = 1000
df = cargar_todo(D, "N1"); y = df.y.to_numpy(); ses = df.sesion.to_numpy(); blk = df.bloque.to_numpy(); g = df.grupo.to_numpy()
ug, inv = np.unique(g, return_inverse=True); idx_g = [np.flatnonzero(inv == i) for i in range(len(ug))]
orden = {s: [i for i in sorted(range(len(ug)), key=lambda i: blk[idx_g[i][0]]) if ses[idx_g[i][0]] == s] for s in np.unique(ses)}


def ic_moviles(p, L, rng):
    vals = []
    for _ in range(NREP):
        sel = []
        for s, bl in orden.items():
            nb = len(bl); tom = []
            while len(tom) < nb:
                i0 = rng.integers(nb); tom += [bl[(i0 + k) % nb] for k in range(L)]
            sel += tom[:nb]
        ii = np.concatenate([idx_g[i] for i in sel]); vals.append(auc_seguro(y[ii], p[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def autocorr(p):
    r = []
    for s, bl in orden.items():
        m = np.array([p[idx_g[i]].mean() for i in bl]); c = np.array([y[idx_g[i][0]] for i in bl])
        e = m.copy()
        for k in (0, 1): e[c == k] -= m[c == k].mean()
        r.append(np.corrcoef(e[:-1], e[1:])[0, 1])
    return float(np.mean(r)), [float(x) for x in r]


res = dict(n_rep=NREP, casos={}); t0 = time.time()
for cj, mod in (("EMG14", "HGB"), ("EMG14", "LDA"), ("MEC7", "HGB"), ("FUS15", "HGB")):
    X = df[CONJUNTOS[cj]].to_numpy()
    for esq, grp in (("C", ses), ("D", df.carga.to_numpy())):
        p = np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, y, grp):
            p[te] = modelo(mod).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        rng = np.random.default_rng(SEMILLA); r = dict(auc=auc_seguro(y, p))
        r["ic_independiente_E1"] = ic_bloques(y, p, g, n=NREP)
        for L in (1, 2, 4, 8): r[f"ic_moviles_L{L}"] = ic_moviles(p, L, rng)
        r["autocorr_rezago1_media"], r["autocorr_por_sesion"] = autocorr(p)
        res["casos"][f"{cj}|{mod}|{esq}"] = r
        print(f"[{time.time()-t0:5.0f}s] {cj}|{mod}|{esq} AUC {r['auc']:.3f} ind {np.round(r['ic_independiente_E1'],3)} "
              + " ".join(f"L{L} {np.round(r[f'ic_moviles_L{L}'],3)}" for L in (1, 2, 4, 8)) + f" ac1 {r['autocorr_rezago1_media']:.2f}", flush=True)
        json.dump(res, open(SAL, "w"), indent=1)
