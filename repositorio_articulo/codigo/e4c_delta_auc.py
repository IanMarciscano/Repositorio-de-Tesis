"""E4c (exploratorio). Diferencia de AUC entre validación por episodio y por sesión excluida,"""
import json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
SEMILLA = 42
d = np.load(sys.argv[1], allow_pickle=True); X, y, g, ses = d["X"], d["y"], d["g"], d["ses"]
sesiones = [str(s) for s in d["sesiones"]]

def oof(divisor, grupos):
    p = np.zeros(len(y))
    for tr, te in divisor.split(X, y, grupos):
        p[te] = RandomForestClassifier(300, random_state=SEMILLA, n_jobs=1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p

pe, ps = oof(GroupKFold(5), g), oof(LeaveOneGroupOut(), ses)
rng = np.random.default_rng(SEMILLA); eps = np.unique(g); idx = {e: np.flatnonzero(g == e) for e in eps}

def boot(mask_eps):
    E = eps[mask_eps]; v = []
    for _ in range(2000):
        ii = np.concatenate([idx[e] for e in rng.choice(E, len(E))])
        v.append(roc_auc_score(y[ii], pe[ii]) - roc_auc_score(y[ii], ps[ii]))
    m = np.isin(g, E)
    return dict(auc_episodio=float(roc_auc_score(y[m], pe[m])), auc_sesion=float(roc_auc_score(y[m], ps[m])),
                delta=float(roc_auc_score(y[m], pe[m]) - roc_auc_score(y[m], ps[m])),
                ic95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], frac_positiva=float(np.mean(np.array(v) > 0)))

res = {"todas": boot(np.ones(len(eps), bool))}
for i, s in enumerate(sesiones):
    res["sin_" + s] = boot((eps // 1000) != i); res["solo_" + s] = boot((eps // 1000) == i)
for k, v in res.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in v.items()})
json.dump(res, open(sys.argv[2], "w"), indent=1)
