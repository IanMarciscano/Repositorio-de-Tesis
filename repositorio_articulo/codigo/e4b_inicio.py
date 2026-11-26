"""E4b (exploratorio, añadido tras ver E4). Incertidumbre del instante de divergencia bajo
Uso: python e4b_inicio.py puerta.npz salida.json"""
import json, sys
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

SEMILLA = 42
d = np.load(sys.argv[1], allow_pickle=True)
X, y, g, ses, TR, tr_ep, t_rel = d["X"], d["y"], d["g"], d["ses"], d["TR"], d["tr_ep"], d["t_rel"]


def tray(divisor, grupos):
    P = np.full(TR.shape[:2], np.nan)
    for tr, te in divisor.split(X, y, grupos):
        m = RandomForestClassifier(300, random_state=SEMILLA, n_jobs=1).fit(X[tr], y[tr])
        for e in np.unique(g[te]):
            k = int(np.flatnonzero(tr_ep == e)[0]); P[k] = m.predict_proba(TR[k])[:, 1]
    return P


def inicio(D):
    n = D.shape[0]; tcrit = stats.t.ppf(0.975, n - 1)
    t = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(n) + 1e-12); sobre = t > tcrit
    mejor = (0.0, None); i = 0
    while i < len(t):
        if sobre[i]:
            j = i
            while j + 1 < len(t) and sobre[j + 1]: j += 1
            m = t[i:j + 1].sum()
            if m > mejor[0]: mejor = (m, t_rel[i])
            i = j + 1
        else: i += 1
    return mejor[1]


Pe, Ps = tray(GroupKFold(5), g), tray(LeaveOneGroupOut(), ses)
np.savez_compressed(sys.argv[2].replace(".json", "_P.npz"), P_episodio=Pe, P_sesion=Ps, t_rel=t_rel, tr_ep=tr_ep, tr_ses=d["tr_ses"])
res = {}; rng = np.random.default_rng(SEMILLA); n = Pe.shape[0]
for lb, (a, b) in {"A": (-2000, -1000), "B": (-1000, -500), "C": (-900, -400)}.items():
    base = (t_rel >= a) & (t_rel < b)
    De, Ds = Pe - Pe[:, base].mean(1, keepdims=True), Ps - Ps[:, base].mean(1, keepdims=True)
    ie, is_, dif = [], [], []
    for _ in range(2000):
        ii = rng.integers(0, n, n); a1, a2 = inicio(De[ii]), inicio(Ds[ii])
        if a1 is not None: ie.append(a1)
        if a2 is not None: is_.append(a2)
        if a1 is not None and a2 is not None: dif.append(a2 - a1)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    res[lb] = dict(episodio=dict(puntual=None if inicio(De) is None else int(inicio(De)), ic=q(ie), frac_antes=float(np.mean(np.array(ie) < 0)), n_validos=len(ie)),
                   sesion=dict(puntual=None if inicio(Ds) is None else int(inicio(Ds)), ic=q(is_), frac_antes=float(np.mean(np.array(is_) < 0)), n_validos=len(is_)),
                   diferencia=dict(ic=q(dif), frac_positiva=float(np.mean(np.array(dif) > 0)), n_validos=len(dif)))
    print(lb, res[lb], flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
