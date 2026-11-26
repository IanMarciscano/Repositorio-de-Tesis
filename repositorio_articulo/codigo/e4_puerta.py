"""E4. Puerta de intención: agrupación por episodio frente a sesión excluida,
Uso: python e4_puerta.py puerta.npz salida.json [n_sorteos]"""
import json, sys, time
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

SEMILLA = 42
d = np.load(sys.argv[1], allow_pickle=True)
SAL = sys.argv[2]
N_SORTEOS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
X, y, g, ses = d["X"], d["y"], d["g"], d["ses"]
TR, tr_ep, tr_ses, t_rel = d["TR"], d["tr_ep"], d["tr_ses"], d["t_rel"]
sesiones = [str(s) for s in d["sesiones"]]
mu_ses, sd_ses = d["mu_ses"], d["sd_ses"]


def rf():
    return RandomForestClassifier(300, random_state=SEMILLA, n_jobs=1)


def estandarizar_por_sesion(A, s_idx):
    return (A - mu_ses[s_idx]) / (sd_ses[s_idx] + 1e-12)


def ic_episodios(y, p, g, n=2000):
    rng = np.random.default_rng(SEMILLA); eps = np.unique(g)
    idx = {e: np.flatnonzero(g == e) for e in eps}; v = []
    for _ in range(n):
        ii = np.concatenate([idx[e] for e in rng.choice(eps, len(eps))])
        v.append(roc_auc_score(y[ii], p[ii]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], float(np.mean(np.array(v) > 0.5))


def conglomerados(D, n_perm=2000):
    n = D.shape[0]; tcrit = stats.t.ppf(0.975, n - 1)

    def masas(M):
        t = M.mean(0) / (M.std(0, ddof=1) / np.sqrt(n) + 1e-12)
        sobre = t > tcrit; out = []; i = 0
        while i < len(t):
            if sobre[i]:
                j = i
                while j + 1 < len(t) and sobre[j + 1]: j += 1
                out.append((float(t[i:j + 1].sum()), i, j)); i = j + 1
            else:
                i += 1
        return out, t
    obs, t = masas(D)
    if not obs:
        return dict(masa=0.0, inicio_ms=None, fin_ms=None, p=1.0)
    m, i, j = max(obs)
    rng = np.random.default_rng(SEMILLA); nulo = []
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], n)[:, None]
        o, _ = masas(D * s); nulo.append(max(o)[0] if o else 0.0)
    nulo = np.array(nulo)
    return dict(masa=m, inicio_ms=int(t_rel[i]), fin_ms=int(t_rel[j]),
                p=float((1 + (nulo >= m).sum()) / (1 + n_perm)),
                nulo_p95=float(np.percentile(nulo, 95)), nulo_max=float(nulo.max()))


def trayectorias(divisor, grupos, tipificar=False):
    Xu = estandarizar_por_sesion(X, ses) if tipificar else X
    P = np.full(TR.shape[:2], np.nan); p_ven = np.zeros(len(y)); aucs = []
    for tr, te in divisor.split(Xu, y, grupos):
        m = rf().fit(Xu[tr], y[tr]); p_ven[te] = m.predict_proba(Xu[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p_ven[te]))
        for e in np.unique(g[te]):
            k = int(np.flatnonzero(tr_ep == e)[0]); T = TR[k]
            if tipificar: T = estandarizar_por_sesion(T, tr_ses[k])
            ok = ~np.isnan(T).any(1); P[k, ok] = m.predict_proba(T[ok])[:, 1]
    return P, p_ven, aucs


def analizar(P):
    out = {}
    for nombre, (a, b) in {"A": (-2000, -1000), "B": (-1000, -500), "C": (-900, -400)}.items():
        base = (t_rel >= a) & (t_rel < b)
        ok = ~np.isnan(P).any(1)
        Dm = P[ok] - P[ok][:, base].mean(1, keepdims=True)
        r = conglomerados(Dm); r["n_episodios"] = int(ok.sum()); out[nombre] = r
    out["prob_media"] = {int(t): float(np.nanmean(P[:, t_rel == t])) for t in
                         (-2000, -1500, -1000, -700, -500, -300, -200, -110, 0, 200, 400, 600)}
    out["curva_media"] = [float(v) for v in np.nanmean(P, 0)]
    out["curva_et"] = [float(v) for v in np.nanstd(P, 0, ddof=1) / np.sqrt((~np.isnan(P)).sum(0))]
    return out


t0 = time.time(); res = dict(n_episodios=int(len(np.unique(g))), n_ventanas=int(len(y)),
                             episodios_por_sesion={sesiones[i]: int(len(np.unique(g[ses == i]))) for i in range(len(sesiones))})

for nombre, divisor, grupos, tip in [
        ("episodio", GroupKFold(5), g, False),
        ("sesion", LeaveOneGroupOut(), ses, False),
        ("episodio_tipificada", GroupKFold(5), g, True),
        ("sesion_tipificada", LeaveOneGroupOut(), ses, True)]:
    P, p_ven, aucs = trayectorias(divisor, grupos, tip)
    ic, frac = ic_episodios(y, p_ven, g)
    r = dict(auc_media=float(np.mean(aucs)), auc_desv=float(np.std(aucs)), auc_pliegues=[float(a) for a in aucs],
             auc_agregada=float(roc_auc_score(y, p_ven)), ic95=ic, frac_sobre_azar=frac,
             auc_por_sesion={sesiones[i]: float(roc_auc_score(y[ses == i], p_ven[ses == i])) for i in range(len(sesiones))},
             trayectoria=analizar(P))
    res[nombre] = r
    print(f"[{time.time()-t0:5.0f}s] {nombre:22s} AUC {r['auc_media']:.4f} ± {r['auc_desv']:.4f}  agregada {r['auc_agregada']:.4f} "
          f"IC {ic}  | línea base B: inicio {r['trayectoria']['B']['inicio_ms']} ms, masa {r['trayectoria']['B']['masa']:.1f}, p {r['trayectoria']['B']['p']:.4f}", flush=True)
    json.dump(res, open(SAL, "w"), indent=1)

rng = np.random.default_rng(SEMILLA); K = [0, 1, 2, 3, 4]; curva = {}
for s in range(len(sesiones)):
    eps = np.unique(g[ses == s]); n_test = len(eps) - 4
    filas = {k: dict(transfer=[], solo_sesion=[]) for k in K}
    for rep in range(N_SORTEOS):
        perm = rng.permutation(eps); test_e = perm[:n_test]; pool = perm[n_test:]
        te = np.isin(g, test_e)
        for k in K:
            cal = np.isin(g, pool[:k])
            tr = (ses != s) | cal
            m = rf().fit(X[tr], y[tr]); filas[k]["transfer"].append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            if k >= 1:
                m = rf().fit(X[cal], y[cal]); filas[k]["solo_sesion"].append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    curva[sesiones[s]] = {str(k): dict(transfer_media=float(np.mean(v["transfer"])), transfer_desv=float(np.std(v["transfer"])),
                                       solo_media=float(np.mean(v["solo_sesion"])) if v["solo_sesion"] else None,
                                       solo_desv=float(np.std(v["solo_sesion"])) if v["solo_sesion"] else None)
                          for k, v in filas.items()}
    print(f"[{time.time()-t0:5.0f}s] calibración {sesiones[s]}: " + "  ".join(
        f"k={k}: {curva[sesiones[s]][str(k)]['transfer_media']:.3f}" for k in K), flush=True)
    res["curva_calibracion"] = curva; res["curva_calibracion_n_sorteos"] = N_SORTEOS
    json.dump(res, open(SAL, "w"), indent=1)
print("hecho en %.0f s" % (time.time() - t0))
