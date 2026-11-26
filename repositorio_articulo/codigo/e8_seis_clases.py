"""E8. Escalera de particiones con las seis clases de la tesis, cuatro sesiones completas.
Uso: python e8_seis_clases.py DIR_DATOS SALIDA.json"""
import json, sys, time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *

D, SAL = sys.argv[1], sys.argv[2]
CLASES = ["NULL"] + GESTOS
COMPLETAS = [s for s in SESIONES if s != "20260806_193017"]
partes = []
for k in COMPLETAS:
    d = normalizar(cargar_sesion(D, k), "N1")
    d = d[d.etiqueta.isin(CLASES) & (d.t_since_label_s >= VENTANA_S)].copy()
    d["y"] = d.etiqueta.map({c: i for i, c in enumerate(CLASES)}).astype(int)
    d["grupo"] = d.sesion + "_" + d.bloque.astype(str); d["carga"] = CARGA[k]; partes.append(d)
df = pd.concat(partes, ignore_index=True)


def mod6(nombre):
    if nombre == "HGB":
        return HistGradientBoostingClassifier(random_state=SEMILLA, class_weight="balanced")
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(6, 1 / 6)))


def metricas(y, P):
    pred = P.argmax(1); g = y > 0
    cond = (P[g][:, 1:].argmax(1) + 1 == y[g]).mean()
    return dict(exactitud=float(accuracy_score(y, pred)), f1_macro=float(f1_score(y, pred, average="macro")), cond_gesto=float(cond))


def correr(cols, nombre, esquema):
    X = df[cols].to_numpy(); y = df.y.to_numpy(); grupo = df.grupo.to_numpy()
    if esquema == "A": divs = StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(X, y)
    elif esquema == "B": divs = StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(X, y, grupo)
    elif esquema == "C": divs = LeaveOneGroupOut().split(X, y, df.sesion.to_numpy())
    else: divs = LeaveOneGroupOut().split(X, y, df.carga.to_numpy())
    P = np.zeros((len(y), 6)); pl = np.zeros(len(y), int); por = []
    for f, (tr, te) in enumerate(divs):
        m = mod6(nombre).fit(X[tr], y[tr]); pp = m.predict_proba(X[te])
        clases = m.classes_ if hasattr(m, "classes_") else m[-1].classes_
        P[np.ix_(te, clases)] = pp; pl[te] = f; por.append(metricas(y[te], P[te]))
    r = metricas(y, P)
    r["por_pliegue"] = por
    r["media_pliegues"] = {k: float(np.mean([p[k] for p in por])) for k in por[0]}
    r["desv_pliegues"] = {k: float(np.std([p[k] for p in por])) for k in por[0]}
    rng = np.random.default_rng(SEMILLA); g, inv = np.unique(grupo, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(g))]; pred = P.argmax(1); a, f1 = [], []
    for _ in range(500):
        ii = np.concatenate([idx[s] for s in rng.integers(0, len(g), len(g))])
        a.append(accuracy_score(y[ii], pred[ii])); f1.append(f1_score(y[ii], pred[ii], average="macro"))
    r["ic_exactitud"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    r["ic_f1"] = [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]
    return r


y = df.y.to_numpy(); may = np.zeros(len(y), int)
res = dict(n_ventanas=int(len(df)), n_bloques=int(df.grupo.nunique()), frac_reposo=float((y == 0).mean()),
           base_mayoritaria=dict(exactitud=float((y == 0).mean()), f1_macro=float(f1_score(y, may, average="macro")), cond_gesto=0.2),
           frac_por_clase={c: float((y == i).mean()) for i, c in enumerate(CLASES)}, resultados={})
t0 = time.time()
for cj in ("EMG8", "EMG14", "MEC7", "FUS15"):
    for nombre in ("HGB", "LDA"):
        r = {e: correr(CONJUNTOS[cj], nombre, e) for e in "ABCD"}
        res["resultados"][f"{cj}|{nombre}"] = r
        print(f"[{time.time()-t0:5.0f}s] {cj}|{nombre}: " + "  ".join(f"{e} ex {r[e]['exactitud']:.3f} F1 {r[e]['f1_macro']:.3f} cond {r[e]['cond_gesto']:.3f}" for e in "ABCD"), flush=True)
        json.dump(res, open(SAL, "w"), indent=1)
print("base mayoritaria", res["base_mayoritaria"])
