"""E1. Escalera de particiones con tarea, modelo y características fijos.
Uso: python e1_escalera.py DIR_DATOS SALIDA.json [NORMALIZACION] [conjuntos] [modelos] [excluir_sesion]"""
import json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
from common import *

D, SAL = sys.argv[1], sys.argv[2]
NORM = sys.argv[3] if len(sys.argv) > 3 else "N1"
CONJ = sys.argv[4].split(",") if len(sys.argv) > 4 else list(CONJUNTOS)
MODS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["HGB", "LDA"]
EXCL = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] != "-" else []
REP_B = 5


def resumir(y, p, ses, grupo, pliegue):
    r = dict(auc=auc_seguro(y, p), bal_acc=exactitud_balanceada(y, p))
    r["auc_pliegues"] = [auc_seguro(y[pliegue == f], p[pliegue == f]) for f in np.unique(pliegue)]
    r["auc_media_pliegues"] = float(np.nanmean(r["auc_pliegues"]))
    r["auc_por_sesion"] = {s: auc_seguro(y[ses == s], p[ses == s]) for s in np.unique(ses)}
    r["auc_media_sesiones"] = float(np.nanmean(list(r["auc_por_sesion"].values())))
    r["ic95"] = ic_bloques(y, p, grupo, n=500)
    return r


def correr(df, cols, mod, esquema, semilla=SEMILLA):
    X = df[cols].to_numpy(); y = df.y.to_numpy()
    ses = df.sesion.to_numpy(); grupo = df.grupo.to_numpy(); carga = df.carga.to_numpy()
    if esquema == "A_aleatoria":
        divs = StratifiedKFold(5, shuffle=True, random_state=semilla).split(X, y)
    elif esquema == "B_bloque":
        divs = StratifiedGroupKFold(5, shuffle=True, random_state=semilla).split(X, y, grupo)
    elif esquema == "C_sesion":
        divs = LeaveOneGroupOut().split(X, y, ses)
    elif esquema == "D_carga":
        divs = LeaveOneGroupOut().split(X, y, carga)
    p = np.full(len(y), np.nan); pl = np.zeros(len(y), int)
    for f, (tr, te) in enumerate(divs):
        m = modelo(mod).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]; pl[te] = f
    return resumir(y, p, ses, grupo, pl)


def dentro_de_sesion(df, cols, mod, esquema, semilla):
    out = {}
    for s in df.sesion.unique():
        d = df[df.sesion == s]
        X = d[cols].to_numpy(); y = d.y.to_numpy(); g = d.grupo.to_numpy()
        if esquema == "A":
            divs = StratifiedKFold(5, shuffle=True, random_state=semilla).split(X, y)
        else:
            divs = StratifiedGroupKFold(5, shuffle=True, random_state=semilla).split(X, y, g)
        p = np.zeros(len(y)); aucs = []
        for tr, te in divs:
            m = modelo(mod).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
            aucs.append(auc_seguro(y[te], p[te]))
        out[s] = dict(auc=auc_seguro(y, p), auc_media_pliegues=float(np.nanmean(aucs)))
    return out


t0 = time.time()
df = cargar_todo(D, NORM)
if EXCL:
    df = df[~df.sesion.isin(EXCL)].reset_index(drop=True)
res = dict(normalizacion=NORM, excluidas=EXCL, n_ventanas=int(len(df)),
           n_bloques=int(df.grupo.nunique()), frac_gesto=float(df.y.mean()), resultados={})
for cj in CONJ:
    cols = CONJUNTOS[cj]
    for mod in MODS:
        clave = f"{cj}|{mod}"; r = {}
        r["A_aleatoria"] = correr(df, cols, mod, "A_aleatoria")
        reps = [correr(df, cols, mod, "B_bloque", SEMILLA + i) for i in range(REP_B)]
        r["B_bloque"] = reps[0]
        r["B_bloque"]["auc_repeticiones"] = [x["auc"] for x in reps]
        r["B_bloque"]["auc_media_sesiones_repeticiones"] = [x["auc_media_sesiones"] for x in reps]
        r["C_sesion"] = correr(df, cols, mod, "C_sesion")
        if df.carga.nunique() > 1:
            r["D_carga"] = correr(df, cols, mod, "D_carga")
        r["intra_A"] = dentro_de_sesion(df, cols, mod, "A", SEMILLA)
        intra_b = [dentro_de_sesion(df, cols, mod, "B", SEMILLA + i) for i in range(REP_B)]
        r["intra_B"] = {s: dict(auc_reps=[x[s]["auc"] for x in intra_b],
                                auc_media_pliegues_reps=[x[s]["auc_media_pliegues"] for x in intra_b])
                        for s in intra_b[0]}
        res["resultados"][clave] = r
        print(f"[{time.time()-t0:6.0f}s] {NORM} {clave:10s} A {r['A_aleatoria']['auc']:.4f} | "
              f"B {np.mean(r['B_bloque']['auc_repeticiones']):.3f} | C {r['C_sesion']['auc']:.3f} "
              f"(por sesión {r['C_sesion']['auc_media_sesiones']:.3f})"
              + (f" | D {r['D_carga']['auc']:.3f}" if "D_carga" in r else ""), flush=True)
        json.dump(res, open(SAL, "w"), indent=1)
print("hecho en %.0f s" % (time.time() - t0))
