"""Capa común de datos para los experimentos del artículo (E1–E5)."""
import glob
import os

import numpy as np
import pandas as pd

SEMILLA = 42
GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro", "levantamiento"]
AMP = ["IAV", "SSI", "WL", "LOG"]
ESP = ["ZC", "SSC", "MNF"]
EMG8 = [f"{a}_{m}" for m in ("bic", "tri") for a in AMP]
EMG14 = [f"{a}_{m}" for m in ("bic", "tri") for a in AMP + ESP]
MEC7 = ["q_sh", "q_el", "dq_sh", "dq_el", "i_sh", "F_el", "F_sh"]
CONJUNTOS = {"EMG8": EMG8, "EMG14": EMG14, "MEC7": MEC7, "FUS15": EMG8 + MEC7}
CARGA = {"20260806_182858": 0.00, "20260806_193017": 0.00, "20260807_150646": 0.00,
         "20260807_170550": 0.85, "20260811_122212": 0.50}
SESIONES = list(CARGA)
CORTA = {"20260806_182858": "S1 (0 kg, 6-ago, IMU)", "20260806_193017": "S2 (0 kg, 6-ago)",
         "20260807_150646": "S3 (0 kg, 7-ago)", "20260807_170550": "S4 (0,85 kg, 7-ago)",
         "20260811_122212": "S5 (0,50 kg, 11-ago)"}
PASO_MS = 10
VENTANA_S = 0.3


def cargar_sesion(dir_datos, clave):
    df = pd.read_csv(os.path.join(dir_datos, f"featok_{clave}.csv"), low_memory=False)
    df["etiqueta"] = df["label_str"].fillna("NULL").astype(str)
    df["bloque"] = (df["etiqueta"] != df["etiqueta"].shift()).cumsum().astype(int)
    df["idx"] = np.arange(len(df))
    df["sesion"] = clave
    return df


def normalizar(df, modo):
    out = df.copy()
    X = df[EMG14].to_numpy(dtype=float)
    if modo == "N1":
        cal = df["etiqueta"] == "calibracion"
        if cal.sum() == 0:
            cal = df["etiqueta"] == "calibracion_con_carga"
        mu = X[cal.to_numpy()].mean(0)
        sd = X[cal.to_numpy()].std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Z = np.clip((X - mu) / sd, -10, 10)
    else:
        L = pd.DataFrame(np.log1p(np.clip(X, 0, None)), columns=EMG14)
        if modo == "N0":
            Z = L.to_numpy()
        elif modo == "N2":
            mu = L.expanding(min_periods=500).mean().bfill()
            sd = L.expanding(min_periods=500).std(ddof=0).bfill()
            Z = ((L - mu) / (sd + 1e-12)).to_numpy()
        elif modo == "N3":
            mu = L.ewm(halflife=6000, adjust=False).mean()
            sd = np.sqrt(L.ewm(halflife=6000, adjust=False).var(bias=True))
            Z = ((L - mu) / (sd + 1e-9)).to_numpy().copy()
            Z[:50] = 0.0
        elif modo == "N4":
            med = L.median()
            iqr = (L.quantile(0.75) - L.quantile(0.25)) / 1.349
            Z = ((L - med) / (iqr + 1e-12)).to_numpy()
        else:
            raise ValueError(modo)
        Z = np.clip(Z, -10, 10)
    out[EMG14] = Z
    return out


def tarea_binaria(df):
    m = (df["etiqueta"].isin(GESTOS + ["NULL"])) & (df["t_since_label_s"] >= VENTANA_S)
    d = df.loc[m].copy()
    d["y"] = d["etiqueta"].isin(GESTOS).astype(int)
    return d


def cargar_todo(dir_datos, modo="N1"):
    partes = []
    for k in SESIONES:
        d = tarea_binaria(normalizar(cargar_sesion(dir_datos, k), modo))
        d["grupo"] = d["sesion"] + "_" + d["bloque"].astype(str)
        d["carga"] = CARGA[k]
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def modelo(nombre):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if nombre == "HGB":
        return HistGradientBoostingClassifier(random_state=SEMILLA)
    if nombre == "LDA":
        return make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    raise ValueError(nombre)


def auc_seguro(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def ic_bloques(y, p, grupos, n=1000, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    g, inv = np.unique(grupos, return_inverse=True)
    idx_por = [np.flatnonzero(inv == i) for i in range(len(g))]
    vals = []
    for _ in range(n):
        sel = rng.integers(0, len(g), len(g))
        ii = np.concatenate([idx_por[s] for s in sel])
        a = auc_seguro(y[ii], p[ii])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def exactitud_balanceada(y, p, umbral=0.5):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, (p >= umbral).astype(int)))
