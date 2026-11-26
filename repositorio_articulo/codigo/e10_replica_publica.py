"""E10. Réplica en datos ajenos: "EMG elbow dataset" (Toro-Ossaba y Tejada, Zenodo, 2023,
Uso: python e10_replica_publica.py DIR_CONJUNTO SALIDA.json [FS]"""
import glob, json, os, sys, time
import numpy as np
from scipy import signal as sg, stats
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from common import modelo, SEMILLA

D, SAL = sys.argv[1], sys.argv[2]
FS = float(sys.argv[3]) if len(sys.argv) > 3 else 1024.0; WIN = int(round(0.3 * FS)); HOP = int(round(0.01 * FS))
CARGAS = [0, 1360, 2270]; N_PERM = 5


def filtrar(x):
    nyq = FS / 2
    b1, a1 = sg.butter(4, [20 / nyq, min(450.0, 0.96 * nyq) / nyq], btype="bandpass"); b2, a2 = sg.iirnotch(60 / nyq, 30)
    return sg.lfilter(b2, a2, sg.lfilter(b1, a1, x.astype(np.float64)))


def ventanas(x, n, h):
    x = np.ascontiguousarray(x); nv = (len(x) - n) // h + 1; s = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (nv, n), (h * s, s), writeable=False)


def caracteristicas(W):
    R = np.abs(W); fr = np.fft.rfftfreq(W.shape[1], 1 / FS); P = np.abs(np.fft.rfft(W, axis=1)) ** 2
    return np.c_[R.sum(1), (R * R).sum(1), np.abs(np.diff(R, axis=1)).sum(1), np.exp(np.log(R + 1e-6).mean(1)),
                 (np.diff(np.sign(W), axis=1) != 0).sum(1), (np.diff(np.sign(np.diff(W, axis=1)), axis=1) != 0).sum(1),
                 (P * fr).sum(1) / np.maximum(P.sum(1), 1e-12)]


def cargar_sujeto(s):
    X, y, reg, rel = [], [], [], []
    for ci, c in enumerate(CARGAS):
        for ri, conj in enumerate(("train", "test")):
            a = np.loadtxt(os.path.join(D, f"subject {s}", f"{s}_flex_{conj}_{c}.txt"))
            a = a[int(FS):]
            F = np.c_[caracteristicas(ventanas(filtrar(a[:, 0]), WIN, HOP)), caracteristicas(ventanas(filtrar(a[:, 1]), WIN, HOP))]
            F = np.sign(F) * np.log1p(np.abs(F) * 1e6)
            X.append(F); y.append(np.full(len(F), ci)); reg.append(np.full(len(F), ci * 2 + ri)); rel.append(np.arange(len(F)))
    return np.concatenate(X), np.concatenate(y), np.concatenate(reg), np.concatenate(rel)


def auc3(y, P):
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))


def aleatoria(X, y, mod, semilla=SEMILLA):
    P = np.zeros((len(y), 3))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=semilla).split(X, y):
        P[te] = modelo(mod).fit(X[tr], y[tr]).predict_proba(X[te])
    return auc3(y, P), float(accuracy_score(y, P.argmax(1)))


def por_registro(X, y, reg, mod):
    P = np.zeros((len(y), 3)); par = reg % 2
    for lado in (0, 1):
        tr, te = par == lado, par != lado
        P[te] = modelo(mod).fit(X[tr], y[tr]).predict_proba(X[te])
    return auc3(y, P), float(accuracy_score(y, P.argmax(1)))


res = dict(fs=FS, ventana_muestras=WIN, paso_muestras=HOP, sujetos={}); t0 = time.time()
rng = np.random.default_rng(SEMILLA)
for s in range(1, 11):
    X, y, reg, rel = cargar_sujeto(s); fila = dict(n_ventanas=int(len(y)))
    for mod in ("HGB", "LDA"):
        a, acc_a = aleatoria(X, y, mod); g, acc_g = por_registro(X, y, reg, mod)
        fila[mod] = dict(auc_aleatoria=a, exactitud_aleatoria=acc_a, auc_registro=g, exactitud_registro=acc_g)
    perm = []
    for _ in range(N_PERM):
        while True:
            nuevo = rng.permutation([0, 0, 1, 1, 2, 2])
            if not all(nuevo[2 * c] == nuevo[2 * c + 1] for c in range(3)): break
        perm.append(aleatoria(X, nuevo[reg], "HGB")[0])
    fila["permutacion_HGB_auc_aleatoria"] = perm
    sol = {}
    for k in (1, 3, 5, 10, 15, 30):
        v = []
        for fase in (0, 20, 40):
            tr = (rel % 60) == fase; te = (rel % 60) == (fase + k) % 60
            m = modelo("HGB").fit(X[tr], y[tr]); v.append(auc3(y[te], m.predict_proba(X[te])))
        sol[str(k * 10)] = float(np.mean(v))
    fila["solapamiento_HGB"] = sol
    res["sujetos"][str(s)] = fila
    print(f"[{time.time()-t0:5.0f}s] sujeto {s}: HGB aleatoria {fila['HGB']['auc_aleatoria']:.3f} registro {fila['HGB']['auc_registro']:.3f} | "
          f"LDA {fila['LDA']['auc_aleatoria']:.3f} / {fila['LDA']['auc_registro']:.3f} | perm {np.mean(perm):.3f} | solap {sol}", flush=True)
    json.dump(res, open(SAL, "w"), indent=1)
for mod in ("HGB", "LDA"):
    a = np.array([v[mod]["auc_aleatoria"] for v in res["sujetos"].values()]); g = np.array([v[mod]["auc_registro"] for v in res["sujetos"].values()])
    w = stats.wilcoxon(a, g, alternative="greater")
    res["resumen_" + mod] = dict(aleatoria_media=float(a.mean()), aleatoria_desv=float(a.std(ddof=1)), registro_media=float(g.mean()), registro_desv=float(g.std(ddof=1)),
                                 diferencia_mediana=float(np.median(a - g)), sujetos_con_aleatoria_mayor=int((a > g).sum()), wilcoxon_p_unilateral=float(w.pvalue))
    print(mod, res["resumen_" + mod])
json.dump(res, open(SAL, "w"), indent=1)
