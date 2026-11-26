"""E9. CNN-LSTM de referencia de la tesis bajo distintas particiones.
Uso: python e9_cnn_lstm.py DIR_DATOS SALIDA.json CONFIG ESQUEMA [max_epocas]"""
import json, os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from common import *

D, SAL, CONFIG, ESQUEMA = sys.argv[1:5]
MAX_EPOCAS = int(sys.argv[5]) if len(sys.argv) > 5 else 100
SEM = int(sys.argv[6]) if len(sys.argv) > 6 else SEMILLA
L, LOTE, SAT = 20, 512, 10.0
CLASES = ["NULL"] + GESTOS
COMPLETAS = [s for s in SESIONES if s != "20260806_193017"]
PSI = EMG8 + MEC7
COLS = {"C": slice(0, 15), "A": slice(0, 8), "B": slice(8, 15)}[CONFIG]


def secuencias(clave):
    df = cargar_sesion(D, clave)
    X = df[PSI].to_numpy(dtype=np.float64)
    cal = (df.etiqueta == "calibracion").to_numpy()
    if cal.sum() == 0: cal = (df.etiqueta == "calibracion_con_carga").to_numpy()
    X[:, :8] = (X[:, :8] - X[cal, :8].mean(0)) / X[cal, :8].std(0)
    y = df.etiqueta.map({c: i for i, c in enumerate(CLASES)}).fillna(-1).astype(int).to_numpy()
    blk = df.bloque.to_numpy(); tsl = df.t_since_label_s.to_numpy(dtype=float)
    S = np.lib.stride_tricks.sliding_window_view(X, (L, X.shape[1])).squeeze(1)
    homog = blk[L - 1:] == blk[:len(blk) - L + 1]
    ok = homog & (y[L - 1:] >= 0)
    return S[ok].astype(np.float32), y[L - 1:][ok], blk[L - 1:][ok], tsl[L - 1:][ok]


def entrenable(y, tsl):
    return tsl >= np.where(y == 0, 2.0, 0.24)


def red(n_ent):
    m = keras.Sequential([
        layers.Input(shape=(L, n_ent)),
        layers.Conv1D(32, 3, padding="same", activation="relu"), layers.Dropout(0.2),
        layers.Conv1D(64, 3, padding="same", activation="relu"), layers.MaxPooling1D(2), layers.Dropout(0.3),
        layers.LSTM(48), layers.Dropout(0.3),
        layers.Dense(32, activation="relu"), layers.Dense(6, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


class MejorExactitud(keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.mejor, self.pesos, self.epoca = -1, None, 0
    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] > self.mejor:
            self.mejor, self.pesos, self.epoca = logs["val_accuracy"], self.model.get_weights(), epoch + 1


def metricas(y, P):
    pred = P.argmax(1); g = y > 0
    return dict(exactitud=float(accuracy_score(y, pred)), f1_macro=float(f1_score(y, pred, average="macro")),
                cond_gesto=float((P[g][:, 1:].argmax(1) + 1 == y[g]).mean()),
                base_mayoritaria=float(np.bincount(y, minlength=6).max() / len(y)),
                f1_base=float(f1_score(y, np.full(len(y), np.bincount(y).argmax()), average="macro")), n=int(len(y)))


def entrenar(Xtr, ytr, Xva, yva, Xte, yte, semilla=None):
    semilla = SEM if semilla is None else semilla
    keras.utils.set_random_seed(semilla)
    mec = Xtr[:, :, 8:].reshape(-1, 7); mu, sd = mec.mean(0), mec.std(0); sd = np.where(sd < 1e-9, 1.0, sd)

    def prep(A):
        A = A.copy(); A[:, :, 8:] = (A[:, :, 8:] - mu) / sd
        return np.clip(A, -SAT, SAT)[:, :, COLS].astype(np.float32)
    Xtr, Xva, Xte = prep(Xtr), prep(Xva), prep(Xte)
    m = red(Xtr.shape[2])
    if CONFIG == "C": assert m.count_params() == 31142, m.count_params()
    frec = np.bincount(ytr, minlength=6); pesos = {i: float(len(ytr) / (6 * max(1, frec[i]))) for i in range(6)}
    cb = MejorExactitud(); t0 = time.time()
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCAS, batch_size=LOTE, class_weight=pesos, verbose=0,
              callbacks=[cb, keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)])
    m.set_weights(cb.pesos)
    r = metricas(yte, m.predict(Xte, batch_size=2048, verbose=0))
    r.update(epocas=len(h.history["loss"]), epoca_elegida=cb.epoca, val_exactitud=float(cb.mejor), parametros=int(m.count_params()),
             n_entrenamiento=int(len(ytr)), n_validacion=int(len(yva)), minutos=round((time.time() - t0) / 60, 1))
    return r


datos = {}
for k in COMPLETAS:
    S, y, blk, tsl = secuencias(k); datos[k] = dict(X=S, y=y, blk=blk, tsl=tsl)
print({k: v["X"].shape for k, v in datos.items()}, flush=True)


def juntar(claves):
    X = np.concatenate([datos[k]["X"] for k in claves]); y = np.concatenate([datos[k]["y"] for k in claves])
    tsl = np.concatenate([datos[k]["tsl"] for k in claves])
    g = np.concatenate([np.char.add(k + "_", datos[k]["blk"].astype(str)) for k in claves])
    return X, y, tsl, g


def val_ultimo_bloque(claves):
    v = []
    for k in claves:
        y, blk = datos[k]["y"], datos[k]["blk"]; m = np.zeros(len(y), bool)
        for c in range(6):
            bs = np.unique(blk[y == c])
            if len(bs) > 1: m |= (blk == bs[-1]) & (y == c)
        v.append(m)
    return np.concatenate(v)


res = json.load(open(SAL)) if os.path.exists(SAL) else {}
clave = f"{CONFIG}|{ESQUEMA}" + ("" if SEM == SEMILLA else f"|s{SEM}"); pliegues = {}
if ESQUEMA in ("aleatoria", "bloque"):
    X, y, tsl, g = juntar(COMPLETAS); idx = np.arange(len(y))
    if ESQUEMA == "aleatoria":
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEM)
        tr, va = train_test_split(tr, test_size=0.125, stratify=y[tr], random_state=SEM)
    else:
        divs = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEM).split(X, y, g))
        te = divs[0][1]; va = divs[1][1]; tr = np.setdiff1d(idx, np.concatenate([te, va]))
    pod = entrenable(y, tsl); tr, va = tr[pod[tr]], va[pod[va]]
    pliegues["unico"] = entrenar(X[tr], y[tr], X[va], y[va], X[te], y[te])
    print(clave, pliegues["unico"], flush=True)
else:
    grupos = {k: [k] for k in COMPLETAS} if ESQUEMA == "sesion" else \
        {f"{c:.2f}kg": [k for k in COMPLETAS if CARGA[k] == c] for c in sorted(set(CARGA[k] for k in COMPLETAS))}
    for nombre, test_k in grupos.items():
        tren_k = [k for k in COMPLETAS if k not in test_k]
        X, y, tsl, _ = juntar(tren_k); es_val = val_ultimo_bloque(tren_k); pod = entrenable(y, tsl)
        Xte, yte, _, _ = juntar(test_k)
        pliegues[nombre] = entrenar(X[(~es_val) & pod], y[(~es_val) & pod], X[es_val & pod], y[es_val & pod], Xte, yte)
        print(clave, nombre, pliegues[nombre], flush=True)
        res[clave] = dict(pliegues=pliegues); json.dump(res, open(SAL, "w"), indent=1)
res = json.load(open(SAL)) if os.path.exists(SAL) else {}
res[clave] = dict(pliegues=pliegues,
                  media={k: float(np.mean([p[k] for p in pliegues.values()])) for k in ("exactitud", "f1_macro", "cond_gesto", "base_mayoritaria", "f1_base")},
                  desv={k: float(np.std([p[k] for p in pliegues.values()])) for k in ("exactitud", "f1_macro", "cond_gesto")})
json.dump(res, open(SAL, "w"), indent=1)
print("HECHO", clave, res[clave]["media"], flush=True)
