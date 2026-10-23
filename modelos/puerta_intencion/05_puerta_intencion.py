"""Primera etapa del modelo: puerta de intencion sobre la ventana de transicion.

Entrena un bosque aleatorio y una red convolucional-recurrente reducida con
validacion agrupada por episodio, y escribe puerta_intencion.json y .npz.
"""
import os
import glob
import json

import numpy as np
import pandas as pd
from scipy import signal as sg

SEMILLA = 42
np.random.seed(SEMILLA)

HERE = os.path.dirname(os.path.abspath(__file__))
_RAIZ      = os.path.dirname(HERE)
DIR_DATOS  = os.path.join(_RAIZ, "data")
DIR_SALIDA = os.path.join(_RAIZ, "resultados")
if not os.path.isdir(DIR_DATOS):  DIR_DATOS = HERE
if not os.path.isdir(DIR_SALIDA): DIR_SALIDA = HERE
os.makedirs(DIR_SALIDA, exist_ok=True)

# parametros
FS          = 1000.0
WIN_MS      = 300
HOP_MS      = 10
ANCHO_PRE   = 300     # ms de ventana de transicion (F.1)
SEP_REPOSO  = 1000    # ms de separacion de la clase de reposo
UMBRAL_VEL  = 5.0     # deg/s
PERSIST_MS  = 50

BP = (20.0, 450.0); NOTCH = (60.0, 30.0); ORDEN = 4
GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro", "levantamiento"]


def filtrar(x):
    nyq = FS / 2
    b1, a1 = sg.butter(ORDEN, [BP[0] / nyq, BP[1] / nyq], btype="bandpass")
    b2, a2 = sg.iirnotch(NOTCH[0] / nyq, NOTCH[1])
    return sg.lfilter(b2, a2, sg.lfilter(b1, a1, x.astype(np.float64)))


def ventanas(x, n, h):
    x = np.ascontiguousarray(x)
    nv = (len(x) - n) // h + 1
    s = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (nv, n), (h * s, s),
                                           writeable=False)


def caracteristicas(W):
    """Las cuatro de la seccion 2.4 mas tres del dominio de la frecuencia."""
    R = np.abs(W)
    fr = np.fft.rfftfreq(W.shape[1], 1 / FS)
    P = np.abs(np.fft.rfft(W, axis=1)) ** 2
    return np.c_[R.sum(1),
                 (R * R).sum(1),
                 np.abs(np.diff(R, axis=1)).sum(1),
                 np.exp(np.log(R + 1.0).mean(1)),
                 (np.diff(np.sign(W), axis=1) != 0).sum(1),
                 (np.diff(np.sign(np.diff(W, axis=1)), axis=1) != 0).sum(1),
                 (P * fr).sum(1) / np.maximum(P.sum(1), 1e-12)]

NOMBRES_CAR = ["IAV", "SSI", "WL", "LOG", "ZC", "SSC", "MNF"]


def primer_cruce(x, umbral, persist):
    s = (x > umbral).astype(int)
    if s.sum() < persist:
        return None
    c = np.convolve(s, np.ones(persist), "valid")
    i = np.flatnonzero(c >= persist)
    return int(i[0]) if len(i) else None


def construir():
    """Devuelve X, y, grupos (episodio) y metadatos de las sesiones."""
    rutas = sorted(glob.glob(os.path.join(DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    if not rutas:
        raise FileNotFoundError("No hay emg_*.csv en " + DIR_DATOS)
    n = int(FS * WIN_MS / 1000); h = int(FS * HOP_MS / 1000)
    X, y, g, info = [], [], [], []
    for si, r in enumerate(rutas):
        e = pd.read_csv(r, low_memory=False)
        lab = e["label"].fillna("NULL").values
        B = filtrar(e["EMG_ch2_raw_bic"].values)
        T = filtrar(e["EMG_ch3_raw_tri"].values)
        F = np.c_[caracteristicas(ventanas(B, n, h)),
                  caracteristicas(ventanas(T, n, h))]
        F = np.sign(F) * np.log1p(np.abs(F))
        fin = np.arange(len(F)) * h + n - 1
        v_el = np.abs(e["ElbowIMUOmega_dps"].values)
        v_sh = np.abs(e["ShoulderSpd_dps"].values)
        cortes = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        n_ep = 0
        for k, i0 in enumerate(cortes):
            if lab[i0] not in GESTOS or k == 0:
                continue
            i1 = cortes[k + 1] if k + 1 < len(cortes) else len(lab)
            v = v_el[i0:i1] if "codo" in lab[i0] else v_sh[i0:i1]
            tm = primer_cruce(v, UMBRAL_VEL, PERSIST_MS)
            if tm is None or tm < ANCHO_PRE + n:
                continue
            t0 = i0 + tm
            m = (fin >= t0 - ANCHO_PRE) & (fin < t0)
            j1 = i0 - SEP_REPOSO
            j0 = max(cortes[k - 1] + n, j1 - ANCHO_PRE)
            if j1 - j0 < n or m.sum() == 0:
                continue
            mr = (fin >= j0) & (fin < j1)
            if mr.sum() == 0:
                continue
            ep = si * 1000 + k
            X.append(F[m]);  y.append(np.ones(m.sum()));   g.append(np.full(m.sum(), ep))
            X.append(F[mr]); y.append(np.zeros(mr.sum())); g.append(np.full(mr.sum(), ep))
            n_ep += 1
        info.append((os.path.basename(r), n_ep))
    return (np.concatenate(X), np.concatenate(y).astype(int),
            np.concatenate(g), info)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    print("PASO 5  PUERTA DE INTENCION  (primera etapa)")
    print("ventana de transicion: {} ms previos al inicio mecanico"
          .format(ANCHO_PRE))

    X, y, g, info = construir()
    print("\nsesiones y episodios utiles:")
    for nom, ne in info:
        print("  {:<32} {:>3} episodios".format(nom, ne))
    print("\n  {:,} ventanas   clase 1 (intencion) {:,}   clase 0 (reposo) {:,}"
          .format(len(y), int(y.sum()), int((y == 0).sum())))
    print("  {} episodios independientes  |  {} caracteristicas"
          .format(len(np.unique(g)), X.shape[1]))
    if len(np.unique(g)) < 10:
        raise ValueError("Muy pocos episodios para validar.")

    # clasificadores
    resultados = {}
    prob_rf = np.zeros(len(y)); prob_nn = np.zeros(len(y))

    gkf = GroupKFold(n_splits=5)
    aucs_rf = []
    for tr, te in gkf.split(X, y, groups=g):
        m = RandomForestClassifier(300, random_state=SEMILLA, n_jobs=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        prob_rf[te] = p
        aucs_rf.append(roc_auc_score(y[te], p))
    resultados["bosque_aleatorio"] = dict(
        auc_media=round(float(np.mean(aucs_rf)), 4),
        auc_desv=round(float(np.std(aucs_rf)), 4),
        auc_agregada=round(float(roc_auc_score(y, prob_rf)), 4))
    print("\n  bosque aleatorio      AUC {:.3f} +- {:.3f}"
          .format(np.mean(aucs_rf), np.std(aucs_rf)))

    # red convolucional-recurrente sobre secuencias cortas
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        tf.random.set_seed(SEMILLA)

        aucs_nn = []
        for tr, te in gkf.split(X, y, groups=g):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr])[:, None, :].repeat(4, axis=1)
            Xte = sc.transform(X[te])[:, None, :].repeat(4, axis=1)
            m = keras.Sequential([
                layers.Input(shape=Xtr.shape[1:]),
                layers.Conv1D(16, 3, padding="same", activation="relu"),
                layers.Dropout(0.3),
                layers.LSTM(16),
                layers.Dropout(0.3),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid")])
            m.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="binary_crossentropy")
            m.fit(Xtr, y[tr], epochs=40, batch_size=128, verbose=0,
                  validation_split=0.15,
                  callbacks=[keras.callbacks.EarlyStopping(
                      patience=8, restore_best_weights=True)])
            p = m.predict(Xte, verbose=0).ravel()
            prob_nn[te] = p
            aucs_nn.append(roc_auc_score(y[te], p))
        resultados["red_neuronal"] = dict(
            auc_media=round(float(np.mean(aucs_nn)), 4),
            auc_desv=round(float(np.std(aucs_nn)), 4),
            auc_agregada=round(float(roc_auc_score(y, prob_nn)), 4))
        print("  red convolucional-recurrente  AUC {:.3f} +- {:.3f}"
              .format(np.mean(aucs_nn), np.std(aucs_nn)))
    except ImportError:
        print("  [aviso] TensorFlow no disponible: solo se reporta el bosque.")
        prob_nn = None

    # mejor de los dos para el barrido de umbral
    mejor = "bosque_aleatorio"
    prob = prob_rf
    if prob_nn is not None and \
       resultados["red_neuronal"]["auc_media"] > resultados["bosque_aleatorio"]["auc_media"]:
        mejor, prob = "red_neuronal", prob_nn
    print("\n  clasificador con mejor AUC: {}".format(mejor))

    print("\n  BARRIDO DEL UMBRAL theta_on  (fija el punto de operacion del DTS)")
    print("  {:>8} {:>12} {:>14} {:>12}".format(
        "theta_on", "sensibilidad", "especificidad", "falsos pos."))
    barrido = []
    for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pred = (prob >= th).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        espe = tn / (tn + fp) if tn + fp else 0.0
        barrido.append(dict(theta_on=th, sensibilidad=round(sens, 4),
                            especificidad=round(espe, 4), falsos_positivos=fp))
        print("  {:>8.2f} {:>11.1%} {:>13.1%} {:>12,}".format(th, sens, espe, fp))

    fpr, tpr, umb = roc_curve(y, prob)
    np.savez_compressed(os.path.join(DIR_SALIDA, "puerta_intencion.npz"),
                        y=y, prob_rf=prob_rf,
                        prob_nn=prob_nn if prob_nn is not None else np.array([]),
                        grupos=g, fpr=fpr, tpr=tpr, umbrales=umb)
    with open(os.path.join(DIR_SALIDA, "puerta_intencion.json"), "w") as f:
        json.dump(dict(ancho_pre_ms=ANCHO_PRE, sep_reposo_ms=SEP_REPOSO,
                       umbral_velocidad_dps=UMBRAL_VEL,
                       n_ventanas=int(len(y)), n_episodios=int(len(np.unique(g))),
                       sesiones=[{"archivo": a, "episodios": b} for a, b in info],
                       resultados=resultados, mejor=mejor, barrido=barrido),
                  f, indent=2, ensure_ascii=False)

    print("\n  referencia: la puerta del modelo de una sola etapa daba")
    print("  AUC 0.565 (fusion), 0.575 (solo sEMG) y 0.604 (solo exo).")
    print("\npuerta_intencion.json y puerta_intencion.npz escritos.")


if __name__ == "__main__":
    main()
