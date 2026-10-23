"""Entrena las variantes con tecnicas de generalizacion.

Cabeza auxiliar de intencion, perdida focal, jitter, CORAL y renormalizacion por
lotes. Escribe un juego de modelos y predicciones por variante.
"""
import os
import glob
import json
import random

import numpy as np

SEMILLA = 42
os.environ["PYTHONHASHSEED"] = str(SEMILLA)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
random.seed(SEMILLA)
np.random.seed(SEMILLA)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEMILLA)

HERE = os.path.dirname(os.path.abspath(__file__))
_RAIZ      = os.path.dirname(HERE)
DIR_SALIDA = os.path.join(_RAIZ, "resultados")
if not os.path.isdir(DIR_SALIDA):
    DIR_SALIDA = HERE
os.makedirs(DIR_SALIDA, exist_ok=True)

# que tecnicas aplicar
TECNICAS = {
    "aumento":   True,   # T1  deformacion de magnitud
    "adabn":     True,   # T2  normalizacion por lotes adaptativa
    "atencion":  True,   # T3  atencion sobre la salida recurrente
}

# Configuraciones del estudio de ablacion, iguales al paso 3.
CONFIGURACIONES = [
    ("C", "fusion multimodal", slice(0, 15)),
    ("A", "solo sEMG",         slice(0, 8)),
    ("B", "solo estado exo",   slice(8, 15)),
]
EJECUTAR_ABLACION = True

# hiperparametros (identicos al paso 3)
L, F           = 20, 15
N_CLASES       = 6
LOTE           = 512
EPOCAS_MAX     = 100
LR_INICIAL     = 1e-3
PACIENCIA_STOP = 15
PACIENCIA_LR   = 4
FACTOR_LR      = 0.5
LR_MINIMA      = 1e-6
DROP_1, DROP_2, DROP_3 = 0.2, 0.3, 0.3

# t1: parametros del aumento
AUM_COPIAS = 2       # copias sinteticas por secuencia original
AUM_NODOS  = 4       # nodos de la spline
AUM_SIGMA  = 0.25    # desviacion de los nodos alrededor de 1

NOMBRES = ["reposo", "flex_codo", "ext_codo",
           "flex_hombro", "ext_hombro", "levantamiento"]


# T1  Deformacion de magnitud
def curva_warping(n_pasos, n_nodos, sigma, rng):
    """Curva suave de ganancia por interpolacion spline cubica."""
    from scipy.interpolate import CubicSpline
    x = np.linspace(0, n_pasos - 1, n_nodos + 2)
    y = rng.normal(1.0, sigma, n_nodos + 2)
    return CubicSpline(x, y)(np.arange(n_pasos))


def aumentar(X, y, n_canales_emg, copias, rng):
    """
    Devuelve el conjunto original mas 'copias' replicas con la amplitud
    de los canales electromiograficos deformada. Solo se deforman esos
    canales: las variables mecanicas no dependen de la ganancia del
    electrodo y deformarlas no tendria sentido fisico.
    """
    if copias <= 0 or n_canales_emg == 0:
        return X, y
    Xs, ys = [X], [y]
    for _ in range(copias):
        Xa = X.copy()
        for i in range(len(Xa)):
            g = curva_warping(X.shape[1], AUM_NODOS, AUM_SIGMA, rng)
            Xa[i, :, :n_canales_emg] *= g[:, None]
        Xs.append(Xa.astype(np.float32))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# T3  Atencion aditiva sobre la secuencia recurrente
class AtencionAditiva(layers.Layer):
    """
    Pondera los pasos de la secuencia recurrente. Para cada paso t
    calcula una puntuacion escalar, las normaliza con softmax y
    devuelve la suma ponderada de los estados ocultos.
    """
    def __init__(self, unidades=32, **kw):
        super().__init__(**kw)
        self.unidades = unidades

    def build(self, forma):
        self.W = layers.Dense(self.unidades, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)
        super().build(forma)

    def call(self, h):
        puntuacion = self.v(self.W(h))                 # (lote, pasos, 1)
        alfa = tf.nn.softmax(puntuacion, axis=1)
        return tf.reduce_sum(alfa * h, axis=1)         # (lote, unidades)

    def get_config(self):
        c = super().get_config(); c.update(unidades=self.unidades); return c


def construir_modelo(n_canales, usar_bn, usar_atencion):
    ent = layers.Input(shape=(L, n_canales))
    x = layers.Conv1D(32, 3, padding="same")(ent)
    if usar_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROP_1)(x)

    x = layers.Conv1D(64, 3, padding="same")(x)
    if usar_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROP_2)(x)

    x = layers.LSTM(48, return_sequences=usar_atencion)(x)
    if usar_atencion:
        x = AtencionAditiva(32)(x)
    x = layers.Dropout(DROP_3)(x)
    x = layers.Dense(32, activation="relu")(x)
    sal = layers.Dense(N_CLASES, activation="softmax")(x)

    m = keras.Model(ent, sal, name="cnn_lstm_mejorado")
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_INICIAL),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# T2  Readaptacion de las estadisticas de normalizacion
def adaptar_bn(modelo, X, lote=512):
    """
    Recorre X en modo entrenamiento para que las capas de normalizacion
    recalculen sus estadisticas moviles sobre el dominio destino. No se
    actualiza ningun peso: solo media y varianza moviles. No intervienen
    las etiquetas.
    """
    hay_bn = any(isinstance(c, layers.BatchNormalization) for c in modelo.layers)
    if not hay_bn:
        return 0
    for c in modelo.layers:
        if isinstance(c, layers.BatchNormalization):
            c.momentum = 0.9
    n = 0
    for i in range(0, len(X), lote):
        modelo(X[i:i + lote], training=True)
        n += 1
    return n


def pesos_de_clase(y):
    n = np.bincount(y, minlength=N_CLASES).astype(np.float64)
    n[n == 0] = 1.0
    w = len(y) / (N_CLASES * n)
    return {i: float(v) for i, v in enumerate(w / w.mean())}


def f1_macro(y, p):
    f = []
    for c in range(N_CLASES):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return np.array(f)


def auc_binaria(y, prob):
    """AUC de la deteccion de intencion: reposo frente a cualquier gesto."""
    from sklearn.metrics import roc_auc_score
    yb = (y > 0).astype(int)
    if len(np.unique(yb)) < 2:
        return float("nan")
    return roc_auc_score(yb, 1.0 - prob[:, 0])


def entrenar(ruta, cfg, sufijo, mostrar=False):
    clave, desc, sel = cfg
    carga = os.path.basename(ruta).replace("pliegue_", "").replace("kg.npz", "")
    d = np.load(ruta, allow_pickle=True)
    Xtr, ytr = d["X_train"][:, :, sel], d["y_train"]
    Xva, yva = d["X_val"][:, :, sel],   d["y_val"]
    Xte, yte = d["X_test"][:, :, sel],  d["y_test"]
    n_can = Xtr.shape[2]
    n_emg = min(8, n_can) if sel.start == 0 else 0

    print("\n" + "=" * 70)
    print("PLIEGUE {} kg  |  config {}: {}  |  tecnicas: {}"
          .format(carga, clave, desc, sufijo))

    n0 = len(ytr)
    if TECNICAS["aumento"]:
        rng = np.random.default_rng(SEMILLA)
        Xtr, ytr = aumentar(Xtr, ytr, n_emg, AUM_COPIAS, rng)
        print("  T1 aumento: {:,} -> {:,} secuencias ({} canales deformados)"
              .format(n0, len(ytr), n_emg))
    print("  train {:>8,}   val {:>6,}   test {:>7,}"
          .format(len(ytr), len(yva), len(yte)))

    modelo = construir_modelo(n_can, TECNICAS["adabn"], TECNICAS["atencion"])
    if mostrar:
        modelo.summary()
    print("  parametros entrenables: {:,}".format(modelo.count_params()))

    ckpt = os.path.join(DIR_SALIDA,
                        "modelo_{}kg_{}_{}.keras".format(carga, clave, sufijo))
    log_dir_tb = os.path.join(DIR_SALIDA, "tensorboard_logs",
                              "{}kg_{}_{}".format(carga, clave, sufijo))
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=PACIENCIA_STOP,
                                         restore_best_weights=True, verbose=1),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                             factor=FACTOR_LR,
                                             patience=PACIENCIA_LR,
                                             min_lr=LR_MINIMA, verbose=1),
           keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy",
                                           save_best_only=True, verbose=0),
           keras.callbacks.CSVLogger(os.path.join(
               DIR_SALIDA,
               "historial_{}kg_{}_{}.csv".format(carga, clave, sufijo))),
           # Solo monitoreo visual en vivo (tensorboard --logdir tensorboard_logs).
           # No participa en el entrenamiento ni afecta ningun resultado numerico.
           keras.callbacks.TensorBoard(log_dir=log_dir_tb)]

    hist = modelo.fit(Xtr, ytr, validation_data=(Xva, yva),
                      epochs=EPOCAS_MAX, batch_size=LOTE,
                      class_weight=pesos_de_clase(ytr), callbacks=cbs,
                      shuffle=True, verbose=2)

    if not os.path.exists(ckpt):
        raise FileNotFoundError("No se escribio {}".format(ckpt))
    modelo = keras.models.load_model(
        ckpt, custom_objects={"AtencionAditiva": AtencionAditiva})

    # antes de adaptar
    pr0 = modelo.predict(Xte, batch_size=1024, verbose=0)
    ex0 = float((pr0.argmax(1) == yte).mean())
    f10 = f1_macro(yte, pr0.argmax(1)).mean()
    auc0 = auc_binaria(yte, pr0)

    ex1, f11, auc1, nb = ex0, f10, auc0, 0
    if TECNICAS["adabn"]:
        nb = adaptar_bn(modelo, Xte)
        pr1 = modelo.predict(Xte, batch_size=1024, verbose=0)
        ex1 = float((pr1.argmax(1) == yte).mean())
        f11 = f1_macro(yte, pr1.argmax(1)).mean()
        auc1 = auc_binaria(yte, pr1)
        np.savez_compressed(os.path.join(
            DIR_SALIDA,
            "predicciones_{}kg_{}_{}.npz".format(carga, clave, sufijo)),
            y_true=yte, y_pred=pr1.argmax(1), prob=pr1.astype(np.float32),
            y_pred_sin_adabn=pr0.argmax(1))
    else:
        np.savez_compressed(os.path.join(
            DIR_SALIDA,
            "predicciones_{}kg_{}_{}.npz".format(carga, clave, sufijo)),
            y_true=yte, y_pred=pr0.argmax(1), prob=pr0.astype(np.float32))

    print("\n  epocas: {}".format(len(hist.history["loss"])))
    print("  TEST sin adaptar : exactitud {:.1%}  F1 {:.3f}  AUC intencion {:.3f}"
          .format(ex0, f10, auc0))
    if TECNICAS["adabn"]:
        print("  TEST con AdaBN   : exactitud {:.1%}  F1 {:.3f}  AUC intencion {:.3f}"
              "   ({} lotes)".format(ex1, f11, auc1, nb))

    return dict(configuracion=clave, descripcion=desc, tecnicas=sufijo,
                carga_kg=float(carga), canales=int(n_can),
                parametros=int(modelo.count_params()),
                epocas=len(hist.history["loss"]),
                exactitud=round(ex1, 4), f1_macro=round(float(f11), 4),
                auc_intencion=round(float(auc1), 4),
                exactitud_sin_adabn=round(ex0, 4),
                auc_sin_adabn=round(float(auc0), 4))


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_SALIDA, "pliegue_*kg.npz")))
    if not rutas:
        raise FileNotFoundError(
            "No hay pliegue_*.npz en {}. Ejecuta antes 02_construir_secuencias.py"
            .format(DIR_SALIDA))
    sufijo = "".join(k[0].upper() for k, v in TECNICAS.items() if v) or "BASE"

    print("PASO 4  ENTRENAMIENTO CON TECNICAS DE GENERALIZACION")
    print("TensorFlow {} | semilla {}".format(tf.__version__, SEMILLA))
    print("tecnicas activas: " + ", ".join(k for k, v in TECNICAS.items() if v))

    cfgs = CONFIGURACIONES if EJECUTAR_ABLACION else CONFIGURACIONES[:1]
    res, primero = [], True
    for cfg in cfgs:
        for r in rutas:
            res.append(entrenar(r, cfg, sufijo, mostrar=primero))
            primero = False
            with open(os.path.join(DIR_SALIDA,
                                   "resumen_mejorado_{}.json".format(sufijo)),
                      "w") as fh:
                json.dump(dict(tensorflow=tf.__version__, semilla=SEMILLA,
                               tecnicas=TECNICAS, detalle=res),
                          fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RESUMEN  (tecnicas: {})".format(sufijo))
    print("  {:<26} {:>11} {:>10} {:>14}".format(
        "configuracion", "exactitud", "F1 macro", "AUC intencion"))
    for clave, desc, _ in cfgs:
        R = [r for r in res if r["configuracion"] == clave]
        if not R:
            continue
        e = [r["exactitud"] for r in R]; f = [r["f1_macro"] for r in R]
        a = [r["auc_intencion"] for r in R]
        print("  {:<26} {:>6.1%} +-{:<3.1%} {:>6.3f} +-{:.2f} {:>9.3f} +-{:.2f}"
              .format("{}. {}".format(clave, desc), np.mean(e), np.std(e),
                      np.mean(f), np.std(f), np.mean(a), np.std(a)))
    print("\n  referencia del paso 3 (sin tecnicas):")
    print("    C fusion 33.9 %  F1 0.286  AUC 0.565")
    print("    A sEMG   19.3 %  F1 0.129  AUC 0.575")
    print("    B exo    36.3 %  F1 0.325  AUC 0.604")


if __name__ == "__main__":
    main()
