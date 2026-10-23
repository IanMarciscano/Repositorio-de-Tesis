"""Compara las variantes de generalizacion entre si y con la configuracion base.

Escribe resumen_pruebas_mejora.json.
"""
import os
import glob
import json
import random
import traceback

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
_RAIZ = os.path.dirname(HERE)
DIR_SALIDA = os.path.join(_RAIZ, "pruebas de mejora")
os.makedirs(DIR_SALIDA, exist_ok=True)

# hiperparametros protegidos (identicos a los pasos 3 y 4)
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

AUM_COPIAS = 2       # T1, igual que en el paso 4
AUM_NODOS  = 4
AUM_SIGMA  = 0.25
JITTER_SIGMA = 0.05   # nuevo: desviacion del ruido aditivo (senal normalizada)

CONFIG = ("C", "fusion multimodal", slice(0, 15))
N_EMG  = 8

NOMBRES = ["reposo", "flex_codo", "ext_codo",
           "flex_hombro", "ext_hombro", "levantamiento"]

EXPERIMENTOS = ["focal", "auxint", "jitter", "coral"]


# Aumento T1 (identico al paso 4) y variante con jitter
def curva_warping(n_pasos, n_nodos, sigma, rng):
    from scipy.interpolate import CubicSpline
    x = np.linspace(0, n_pasos - 1, n_nodos + 2)
    y = rng.normal(1.0, sigma, n_nodos + 2)
    return CubicSpline(x, y)(np.arange(n_pasos))


def aumentar_magnitud(X, y, n_emg, copias, rng):
    if copias <= 0 or n_emg == 0:
        return X, y
    Xs, ys = [X], [y]
    for _ in range(copias):
        Xa = X.copy()
        for i in range(len(Xa)):
            g = curva_warping(X.shape[1], AUM_NODOS, AUM_SIGMA, rng)
            Xa[i, :, :n_emg] *= g[:, None]
        Xs.append(Xa.astype(np.float32))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def aumentar_jitter(X, y, n_emg, sigma, rng):
    """Una copia adicional con ruido gaussiano aditivo sobre el sEMG."""
    if n_emg == 0:
        return X, y
    Xa = X.copy()
    ruido = rng.normal(0.0, sigma, Xa[:, :, :n_emg].shape).astype(np.float32)
    Xa[:, :, :n_emg] += ruido
    return np.concatenate([X, Xa.astype(np.float32)]), np.concatenate([y, y])


# Arquitectura (Conv+BN+Conv+BN+LSTM+Atencion, igual al paso 4)
class AtencionAditiva(layers.Layer):
    def __init__(self, unidades=32, **kw):
        super().__init__(**kw)
        self.unidades = unidades

    def build(self, forma):
        self.W = layers.Dense(self.unidades, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)
        super().build(forma)

    def call(self, h):
        puntuacion = self.v(self.W(h))
        alfa = tf.nn.softmax(puntuacion, axis=1)
        return tf.reduce_sum(alfa * h, axis=1)

    def get_config(self):
        c = super().get_config(); c.update(unidades=self.unidades); return c


def construir_modelo(n_canales, con_aux=False):
    ent = layers.Input(shape=(L, n_canales))
    x = layers.Conv1D(32, 3, padding="same")(ent)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROP_1)(x)

    x = layers.Conv1D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROP_2)(x)

    x = layers.LSTM(48, return_sequences=True)(x)
    feats = AtencionAditiva(32, name="atencion")(x)
    h = layers.Dropout(DROP_3)(feats)
    h = layers.Dense(32, activation="relu")(h)
    sal = layers.Dense(N_CLASES, activation="softmax", name="clase")(h)

    salidas = [sal]
    if con_aux:
        aux = layers.Dense(1, activation="sigmoid", name="intencion")(h)
        salidas.append(aux)

    return keras.Model(ent, salidas, name="cnn_lstm_pruebas_{}".format(n_canales))


# FOCAL  --  Lin et al. (2017)
def perdida_focal(gamma=2.0, eps=1e-7):
    def loss(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        p_t = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        return -tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)   # sin reducir
    return loss


# CORAL  --  Sun y Saenko (2016)
def perdida_coral(feat_src, feat_tgt):
    src = feat_src - tf.reduce_mean(feat_src, axis=0, keepdims=True)
    tgt = feat_tgt - tf.reduce_mean(feat_tgt, axis=0, keepdims=True)
    ns = tf.maximum(tf.cast(tf.shape(src)[0] - 1, tf.float32), 1.0)
    nt = tf.maximum(tf.cast(tf.shape(tgt)[0] - 1, tf.float32), 1.0)
    cov_s = tf.matmul(src, src, transpose_a=True) / ns
    cov_t = tf.matmul(tgt, tgt, transpose_a=True) / nt
    d = tf.cast(tf.shape(src)[1], tf.float32)
    return tf.reduce_sum(tf.square(cov_s - cov_t)) / (4.0 * d * d)


def adaptar_coral(modelo, extractor, Xtr, ytr, Xte, pasos=200, lote=256,
                  lr=1e-4, lam=1.0):
    """
    Ajuste fino post-hoc breve: perdida de clasificacion sobre un lote de
    entrenamiento MAS alineacion CORAL entre sus caracteristicas y las de
    un lote del conjunto de prueba, sin usar las etiquetas de prueba.
    """
    opt = keras.optimizers.Adam(lr)
    rng = np.random.default_rng(SEMILLA)
    cce = keras.losses.SparseCategoricalCrossentropy()
    n_tr, n_te = len(Xtr), len(Xte)
    for _ in range(pasos):
        i_tr = rng.choice(n_tr, size=min(lote, n_tr), replace=False)
        i_te = rng.choice(n_te, size=min(lote, n_te), replace=False)
        xb, yb, xt = Xtr[i_tr], ytr[i_tr], Xte[i_te]
        with tf.GradientTape() as tape:
            feat_s = extractor(xb, training=True)
            feat_t = extractor(xt, training=True)
            pred = modelo(xb, training=True)
            l_cls = cce(yb, pred)
            l_coral = perdida_coral(feat_s, feat_t)
            l = l_cls + lam * l_coral
        grads = tape.gradient(l, modelo.trainable_variables)
        opt.apply_gradients(zip(grads, modelo.trainable_variables))
    return pasos


# Utilidades comunes
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
    from sklearn.metrics import roc_auc_score
    yb = (y > 0).astype(int)
    if len(np.unique(yb)) < 2:
        return float("nan")
    return roc_auc_score(yb, 1.0 - prob[:, 0])


def evaluar(modelo, Xte, yte, con_aux=False):
    pr = modelo.predict(Xte, batch_size=1024, verbose=0)
    if con_aux:
        pr = pr[0]
    pred = pr.argmax(1)
    ex = float((pred == yte).mean())
    f1 = float(f1_macro(yte, pred).mean())
    auc = float(auc_binaria(yte, pr))
    return ex, f1, auc, pr


def carga_de_ruta(ruta):
    return os.path.basename(ruta).replace("pliegue_", "").replace("kg.npz", "")


# Entrenamiento de un experimento en un pliegue
def entrenar_experimento(ruta, exp, mostrar=False):
    clave, desc, sel = CONFIG
    carga = carga_de_ruta(ruta)
    d = np.load(ruta, allow_pickle=True)
    Xtr, ytr = d["X_train"][:, :, sel], d["y_train"]
    Xva, yva = d["X_val"][:, :, sel],   d["y_val"]
    Xte, yte = d["X_test"][:, :, sel],  d["y_test"]
    n_can = Xtr.shape[2]

    print("\n" + "=" * 70)
    print("PLIEGUE {} kg  |  config C  |  experimento: {}".format(carga, exp))

    rng = np.random.default_rng(SEMILLA)
    Xtr_a, ytr_a = aumentar_magnitud(Xtr, ytr, N_EMG, AUM_COPIAS, rng)
    if exp == "jitter":
        Xtr_a, ytr_a = aumentar_jitter(Xtr_a, ytr_a, N_EMG, JITTER_SIGMA, rng)
    print("  aumento: {:,} -> {:,} secuencias".format(len(ytr), len(ytr_a)))

    con_aux = (exp == "auxint")
    modelo = construir_modelo(n_can, con_aux=con_aux)

    sw = None
    if exp == "focal":
        modelo.compile(optimizer=keras.optimizers.Adam(LR_INICIAL),
                       loss=perdida_focal(2.0), metrics=["accuracy"])
        cw = None
        y_fit, yv_fit = ytr_a, yva
        monitor_ckpt = "val_accuracy"
    elif exp == "auxint":
        modelo.compile(
            optimizer=keras.optimizers.Adam(LR_INICIAL),
            loss=["sparse_categorical_crossentropy", "binary_crossentropy"],
            loss_weights=[1.0, 0.3],
            metrics=[["accuracy"], []])
        cw = None
        pesos = pesos_de_clase(ytr_a)
        peso_clase = np.array([pesos[c] for c in ytr_a], dtype=np.float32)
        sw = [peso_clase, np.ones_like(peso_clase)]
        y_fit = [ytr_a, (ytr_a > 0).astype(np.float32)]
        yv_fit = [yva, (yva > 0).astype(np.float32)]
        monitor_ckpt = "val_clase_accuracy"
    else:  # jitter, coral: entrenamiento estandar
        modelo.compile(optimizer=keras.optimizers.Adam(LR_INICIAL),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cw = pesos_de_clase(ytr_a)
        y_fit, yv_fit = ytr_a, yva
        monitor_ckpt = "val_accuracy"

    if mostrar:
        modelo.summary()
    print("  parametros: {:,}".format(modelo.count_params()))

    ckpt = os.path.join(DIR_SALIDA, "modelo_{}kg_{}.keras".format(carga, exp))
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA_STOP,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=FACTOR_LR,
                                          patience=PACIENCIA_LR, min_lr=LR_MINIMA,
                                          verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt, monitor=monitor_ckpt, mode="max",
                                        save_best_only=True, verbose=0),
        keras.callbacks.CSVLogger(os.path.join(
            DIR_SALIDA, "historial_{}kg_{}.csv".format(carga, exp))),
        keras.callbacks.TensorBoard(log_dir=os.path.join(
            DIR_SALIDA, "tensorboard_logs", "{}kg_{}".format(carga, exp))),
    ]

    hist = modelo.fit(Xtr_a, y_fit, validation_data=(Xva, yv_fit),
                      epochs=EPOCAS_MAX, batch_size=LOTE, class_weight=cw,
                      sample_weight=sw, callbacks=cbs, shuffle=True, verbose=2)

    if not os.path.exists(ckpt):
        raise FileNotFoundError("No se guardo {}".format(ckpt))
    modelo = keras.models.load_model(
        ckpt, custom_objects={"AtencionAditiva": AtencionAditiva}, compile=False)

    ex0, f10, auc0, pr0 = evaluar(modelo, Xte, yte, con_aux=con_aux)
    resultado = dict(experimento=exp, carga_kg=float(carga), canales=int(n_can),
                     parametros=int(modelo.count_params()),
                     epocas=len(hist.history["loss"]),
                     exactitud=round(ex0, 4), f1_macro=round(f10, 4),
                     auc_intencion=round(auc0, 4))

    pred_final, prob_final = pr0.argmax(1), pr0
    if exp == "coral":
        extractor = keras.Model(modelo.input, modelo.get_layer("atencion").output)
        pasos = adaptar_coral(modelo, extractor, Xtr_a, ytr_a, Xte)
        ex1, f11, auc1, pr1 = evaluar(modelo, Xte, yte, con_aux=False)
        resultado.update(exactitud_coral=round(ex1, 4), f1_macro_coral=round(f11, 4),
                         auc_intencion_coral=round(auc1, 4), pasos_coral=pasos)
        pred_final, prob_final = pr1.argmax(1), pr1
        print("  antes de CORAL   : exactitud {:.1%}  F1 {:.3f}  AUC {:.3f}"
              .format(ex0, f10, auc0))
        print("  despues de CORAL : exactitud {:.1%}  F1 {:.3f}  AUC {:.3f}  ({} pasos)"
              .format(ex1, f11, auc1, pasos))
    else:
        print("  epocas {}   TEST exactitud {:.1%}  F1 {:.3f}  AUC intencion {:.3f}"
              .format(resultado["epocas"], ex0, f10, auc0))

    np.savez_compressed(
        os.path.join(DIR_SALIDA, "predicciones_{}kg_{}.npz".format(carga, exp)),
        y_true=yte, y_pred=pred_final, prob=prob_final.astype(np.float32))

    return resultado


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_SALIDA, "pliegue_*kg.npz")))
    if not rutas:
        raise FileNotFoundError(
            "No hay pliegue_*.npz en {}. Copia ahi los pliegues del paso 2."
            .format(DIR_SALIDA))

    print("PASO 5 (EXPLORATORIO)  PRUEBAS DE ALTERNATIVAS DE MEJORA")
    print("TensorFlow {} | semilla {} | solo configuracion C".format(
        tf.__version__, SEMILLA))
    print("experimentos: {}".format(", ".join(EXPERIMENTOS)))

    resultados = []
    primero = True
    for exp in EXPERIMENTOS:
        for ruta in rutas:
            try:
                r = entrenar_experimento(ruta, exp, mostrar=primero)
                primero = False
                resultados.append(r)
            except Exception as e:
                print("\n[FALLO] experimento {} pliegue {}: {}".format(
                    exp, os.path.basename(ruta), e))
                traceback.print_exc()
                resultados.append(dict(experimento=exp,
                                       carga_kg=float(carga_de_ruta(ruta)),
                                       error=str(e)))
            with open(os.path.join(DIR_SALIDA, "resumen_pruebas_mejora.json"),
                     "w") as fh:
                json.dump(dict(tensorflow=tf.__version__, semilla=SEMILLA,
                               detalle=resultados),
                          fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RESUMEN POR EXPERIMENTO (configuracion C, 3 pliegues)")
    print("  {:<10} {:>11} {:>10} {:>14}".format(
        "experimento", "exactitud", "F1 macro", "AUC intencion"))
    for exp in EXPERIMENTOS:
        R = [r for r in resultados if r.get("experimento") == exp and "error" not in r]
        if not R:
            print("  {:<10}   sin resultados validos (revisar fallos arriba)".format(exp))
            continue
        campo_ex = "exactitud_coral" if exp == "coral" and "exactitud_coral" in R[0] else "exactitud"
        campo_f1 = "f1_macro_coral" if exp == "coral" and "f1_macro_coral" in R[0] else "f1_macro"
        campo_auc = "auc_intencion_coral" if exp == "coral" and "auc_intencion_coral" in R[0] else "auc_intencion"
        e = [r[campo_ex] for r in R]; f = [r[campo_f1] for r in R]
        a = [r[campo_auc] for r in R]
        print("  {:<10} {:>6.1%} +-{:<3.1%} {:>6.3f} +-{:.2f} {:>9.3f} +-{:.2f}"
              .format(exp, np.mean(e), np.std(e), np.mean(f), np.std(f),
                      np.mean(a), np.std(a)))

    print("\n  referencias ya conocidas, configuracion C, mismos pliegues:")
    print("    paso 3 (sin tecnicas)        33.9 %  F1 0.286  AUC 0.565")
    print("    paso 4 T1+T3 sin adaptar     38.0 %  F1  n/d    AUC 0.613")
    print("    paso 4 T1+T2+T3 (AdaBN)      35.0 %  F1 0.308  AUC 0.586")
    print("\nresumen_pruebas_mejora.json escrito.")


if __name__ == "__main__":
    main()
