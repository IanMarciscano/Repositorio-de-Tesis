"""Entrena la CNN-LSTM y el estudio de ablacion entre modalidades.

Tres configuraciones, fusion multimodal, solo sEMG y solo exoesqueleto, por cada
pliegue. Escribe modelo_*.keras, historial_*.csv, predicciones_*.npz y
resumen_entrenamiento.json.
"""
import os
import glob
import json
import random

import numpy as np

# La semilla debe fijarse antes de importar TensorFlow.
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
DIR_DATOS  = os.path.join(_RAIZ, "data")
DIR_SALIDA = os.path.join(_RAIZ, "resultados")

if not os.path.isdir(DIR_DATOS):      # ejecucion en carpeta unica
    DIR_DATOS = HERE
if not os.path.isdir(DIR_SALIDA):
    DIR_SALIDA = HERE
os.makedirs(DIR_SALIDA, exist_ok=True)


# hiperparametros (v5, seccion 3.1)
L, F           = 20, 21
N_CLASES       = 6
LOTE           = 512
EPOCAS_MAX     = 100
LR_INICIAL     = 1e-3
PACIENCIA_STOP = 15
PACIENCIA_LR   = 4
FACTOR_LR      = 0.5
LR_MINIMA      = 1e-6

DROP_1, DROP_2, DROP_3 = 0.2, 0.3, 0.3

NOMBRES = ["reposo", "flex_codo", "ext_codo",
           "flex_hombro", "ext_hombro", "levantamiento"]

# Configuraciones del estudio de ablacion. El orden es deliberado: C
# primero para disponer del resultado principal cuanto antes.
CONFIGURACIONES = [
    ("C", "fusion multimodal", slice(0, 21)),
    ("A", "solo sEMG",         slice(0, 14)),
    ("B", "solo estado exo",   slice(14, 21)),
]

# Poner a False para entrenar unicamente la configuracion C.
EJECUTAR_ABLACION = True


def construir_modelo(n_canales=None):
    """
    Arquitectura literal de la v5, seccion 3.1.

    n_canales permite las configuraciones del estudio de ablacion. Con el
    valor por defecto (15) el modelo es exactamente el de la propuesta.
    """
    if n_canales is None:
        n_canales = F
    m = keras.Sequential([
        layers.Input(shape=(L, n_canales)),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.Dropout(DROP_1),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(DROP_2),
        layers.LSTM(48, return_sequences=False),
        layers.Dropout(DROP_3),
        layers.Dense(32, activation="relu"),
        layers.Dense(N_CLASES, activation="softmax"),
    ], name="cnn_lstm_agroarm1_{}c".format(n_canales))
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_INICIAL),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return m


def pesos_de_clase(y):
    """Ponderacion inversamente proporcional a la frecuencia (He y Garcia)."""
    n = np.bincount(y, minlength=N_CLASES).astype(np.float64)
    n[n == 0] = 1.0
    w = len(y) / (N_CLASES * n)
    w = w / w.mean()
    return {i: float(w[i]) for i in range(N_CLASES)}


def f1_macro(y, p):
    f = []
    for c in range(N_CLASES):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return np.array(f)


def entrenar(ruta, cfg, mostrar_resumen=False):
    clave, desc, sel = cfg
    carga = os.path.basename(ruta).replace("pliegue_", "").replace("kg.npz", "")
    d = np.load(ruta, allow_pickle=True)
    Xtr, ytr = d["X_train"][:, :, sel], d["y_train"]
    Xva, yva = d["X_val"][:, :, sel],   d["y_val"]
    Xte, yte = d["X_test"][:, :, sel],  d["y_test"]
    n_can = Xtr.shape[2]

    print("\n" + "=" * 70)
    print("PLIEGUE {} kg   |   configuracion {}: {} ({} canales)"
          .format(carga, clave, desc, n_can))
    print("  train {:>7,}   val {:>6,}   test {:>7,}"
          .format(len(ytr), len(yva), len(yte)))
    print("  sesiones train: {}".format(", ".join(d["sesiones_train"])))
    print("  sesiones test : {}".format(", ".join(d["sesiones_test"])))

    if Xtr.shape[1] != L:
        raise ValueError("Longitud de secuencia {}, se esperaba {}"
                         .format(Xtr.shape[1], L))
    for nombre, A in [("train", Xtr), ("val", Xva), ("test", Xte)]:
        if not np.isfinite(A).all():
            raise ValueError("Hay NaN o infinitos en {}".format(nombre))

    pesos = pesos_de_clase(ytr)
    print("  pesos de clase: {}".format(
        {NOMBRES[i]: round(v, 2) for i, v in pesos.items()}))

    modelo = construir_modelo(n_can)
    total = modelo.count_params()
    if mostrar_resumen:
        modelo.summary()
    print("  parametros entrenables: {:,}{}".format(
        total, "  (esperado 31.718 con psi en R^21)" if clave == "C" else ""))
    if clave == "C" and total != 31718:
        raise ValueError(
            "La configuracion C tiene {:,} parametros; con psi en R^21 se "
            "esperan 31.718. Revisa la arquitectura.".format(total))

    ruta_ckpt = os.path.join(DIR_SALIDA,
                             "modelo_{}kg_{}.keras".format(carga, clave))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PACIENCIA_STOP,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=FACTOR_LR, patience=PACIENCIA_LR,
            min_lr=LR_MINIMA, verbose=1),
        keras.callbacks.ModelCheckpoint(
            ruta_ckpt, monitor="val_accuracy", save_best_only=True,
            verbose=0),
        keras.callbacks.CSVLogger(
            os.path.join(DIR_SALIDA,
                         "historial_{}kg_{}.csv".format(carga, clave))),
    ]

    hist = modelo.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCAS_MAX, batch_size=LOTE,
        class_weight=pesos, callbacks=callbacks,
        shuffle=True, verbose=2)

    if not os.path.exists(ruta_ckpt):
        raise FileNotFoundError(
            "No se escribio {}. El entrenamiento no completo ninguna epoca."
            .format(ruta_ckpt))
    modelo = keras.models.load_model(ruta_ckpt)

    prob = modelo.predict(Xte, batch_size=1024, verbose=0)
    pred = prob.argmax(1)
    exact = float((pred == yte).mean())
    f1 = f1_macro(yte, pred)

    np.savez_compressed(
        os.path.join(DIR_SALIDA,
                     "predicciones_{}kg_{}.npz".format(carga, clave)),
        y_true=yte, y_pred=pred, prob=prob.astype(np.float32),
        t_since_label=d["t_since_label_test"],
        blk=d["blk_test"], sesion=d["sesion_test"])

    print("\n  epocas ejecutadas: {}".format(len(hist.history["loss"])))
    print("  TEST  exactitud {:.1%}   F1 macro {:.3f}".format(exact, f1.mean()))
    print("  F1 por clase: " + "  ".join(
        "{}={:.2f}".format(NOMBRES[i], f1[i]) for i in range(N_CLASES)))

    return dict(configuracion=clave, descripcion=desc, canales=int(n_can),
                parametros=int(total),
                carga_kg=float(carga), exactitud=round(exact, 4),
                f1_macro=round(float(f1.mean()), 4),
                f1_por_clase={NOMBRES[i]: round(float(f1[i]), 3)
                              for i in range(N_CLASES)},
                epocas=len(hist.history["loss"]),
                n_train=int(len(ytr)), n_test=int(len(yte)))


def volcar(resultados):
    """Escribe el resumen tras cada entrenamiento, para no perder trabajo."""
    por_cfg = {}
    for clave in {r["configuracion"] for r in resultados}:
        R = [r for r in resultados if r["configuracion"] == clave]
        e = [r["exactitud"] for r in R]
        f = [r["f1_macro"] for r in R]
        por_cfg[clave] = dict(
            descripcion=R[0]["descripcion"], canales=R[0]["canales"],
            parametros=R[0]["parametros"], pliegues=len(R),
            exactitud_media=round(float(np.mean(e)), 4),
            exactitud_desv=round(float(np.std(e)), 4),
            f1_macro_media=round(float(np.mean(f)), 4),
            f1_macro_desv=round(float(np.std(f)), 4))
    with open(os.path.join(DIR_SALIDA, "resumen_entrenamiento.json"), "w") as fh:
        json.dump(dict(tensorflow=tf.__version__, semilla=SEMILLA,
                       resumen_por_configuracion=por_cfg,
                       detalle=resultados),
                  fh, indent=2, ensure_ascii=False)


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_SALIDA, "pliegue_*kg.npz")))
    if not rutas:
        raise FileNotFoundError(
            "No hay pliegue_*.npz en {}. Ejecuta antes 02_construir_secuencias.py"
            .format(DIR_SALIDA))

    print("PASO 3  ENTRENAMIENTO CNN-LSTM")
    print("TensorFlow {} | semilla {} | {} pliegues"
          .format(tf.__version__, SEMILLA, len(rutas)))

    cfgs = CONFIGURACIONES if EJECUTAR_ABLACION else CONFIGURACIONES[:1]
    print("configuraciones: {}   ->  {} entrenamientos"
          .format(", ".join(c[0] for c in cfgs), len(cfgs) * len(rutas)))

    resultados = []
    primero = True
    for cfg in cfgs:
        for ruta in rutas:
            resultados.append(entrenar(ruta, cfg, mostrar_resumen=primero))
            primero = False
            # se vuelca el resumen tras cada entrenamiento, de modo que una
            # interrupcion no pierda lo ya calculado
            volcar(resultados)

    # tabla por configuracion: alimenta la tabla 4.2 del borrador
    print("\n" + "=" * 70)
    print("VALIDACION LEAVE-ONE-CONDITION-OUT")
    for clave, desc, _ in cfgs:
        R = [r for r in resultados if r["configuracion"] == clave]
        if not R:
            continue
        print("\n  Configuracion {}: {}  ({} canales, {:,} parametros)"
              .format(clave, desc, R[0]["canales"], R[0]["parametros"]))
        print("  {:<16} {:>10} {:>10} {:>8}".format(
            "carga excluida", "exactitud", "F1 macro", "epocas"))
        for r in R:
            print("  {:<16} {:>9.1%} {:>10.3f} {:>8}".format(
                "{:.2f} kg".format(r["carga_kg"]), r["exactitud"],
                r["f1_macro"], r["epocas"]))
        e = [r["exactitud"] for r in R]
        f = [r["f1_macro"] for r in R]
        print("  " + "-" * 46)
        print("  {:<16} {:>9.1%} {:>10.3f}".format(
            "media", np.mean(e), np.mean(f)))
        print("  {:<16} {:>9.1%} {:>10.3f}".format(
            "desv. tipica", np.std(e), np.std(f)))

    print("\n  " + "=" * 60)
    print("  RESUMEN COMPARATIVO (Tabla 4.2 del borrador)")
    print("  {:<26} {:>12} {:>12}".format(
        "configuracion", "exactitud", "F1 macro"))
    for clave, desc, _ in cfgs:
        R = [r for r in resultados if r["configuracion"] == clave]
        if not R:
            continue
        e = [r["exactitud"] for r in R]
        f = [r["f1_macro"] for r in R]
        print("  {:<26} {:>7.1%} +-{:<3.1%} {:>7.3f} +-{:.3f}".format(
            "{}. {}".format(clave, desc), np.mean(e), np.std(e),
            np.mean(f), np.std(f)))
    print("\n  azar con 6 clases: 16.7 %")

    print("\nresumen_entrenamiento.json escrito. Siguiente: 04_evaluar.py")


if __name__ == "__main__":
    main()
