"""Mide el tiempo de computo de un ciclo de la cadena supervisora.

Cronometra el acondicionamiento, la extraccion de caracteristicas y la inferencia.
Escribe latencia_computo.json.
"""
# -*- coding: utf-8 -*-
import json
import os
import sys
import time

import numpy as np
from scipy import signal as sg

# constantes, identicas a 01_extraer_features.py y 02_construir_secuencias.py
FS_EMG = 1000.0
WIN_SAMPLES = 300          # T_w = 300 ms
HOP_SAMPLES = 10           # T_s = 10 ms
BP_LOW_HZ, BP_HIGH_HZ = 20.0, 450.0
NOTCH_HZ, NOTCH_Q = 60.0, 30.0
FILTER_ORDER = 4
LOG_EPS = 1.0
L = 20                     # longitud de secuencia
N_CANALES_EMG = 2          # biceps y triceps

N_REPETICIONES = 2000      # ciclos cronometrados
N_CALENTAMIENTO = 200      # ciclos descartados


def construir_filtros(fs=FS_EMG):
    nyq = 0.5 * fs
    b_bp, a_bp = sg.butter(FILTER_ORDER,
                           [BP_LOW_HZ / nyq, BP_HIGH_HZ / nyq],
                           btype="bandpass")
    b_n, a_n = sg.iirnotch(NOTCH_HZ / nyq, NOTCH_Q)
    return (b_bp, a_bp), (b_n, a_n)


def caracteristicas(W):
    """phi_k en R^7 sobre (n_ventanas, n_muestras). Copia literal de 01."""
    R = np.abs(W)
    fr = np.fft.rfftfreq(W.shape[1], 1.0 / FS_EMG)
    P = np.abs(np.fft.rfft(W, axis=1)) ** 2
    return np.c_[R.sum(1),
                 (R * R).sum(1),
                 np.abs(np.diff(R, axis=1)).sum(1),
                 np.exp(np.log(R + LOG_EPS).mean(1)),
                 (np.diff(np.sign(W), axis=1) != 0).sum(1),
                 (np.diff(np.sign(np.diff(W, axis=1)), axis=1) != 0).sum(1),
                 (P * fr).sum(1) / np.maximum(P.sum(1), 1e-12)]


def percentiles(v):
    a = np.asarray(v, float) * 1000.0     # a milisegundos
    return {"mediana_ms": round(float(np.median(a)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "p99_ms": round(float(np.percentile(a, 99)), 4),
            "maximo_ms": round(float(a.max()), 4),
            "n": int(a.size)}


def main():
    ruta_modelo = sys.argv[1] if len(sys.argv) > 1 else None
    ruta_npz = sys.argv[2] if len(sys.argv) > 2 else None

    print("Medicion del tiempo de computo de la cadena supervisora")

    # modelo
    modelo = None
    n_canales_psi = 15
    if ruta_modelo and os.path.isfile(ruta_modelo):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf

        class AtencionAditiva(layers.Layer):
            """Copia literal de la capa de 04_entrenar_mejorado.py, necesaria
            para deserializar los modelos que la incorporan."""

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
                c = super().get_config()
                c.update(unidades=self.unidades)
                return c

        modelo = keras.models.load_model(
            ruta_modelo, compile=False,
            custom_objects={"AtencionAditiva": AtencionAditiva})
        forma = modelo.input_shape
        if isinstance(forma, list):
            forma = forma[0]
        n_canales_psi = int(forma[-1])
        print("  modelo: %s  (entrada %s)" % (os.path.basename(ruta_modelo),
                                              modelo.input_shape))
    else:
        print("  modelo: NO indicado. Se mide solo filtrado y extraccion.")

    # secuencia de entrada
    if ruta_npz and os.path.isfile(ruta_npz):
        d = np.load(ruta_npz, allow_pickle=True)
        X = d["X_test"] if "X_test" in d else d["X_train"]
        secuencia = X[:1].astype(np.float32)
        origen = os.path.basename(ruta_npz)
        print("  secuencia: %s  %s" % (origen, secuencia.shape))
    else:
        secuencia = np.zeros((1, L, n_canales_psi), np.float32)
        origen = "sintetica, misma forma"
        print("  secuencia: sintetica %s" % (secuencia.shape,))

    inferir = None
    inferir_eager = None
    if modelo is not None:
        import tensorflow as tf
        inferir_eager = lambda x: modelo(x, training=False)
        inferir = tf.function(lambda x: modelo(x, training=False),
                              reduce_retracing=True)
        for _ in range(20):
            inferir(secuencia)          # traza y calienta

    # estado del filtro, como en operacion en linea
    (b1, a1), (b2, a2) = construir_filtros()
    zi1 = [sg.lfilter_zi(b1, a1) * 0.0 for _ in range(N_CANALES_EMG)]
    zi2 = [sg.lfilter_zi(b2, a2) * 0.0 for _ in range(N_CANALES_EMG)]
    buffer = np.zeros((N_CANALES_EMG, WIN_SAMPLES), float)
    generador = np.random.default_rng(0)

    t_filtro, t_carac, t_infer, t_ciclo = [], [], [], []

    for k in range(N_CALENTAMIENTO + N_REPETICIONES):
        bloque = generador.standard_normal((N_CANALES_EMG, HOP_SAMPLES))
        medir = k >= N_CALENTAMIENTO

        c0 = time.perf_counter()

        # filtrado causal del bloque nuevo, con estado
        t0 = time.perf_counter()
        for c in range(N_CANALES_EMG):
            y, zi1[c] = sg.lfilter(b1, a1, bloque[c], zi=zi1[c])
            y, zi2[c] = sg.lfilter(b2, a2, y, zi=zi2[c])
            buffer[c] = np.roll(buffer[c], -HOP_SAMPLES)
            buffer[c, -HOP_SAMPLES:] = y
        t1 = time.perf_counter()

        # caracteristicas sobre la ventana vigente, ambos canales
        _ = caracteristicas(buffer)
        t2 = time.perf_counter()

        # una inferencia
        if inferir is not None:
            _ = inferir(secuencia)
        t3 = time.perf_counter()

        if medir:
            t_filtro.append(t1 - t0)
            t_carac.append(t2 - t1)
            if modelo is not None:
                t_infer.append(t3 - t2)
            t_ciclo.append(t3 - c0)

    # invocacion directa, para dejar constancia del sobrecoste del framework
    t_eager = []
    if inferir_eager is not None:
        for _ in range(200):
            t0 = time.perf_counter()
            inferir_eager(secuencia)
            t_eager.append(time.perf_counter() - t0)

    salida = {
        "descripcion": "Tercer termino del presupuesto de latencia, "
                       "seccion 3.8.2. Coste de computo por ciclo del "
                       "supervisor en esta computadora.",
        "no_incluye": ["retardo de adquisicion",
                       "acumulacion de la ventana",
                       "retardo de comunicacion con el sistema embebido",
                       "periodo de actualizacion del supervisor"],
        "parametros": {"fs_emg_hz": FS_EMG, "ventana_muestras": WIN_SAMPLES,
                       "salto_muestras": HOP_SAMPLES, "longitud_secuencia": L,
                       "canales_emg": N_CANALES_EMG,
                       "componentes_psi": n_canales_psi,
                       "ciclos_medidos": N_REPETICIONES,
                       "origen_secuencia": origen,
                       "modelo": os.path.basename(ruta_modelo)
                       if ruta_modelo else None},
        "filtrado_causal": percentiles(t_filtro),
        "extraccion_caracteristicas": percentiles(t_carac),
        "inferencia_compilada": percentiles(t_infer) if t_infer else
                                "no medida: no se indico modelo",
        "inferencia_invocacion_directa": percentiles(t_eager) if t_eager else
                                         "no medida",
        "ciclo_completo": percentiles(t_ciclo),
    }

    with open("latencia_computo.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print("\n  RESULTADOS (ms)")
    print("  %-32s %9s %9s %9s" % ("etapa", "mediana", "p95", "p99"))
    for clave in ("filtrado_causal", "extraccion_caracteristicas",
                  "inferencia_compilada", "inferencia_invocacion_directa",
                  "ciclo_completo"):
        v = salida[clave]
        if isinstance(v, dict):
            print("  %-32s %9.4f %9.4f %9.4f"
                  % (clave, v["mediana_ms"], v["p95_ms"], v["p99_ms"]))
        else:
            print("  %-32s %s" % (clave, v))
    print("\n  Escrito: latencia_computo.json")
    if modelo is None:
        print("  AVISO: falta la inferencia. Vuelve a ejecutar indicando el "
              "modelo entrenado (.keras) para completar el termino.")


if __name__ == "__main__":
    main()
