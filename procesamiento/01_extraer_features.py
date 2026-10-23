"""Extrae las caracteristicas por ventana desde el EMG crudo de cada sesion.

Aplica el filtrado causal, calcula los descriptores por canal y escribe un
featok_<sesion>.csv por cada emg_<sesion>.csv de entrada.
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy import signal as sg

HERE = os.path.dirname(os.path.abspath(__file__))

_RAIZ      = os.path.dirname(HERE)
DIR_DATOS  = os.path.join(_RAIZ, "data")
DIR_SALIDA = os.path.join(_RAIZ, "resultados")

if not os.path.isdir(DIR_DATOS):      # ejecucion en carpeta unica
    DIR_DATOS = HERE
if not os.path.isdir(DIR_SALIDA):
    DIR_SALIDA = HERE
os.makedirs(DIR_SALIDA, exist_ok=True)


# parametros (identicos a la propuesta metodologica)
FS_EMG      = 1000.0
WIN_MS      = 300.0
WIN_HOP_MS  = 10
WIN_SAMPLES = int(FS_EMG * WIN_MS / 1000)      # 300
HOP_SAMPLES = int(FS_EMG * WIN_HOP_MS / 1000)  # 10

BP_LOW_HZ, BP_HIGH_HZ = 20.0, 450.0
NOTCH_HZ, NOTCH_Q     = 60.0, 30.0
FILTER_ORDER          = 4
LOG_EPS               = 1.0
DESCARTE_INICIAL_S    = 1.0

REGISTRO_SESIONES = {
    "20260806_182858": dict(carga_kg=0.00, usar_para="completa",
                            nota="IMU congelada; se usa q_el del servo"),
    "20260806_193017": dict(carga_kg=0.00, usar_para="parcial",
                            nota="sin levantamiento ni bajada; 3 reps de ext_hombro"),
    "20260807_150646": dict(carga_kg=0.00, usar_para="completa",
                            nota="calibracion de reposo puro"),
    "20260807_170550": dict(carga_kg=0.85, usar_para="completa",
                            nota="calibracion tomada con el peso puesto; 3 reps de bajada"),
    "20260811_122212": dict(carga_kg=0.50, usar_para="completa",
                            nota="calibracion sin carga solo los primeros 13 s"),
}

# Instante a partir del cual la calibracion deja de ser reposo sin carga.
# None significa que el bloque completo es valido.
CALIBRACION_VALIDA_HASTA_S = {
    "20260811_122212": 13.0,   # a partir del segundo 13 se toma la pesa
    "20260807_170550": 0.0,    # la calibracion se hizo con el peso puesto
}

# x_NN en R^7. q_el y dq_el proceden de la IMU, como exige la propuesta.
EXO_MAP = {
    "q_sh":  "ShoulderPos_deg",
    "q_el":  "ElbowJointAngle_deg",   # coordenada articular (v5, sec. instrumentacion)
    "dq_sh": "ShoulderSpd_dps",
    "dq_el": "ElbowIMUOmega_dps",     # IMU, medida, no derivada
    "i_sh":  "ShoulderCur_A",
    "F_el":  "LoadE_g",
    "F_sh":  "LoadS_g",
}

# Columnas del exo que se interpolan si traen NaN por tramas perdidas.
COLUMNAS_EXO = list(EXO_MAP.values())


def construir_filtros(fs):
    nyq = 0.5 * fs
    b_bp, a_bp = sg.butter(FILTER_ORDER,
                           [BP_LOW_HZ / nyq, BP_HIGH_HZ / nyq],
                           btype="bandpass")
    b_n, a_n = sg.iirnotch(NOTCH_HZ / nyq, NOTCH_Q)
    return (b_bp, a_bp), (b_n, a_n)


def filtrar(x, fs=FS_EMG):
    """Pasa-banda 20-450 Hz orden 4 y notch de 60 Hz, ambos causales."""
    (b1, a1), (b2, a2) = construir_filtros(fs)
    return sg.lfilter(b2, a2, sg.lfilter(b1, a1, x.astype(np.float64)))


def caracteristicas(W):
    """
    Vector phi_k en R^7 sobre una matriz de ventanas ya filtradas, de
    forma (n_ventanas, n_muestras). Las cuatro primeras son las formulas
    literales de la seccion 2.4 de la propuesta metodologica; las tres
    ultimas corresponden a la ampliacion del apartado 2.4.1.

    La propuesta define primero la senal rectificada
        r_k[n] = |x_f[n]|
    y construye IAV, SSI y WL a partir de ella. LOG se define sobre
    |x_f[n]| directamente.

        IAV_k = suma_{n=1..N} r_k[n]
        SSI_k = suma_{n=1..N} r_k[n]^2
        WL_k  = suma_{n=2..N} |r_k[n] - r_k[n-1]|      <-- sobre r, no sobre x_f
        LOG_k = exp( (1/N) suma_{n=1..N} ln(|x_f[n]| + eps) ),  eps > 0

    Nota sobre SSI: como r = |x_f|, se cumple r^2 = x_f^2, de modo que
    sumar r^2 y sumar x_f^2 es identico. Se escribe sobre R por fidelidad
    literal a la formula del documento.

    Nota sobre WL: la propuesta la define sobre la senal RECTIFICADA.
    Calcularla sobre x_f sin rectificar sobrestima el valor en torno a un
    16 % y no corresponde a la formula publicada.

    AMPLIACION AL DOMINIO DE LA FRECUENCIA (apartado 2.4.1)
    -------------------------------------------------------
        ZC_k  = numero de cruces por cero de x_f
        SSC_k = numero de cambios de signo de la pendiente de x_f
        MNF_k = frecuencia media del espectro de potencia

    Motivo. Las cuatro primeras son descriptores de amplitud, y la carga
    externa actua multiplicando la amplitud, de modo que las afecta por
    igual. Los tres anadidos no dependen de la ganancia del electrodo y
    reflejan el contenido espectral, que cambia con el reclutamiento de
    unidades motoras al iniciarse el movimiento.

    ZC y SSC pertenecen al conjunto clasico de Hudgins et al. (1993), ya
    citado en la metodologia, del que la formulacion anterior retenia
    solo WL. Phinyomark et al. (2012) documentan la redundancia entre
    descriptores de amplitud y la conveniencia de combinar familias.

    Evidencia medida sobre la ventana de transicion, con validacion
    agrupada por episodio: los cuatro de amplitud dan AUC 0.543, los tres
    espectrales 0.624, y los siete juntos 0.632.
    """
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


def ventanas(x, n, h):
    """Vista deslizante sin copiar memoria: (n_ventanas, n)."""
    x = np.ascontiguousarray(x)
    n_v = (len(x) - n) // h + 1
    s = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_v, n), strides=(h * s, s), writeable=False)


def clave_sesion(ruta):
    base = os.path.basename(ruta)
    return base.replace("emg_", "").replace(".csv", "")


def procesar(ruta):
    clave = clave_sesion(ruta)
    meta = REGISTRO_SESIONES.get(clave)
    if meta is None:
        raise KeyError("Sesion {} no esta en REGISTRO_SESIONES. "
                       "Anade su carga antes de procesarla.".format(clave))

    print("\n" + "=" * 66)
    print("{}   carga {:.2f} kg   ({})".format(clave, meta["carga_kg"],
                                               meta["usar_para"]))

    df = pd.read_csv(ruta, low_memory=False)
    faltan = [c for c in ["EMG_ch2_raw_bic", "EMG_ch3_raw_tri",
                          "t_since_label_s", "label", "EMG_t_us"]
              + COLUMNAS_EXO if c not in df.columns]
    if faltan:
        raise KeyError("Faltan columnas: {}".format(faltan))

    # tramas perdidas del exo
    n_nan = int(df[COLUMNAS_EXO].isna().any(axis=1).sum())
    if n_nan:
        print("  {} filas con NaN en el exo (tramas perdidas): se interpolan"
              .format(n_nan))
        df[COLUMNAS_EXO] = (df[COLUMNAS_EXO]
                            .interpolate(limit_direction="both"))
    if df[COLUMNAS_EXO].isna().any().any():
        raise ValueError("Quedan NaN en el exo tras interpolar.")

    # verificacion de integridad temporal
    dt = np.diff(df["EMG_t_us"].values)
    fs_real = 1e6 / np.median(dt)
    huecos = int((dt > 2000).sum())
    print("  fs real {:.2f} Hz   huecos > 2 ms: {}".format(fs_real, huecos))
    if huecos:
        print("  [AVISO] hay discontinuidades en el reloj del EMG.")

    # descarte del transitorio del filtro
    n_desc = int(DESCARTE_INICIAL_S * FS_EMG)
    df = df.iloc[n_desc:].reset_index(drop=True)
    t_rel = (df["EMG_t_us"].values - df["EMG_t_us"].values[0]) / 1e6

    # filtrado
    bic = filtrar(df["EMG_ch2_raw_bic"].values)
    tri = filtrar(df["EMG_ch3_raw_tri"].values)
    print("  offset crudo  bic {:.0f}  tri {:.0f} cuentas"
          .format(df["EMG_ch2_raw_bic"].mean(), df["EMG_ch3_raw_tri"].mean()))
    print("  tras filtrar  bic {:.4f}  tri {:.4f}  (debe ser ~0)"
          .format(bic.mean(), tri.mean()))

    # caracteristicas por ventana
    fb = caracteristicas(ventanas(bic, WIN_SAMPLES, HOP_SAMPLES))
    ft = caracteristicas(ventanas(tri, WIN_SAMPLES, HOP_SAMPLES))
    n_v = len(fb)
    fin = np.arange(n_v) * HOP_SAMPLES + WIN_SAMPLES - 1   # ultima muestra

    out = pd.DataFrame({
        "wall_clock": df["wall_clock"].values[fin] if "wall_clock" in df else "",
        "subject_id": df["subject_id"].values[fin] if "subject_id" in df else "S00",
        "load_kg":    meta["carga_kg"],
        "IAV_bic": fb[:, 0], "SSI_bic": fb[:, 1],
        "WL_bic":  fb[:, 2], "LOG_bic": fb[:, 3],
        "ZC_bic":  fb[:, 4], "SSC_bic": fb[:, 5], "MNF_bic": fb[:, 6],
        "IAV_tri": ft[:, 0], "SSI_tri": ft[:, 1],
        "WL_tri":  ft[:, 2], "LOG_tri": ft[:, 3],
        "ZC_tri":  ft[:, 4], "SSC_tri": ft[:, 5], "MNF_tri": ft[:, 6],
    })
    for destino, origen in EXO_MAP.items():
        out[destino] = df[origen].values[fin]
    out["t_since_label_s"] = df["t_since_label_s"].values[fin]

    # diagnostico del estado de la IMU
    q = out["q_el"].values   # ElbowJointAngle_deg
    plano = float((np.diff(q) == 0).mean()) * 100.0
    cambios = np.flatnonzero(np.diff(q) != 0)
    tramos = np.diff(np.r_[0, cambios, len(q)]) * (WIN_HOP_MS / 1000.0)
    print("  IMU q_el (ElbowJointAngle_deg): {:.1f} % sin cambio, tramo plano maximo "
          "{:.1f} s, desviacion {:.3f}".format(plano, tramos.max(), q.std()))
    if tramos.max() > 5.0:
        print("  [AVISO] la IMU permanece congelada mas de 5 s. La sesion se "
              "conserva tal cual; es una condicion de sensor degradado.")

    # etiqueta, con la correccion de la calibracion
    etiqueta = df["label"].fillna("NULL").values[fin].astype(object)
    corte = CALIBRACION_VALIDA_HASTA_S.get(clave, None)
    if corte is not None:
        t_v = t_rel[fin]
        es_cal = etiqueta == "calibracion"
        t0_cal = t_v[es_cal].min() if es_cal.any() else 0.0
        invalida = es_cal & ((t_v - t0_cal) >= corte)
        etiqueta[invalida] = "calibracion_con_carga"
        print("  calibracion valida hasta {:.0f} s: {} ventanas reetiquetadas "
              "como calibracion_con_carga".format(corte, int(invalida.sum())))

    out["label_str"] = etiqueta
    out["label_num"] = np.nan          # lo asigna el paso 2

    out = out[["wall_clock", "subject_id", "load_kg",
               "IAV_bic", "SSI_bic", "WL_bic", "LOG_bic",
               "ZC_bic", "SSC_bic", "MNF_bic",
               "IAV_tri", "SSI_tri", "WL_tri", "LOG_tri",
               "ZC_tri", "SSC_tri", "MNF_tri",
               "q_sh", "q_el", "dq_sh", "dq_el", "i_sh", "F_el", "F_sh",
               "t_since_label_s", "label_str", "label_num"]]

    destino = os.path.join(DIR_SALIDA, "featok_" + clave + ".csv")
    out.to_csv(destino, index=False)
    print("  {:,} ventanas a {:.0f} Hz  ->  {}"
          .format(len(out), FS_EMG / HOP_SAMPLES, os.path.basename(destino)))

    # informe de contraste
    et = out["label_str"]
    gestos = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro",
              "levantamiento"]
    base_cal = out.loc[et == "calibracion", "IAV_bic"]
    base_null = out.loc[et == "NULL", "IAV_bic"]
    act = out.loc[et.isin(gestos), "IAV_bic"]
    if len(base_cal):
        print("  IAV biceps  calibracion p20 = {:>10,.0f}".format(
            base_cal.quantile(0.20)))
    if len(base_null) and len(act):
        print("  IAV biceps  NULL medio     = {:>10,.0f}".format(base_null.mean()))
        print("  IAV biceps  gestos medio   = {:>10,.0f}   ratio {:.2f}x"
              .format(act.mean(), act.mean() / base_null.mean()))
    return len(out)


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    if not rutas:
        raise FileNotFoundError(
            "No hay emg_*.csv en {} ni en sus subcarpetas. Descomprime ahi "
            "los ZIP de las sesiones.".format(DIR_DATOS))
    print("PASO 1  EXTRACCION DE CARACTERISTICAS")
    print("ventana {:.0f} ms | salto {} ms | {} sesiones"
          .format(WIN_MS, WIN_HOP_MS, len(rutas)))
    print("datos   : {}".format(DIR_DATOS))
    print("salida  : {}".format(DIR_SALIDA))
    total = sum(procesar(r) for r in rutas)
    print("\n" + "=" * 66)
    print("{:,} ventanas en total. Siguiente: 02_construir_secuencias.py"
          .format(total))


if __name__ == "__main__":
    main()
