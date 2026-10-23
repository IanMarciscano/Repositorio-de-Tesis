"""Construye las secuencias de entrada y los tres pliegues por condicion de carga.

Normaliza por sesion, recorta la saturacion y escribe pliegue_<carga>kg.npz,
informe_paso2.json y sesiones_parciales.npz.
"""
import os
import glob
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

_RAIZ      = os.path.dirname(HERE)
DIR_DATOS  = os.path.join(_RAIZ, "data")
DIR_SALIDA = os.path.join(_RAIZ, "resultados")

if not os.path.isdir(DIR_DATOS):      # ejecucion en carpeta unica
    DIR_DATOS = HERE
if not os.path.isdir(DIR_SALIDA):
    DIR_SALIDA = HERE
os.makedirs(DIR_SALIDA, exist_ok=True)


# parametros
L            = 20        # v5: longitud de secuencia
WIN_MS       = 300.0     # v5: T_w
HOP_MS       = 10        # v5: T_s
PUREZA       = 0.80      # v5: umbral de pureza
TRIM_PUREZA_S = PUREZA * WIN_MS / 1000.0        # 0.240 s
TRIM_REPOSO_S = 2.0      # D5
SATURACION   = 10.0      # D7: recorte de |z| tras normalizar

H_EFF_MS = (L - 1) * HOP_MS + WIN_MS            # 490 ms

EMG_COLS = ["IAV_bic", "SSI_bic", "WL_bic", "LOG_bic",
            "ZC_bic", "SSC_bic", "MNF_bic",
            "IAV_tri", "SSI_tri", "WL_tri", "LOG_tri",
            "ZC_tri", "SSC_tri", "MNF_tri"]          # 14, apartado 2.4.1
MEC_COLS = ["q_sh", "q_el", "dq_sh", "dq_el", "i_sh", "F_el", "F_sh"]
PSI_COLS = EMG_COLS + MEC_COLS                  # ψ ∈ R^21
N_EMG    = len(EMG_COLS)                        # 14

# v5 seccion 3.1. La calibracion no pertenece a C.
CLASES = {
    "NULL":          0,   # reposo entre gestos
    "flex_codo":     1,
    "ext_codo":      2,
    "flex_hombro":   3,
    "ext_hombro":    4,
    "levantamiento": 5,
}
NOMBRES = ["reposo", "flex_codo", "ext_codo",
           "flex_hombro", "ext_hombro", "levantamiento"]

# Etiquetas presentes en los CSV que quedan fuera del clasificador.
EXCLUIDAS = ["calibracion", "calibracion_con_carga", "bajada", "STOPPED"]

SESIONES_PARCIALES = ["20260806_193017"]


_OFFSET_BLOQUE = 100000   # margen holgado: cada sesion tiene ~50 bloques


# Prefijos de todas las caracteristicas sEMG que puede contener un
# featok_*.csv. Sirve para detectar versiones incompatibles del paso 1.
_PREFIJOS_EMG = ("IAV_", "SSI_", "WL_", "LOG_", "ZC_", "SSC_", "MNF_")


def cargar_sesion(ruta):
    df = pd.read_csv(ruta, low_memory=False)
    clave = os.path.basename(ruta).replace("featok_", "").replace(".csv", "")
    df["sesion"] = clave
    faltan = [c for c in PSI_COLS + ["label_str", "t_since_label_s", "load_kg"]
              if c not in df.columns]
    if faltan:
        raise KeyError(
            "{}: faltan columnas {}. Es probable que featok_*.csv proceda de "
            "una version anterior de 01_extraer_features.py. Vuelve a "
            "ejecutar el paso 1 con la version actual."
            .format(clave, faltan))

    presentes = [c for c in df.columns if c.startswith(_PREFIJOS_EMG)]
    sobran = sorted(set(presentes) - set(EMG_COLS))
    if sobran:
        raise ValueError(
            "{}: el archivo contiene {} caracteristicas sEMG pero este script "
            "solo consume {}. Sin usar: {}. Los pasos 1 y 2 pertenecen a "
            "versiones distintas del pipeline; actualiza 02_construir_"
            "secuencias.py o vuelve a ejecutar el paso 1."
            .format(clave, len(presentes), len(EMG_COLS), sobran))
    return clave, df


def parametros_emg(df, clave):
    """
    mu y sigma de las ocho caracteristicas sEMG, del bloque L6 de la
    propia sesion (v5 seccion 2.5). Devuelve tambien un diagnostico del
    estado del bloque para poder declarar D2, D3 y D4.
    """
    cal = df[df["label_str"] == "calibracion"]
    cal_carga = df[df["label_str"] == "calibracion_con_carga"]

    if len(cal) < L:
        # Caso D2: no hay ni un tramo de calibracion sin carga.
        if len(cal_carga) >= L:
            cal = cal_carga
            estado = "CONTAMINADA: bloque ejecutado con carga (D2)"
        else:
            raise ValueError(
                "{}: no hay bloque de calibracion utilizable".format(clave))
    elif len(cal_carga):
        estado = ("PARCIAL: {:.1f} s validos de {:.1f} s (D3)"
                  .format(len(cal) / 100.0,
                          (len(cal) + len(cal_carga)) / 100.0))
    else:
        estado = "completa"

    mu = cal[EMG_COLS].mean().values
    sd = cal[EMG_COLS].std(ddof=0).values   # mismo ddof que la mecanica
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd, estado, len(cal) / 100.0


def construir(df, mu_emg, sd_emg):
    """
    Devuelve las secuencias de una sesion sin normalizar la parte
    mecanica. La parte sEMG si queda normalizada aqui, porque su
    referencia es la calibracion de la propia sesion y no depende de la
    particion.
    """
    df = df.reset_index(drop=True)

    X = df[PSI_COLS].values.astype(np.float64)
    X[:, :N_EMG] = (X[:, :N_EMG] - mu_emg) / sd_emg  # v5 ec. (18)

    etiqueta = df["label_str"].fillna("NULL").values
    tsl = df["t_since_label_s"].values.astype(np.float64)

    desconocidas = sorted(set(etiqueta) - set(CLASES) - set(EXCLUIDAS))
    if desconocidas:
        raise ValueError(
            "Etiquetas no reconocidas: {}. Anadelas a CLASES si son clases "
            "del clasificador, o a EXCLUIDAS si deben descartarse."
            .format(desconocidas))

    # bloques fisicos: cambio de etiqueta en el CSV
    blk = np.cumsum(np.r_[0, (etiqueta[1:] != etiqueta[:-1])])

    y = np.array([CLASES.get(e, -1) for e in etiqueta])

    # secuencias de l vectores con etiqueta homogenea
    S = np.lib.stride_tricks.sliding_window_view(X, (L, X.shape[1]))
    S = S.squeeze(1)                                   # (n, L, 15)
    y_s   = y[L - 1:]
    blk_s = blk[L - 1:]
    tsl_s = tsl[L - 1:]

    homogenea = np.array([len(set(y[i:i + L])) == 1 for i in range(len(S))])
    valida = homogenea & (y_s >= 0)

    return (S[valida].astype(np.float32), y_s[valida],
            blk_s[valida], tsl_s[valida])


def mascara_entrenable(y, tsl):
    """
    Poda que se aplica SOLO a entrenamiento y validacion (v5 seccion 2.6).

      - 240 ms desde el cambio de etiqueta: criterio de pureza del 80 %.
      - 2.0 s adicionales en los bloques de reposo: el inicio del bloque
        contiene el retorno a la posicion inicial, que no es reposo (D5).
    """
    umbral = np.where(y == 0, TRIM_REPOSO_S, TRIM_PUREZA_S)
    return tsl >= umbral


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_SALIDA, "featok_*.csv")))
    if not rutas:
        raise FileNotFoundError(
            "No hay featok_*.csv en {}. Ejecuta antes 01_extraer_features.py"
            .format(DIR_SALIDA))

    print("PASO 2  CONSTRUCCION DE SECUENCIAS")
    print("L = {} | T_w = {:.0f} ms | T_s = {} ms | H_eff = {:.0f} ms | psi en R^{}"
          .format(L, WIN_MS, HOP_MS, H_EFF_MS, len(PSI_COLS)))
    print("pureza {:.0f} % -> {:.0f} ms | recorte de reposo {:.1f} s | "
          "saturacion +-{:.0f}"
          .format(PUREZA * 100, TRIM_PUREZA_S * 1000, TRIM_REPOSO_S,
                  SATURACION))

    datos = {}
    informe = {}
    for ruta in rutas:
        clave, df = cargar_sesion(ruta)
        carga = float(df["load_kg"].iloc[0])
        mu, sd, estado, dur_cal = parametros_emg(df, clave)
        S, y, blk, tsl = construir(df, mu, sd)

        blk = blk + _OFFSET_BLOQUE * len(datos)

        parcial = clave in SESIONES_PARCIALES
        datos[clave] = dict(X=S, y=y, blk=blk, tsl=tsl,
                            carga=carga, parcial=parcial)
        informe[clave] = dict(carga_kg=carga, calibracion=estado,
                              dur_calibracion_s=round(dur_cal, 1),
                              secuencias=int(len(y)),
                              parcial=parcial,
                              mu_IAV_bic=round(float(mu[0]), 1),
                              sd_IAV_bic=round(float(sd[0]), 1))

        print("\n{}   carga {:.2f} kg{}".format(
            clave, carga, "   [PARCIAL]" if parcial else ""))
        print("   calibracion: {}  ({:.1f} s)".format(estado, dur_cal))
        print("   mu IAV_bic = {:>12,.0f}   sigma = {:>12,.0f}"
              .format(mu[0], sd[0]))
        print("   {:,} secuencias   clases {}".format(
            len(y), np.bincount(y, minlength=6).tolist()))

    # comprobacion del sesgo de d2
    mus = {k: v["mu_IAV_bic"] for k, v in informe.items()}
    sanas = [v for k, v in mus.items() if "CONTAMINADA" not in informe[k]["calibracion"]]
    for k, v in informe.items():
        if "CONTAMINADA" in v["calibracion"] and sanas:
            factor = mus[k] / np.mean(sanas)
            print("\n[D2] {}: mu IAV_bic es {:.1f} veces la media de las "
                  "calibraciones sin carga.".format(k, factor))
            informe[k]["sesgo_mu_vs_sanas"] = round(float(factor), 2)

    # pliegues leave-one-condition-out
    cargas = sorted({v["carga"] for v in datos.values()})
    print("\n" + "=" * 70)
    print("PARTICION leave-one-condition-out sobre {} condiciones: {}"
          .format(len(cargas), cargas))

    for excluida in cargas:
        # Las sesiones parciales no participan en los pliegues de 6 clases.
        completas = {k: v for k, v in datos.items() if not v["parcial"]}
        test_k = [k for k, v in completas.items() if v["carga"] == excluida]
        tren_k = [k for k, v in completas.items() if v["carga"] != excluida]
        if not test_k:
            print("\n[omitido] carga {:.2f} kg: sin sesiones completas"
                  .format(excluida))
            continue
        if not tren_k:
            print("\n[omitido] carga {:.2f} kg: no queda nada para entrenar"
                  .format(excluida))
            continue

        Xtr = np.concatenate([completas[k]["X"] for k in tren_k])
        ytr = np.concatenate([completas[k]["y"] for k in tren_k])
        btr = np.concatenate([completas[k]["blk"] for k in tren_k])
        ttr = np.concatenate([completas[k]["tsl"] for k in tren_k])

        Xte = np.concatenate([completas[k]["X"] for k in test_k])
        yte = np.concatenate([completas[k]["y"] for k in test_k])
        bte = np.concatenate([completas[k]["blk"] for k in test_k])
        tte = np.concatenate([completas[k]["tsl"] for k in test_k])
        ste = np.concatenate([np.full(len(completas[k]["y"]), k)
                              for k in test_k])

        # validacion: ultimo bloque de cada clase dentro del entrenamiento
        es_val = np.zeros(len(ytr), bool)
        for c in range(6):
            bs = np.unique(btr[ytr == c])
            if len(bs) > 1:
                es_val |= np.isin(btr, bs[-1:]) & (ytr == c)

        # poda asimetrica: solo train y validacion
        pod = mascara_entrenable(ytr, ttr)
        m_tr = (~es_val) & pod
        m_va = es_val & pod

        faltan_val = [NOMBRES[c] for c in range(6)
                      if not (ytr[m_va] == c).any()]
        if faltan_val:
            print("   [AVISO] la validacion no cubre: {}. El criterio de "
                  "parada temprana no vera esas clases."
                  .format(", ".join(faltan_val)))
        faltan_tr = [NOMBRES[c] for c in range(6)
                     if not (ytr[m_tr] == c).any()]
        if faltan_tr:
            raise ValueError("El entrenamiento no contiene la clase / las "
                             "clases: {}".format(", ".join(faltan_tr)))

        # normalizacion mecanica: solo con el entrenamiento del pliegue
        mec = Xtr[m_tr][:, :, N_EMG:].reshape(-1, len(MEC_COLS))
        mu_m = mec.mean(0)
        sd_m = mec.std(0)
        sd_m = np.where(sd_m < 1e-9, 1.0, sd_m)

        def norm(A):
            A = A.copy()
            A[:, :, N_EMG:] = (A[:, :, N_EMG:] - mu_m) / sd_m
            np.clip(A, -SATURACION, SATURACION, out=A)      # D7
            return A.astype(np.float32)

        # porcentaje de valores que alcanzan la saturacion, por bloque
        _pre = Xtr[m_tr].copy()
        _pre[:, :, N_EMG:] = (_pre[:, :, N_EMG:] - mu_m) / sd_m
        sat_emg = 100.0 * (np.abs(_pre[:, :, :N_EMG]) > SATURACION).mean()
        sat_mec = 100.0 * (np.abs(_pre[:, :, N_EMG:]) > SATURACION).mean()
        del _pre

        destino = os.path.join(DIR_SALIDA, "pliegue_{:.2f}kg.npz".format(excluida))
        np.savez_compressed(
            destino,
            X_train=norm(Xtr[m_tr]), y_train=ytr[m_tr],
            X_val=norm(Xtr[m_va]),   y_val=ytr[m_va],
            X_test=norm(Xte),        y_test=yte,
            t_since_label_test=tte, blk_test=bte, sesion_test=ste,
            mu_mecanica=mu_m, sd_mecanica=sd_m, saturacion=SATURACION,
            pct_saturado_emg=sat_emg, pct_saturado_mec=sat_mec,
            carga_excluida=excluida,
            sesiones_train=np.array(tren_k), sesiones_test=np.array(test_k),
            columnas=np.array(PSI_COLS))

        print("\ncarga excluida {:.2f} kg   test: {}".format(
            excluida, ", ".join(test_k)))
        print("   train {:>7,}  val {:>6,}  test {:>7,}   (test sin podar)"
              .format(int(m_tr.sum()), int(m_va.sum()), len(yte)))
        print("   clases train {}".format(
            np.bincount(ytr[m_tr], minlength=6).tolist()))
        print("   clases test  {}".format(
            np.bincount(yte, minlength=6).tolist()))
        print("   saturacion a +-{:.0f}: sEMG {:.1f} %   mecanica {:.1f} %"
              .format(SATURACION, sat_emg, sat_mec))
        print("   -> {}".format(os.path.basename(destino)))

    # sesiones parciales, guardadas aparte
    parciales = [k for k, v in datos.items() if v["parcial"]]
    if parciales:
        np.savez_compressed(
            os.path.join(DIR_SALIDA, "sesiones_parciales.npz"),
            **{"X_" + k: datos[k]["X"] for k in parciales},
            **{"y_" + k: datos[k]["y"] for k in parciales},
            **{"blk_" + k: datos[k]["blk"] for k in parciales},
            sesiones=np.array(parciales))
        print("\n{} excluida(s) de los pliegues de 6 clases y guardada(s) "
              "en sesiones_parciales.npz: {}"
              .format(len(parciales), ", ".join(parciales)))

    with open(os.path.join(DIR_SALIDA, "informe_paso2.json"), "w") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("informe_paso2.json escrito. Siguiente: 03_entrenar.py")


if __name__ == "__main__":
    main()
