"""Marca los instantes de inicio mecanico y construye la poblacion de episodios.

Escribe anticipacion_episodios.json.
"""
# -*- coding: utf-8 -*-
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal as sg

DIR_DATOS = sys.argv[1] if len(sys.argv) > 1 else "data"

FS = 1000.0
BP = (20.0, 450.0)
NOTCH = (60.0, 30.0)
ORDEN = 4
UMBRAL_VEL = 5.0
PERSIST_MS = 50
SUAVIZADO_MS = 50
K_SIGMA = 3.0
BUSQUEDA_PREVIA_MS = 2000     # tope de busqueda antes del cambio de etiqueta

GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro",
          "levantamiento"]
CANALES = ["EMG_ch2_raw_bic", "EMG_ch3_raw_tri"]


def filtrar(x):
    nyq = FS / 2
    b1, a1 = sg.butter(ORDEN, [BP[0] / nyq, BP[1] / nyq], btype="bandpass")
    b2, a2 = sg.iirnotch(NOTCH[0] / nyq, NOTCH[1])
    return sg.lfilter(b2, a2, sg.lfilter(b1, a1, x.astype(np.float64)))


def envolvente(x):
    """Rectificacion y media movil causal de 50 ms, como en Hodges y Bui."""
    n = int(FS * SUAVIZADO_MS / 1000)
    return np.convolve(np.abs(x), np.ones(n) / n, mode="full")[:len(x)]


def primer_cruce(x, umbral, persist):
    s = (x > umbral).astype(int)
    if s.sum() < persist:
        return None
    c = np.convolve(s, np.ones(persist), "valid")
    i = np.flatnonzero(c >= persist)
    return int(i[0]) if len(i) else None


def inicio_de_la_activacion(x, umbral, persist):
    """Primer cruce sostenido del umbral dentro de la ventana de búsqueda,
    con la condición de que la señal esté POR DEBAJO del umbral al abrirse
    esa ventana.

    La condición es lo que hace determinable el instante. Si la envolvente
    ya está por encima del umbral cuando la ventana se abre, no hay
    transición de reposo a activación que localizar: lo que se observaría
    sería el punto arbitrario en que empieza a mirarse, no un inicio. Esos
    episodios se declaran indeterminados en lugar de asignarles una marca
    que no significa nada.

    La ventana se extiende hasta el final del episodio, no hasta t_mov, de
    modo que el signo del margen t_mov − t_EMG queda libre: la activación
    puede resultar anterior o posterior al movimiento observable.
    """
    if len(x) < persist:
        return None
    if x[0] > umbral:                     # ya activa al abrirse la ventana
        return None
    return primer_cruce(x, umbral, persist)


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    if not rutas:
        sys.exit("No hay emg_*.csv en " + DIR_DATOS)

    persist = int(FS * PERSIST_MS / 1000)
    registros = []
    resumen_sesiones = []

    for ruta in rutas:
        nombre = os.path.basename(ruta)
        e = pd.read_csv(ruta, low_memory=False)
        lab = e["label"].fillna("NULL").values
        env = {c: envolvente(filtrar(pd.to_numeric(e[c], errors="coerce")
                                     .fillna(0).values)) for c in CANALES}
        v_el = np.abs(pd.to_numeric(e["ElbowIMUOmega_dps"],
                                    errors="coerce").fillna(0).values)
        v_sh = np.abs(pd.to_numeric(e["ShoulderSpd_dps"],
                                    errors="coerce").fillna(0).values)

        # linea base de la sesion, del bloque de calibracion sin carga
        cal = lab == "calibracion"
        base_sesion = None
        if cal.sum() >= FS:
            base_sesion = {c: (float(env[c][cal].mean()),
                               float(env[c][cal].std())) for c in CANALES}

        cortes = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        n_tot = n_mov = n_ambas = n_respaldo = 0
        for k, i0 in enumerate(cortes):
            if lab[i0] not in GESTOS or k == 0:
                continue
            i1 = cortes[k + 1] if k + 1 < len(cortes) else len(lab)
            clase = lab[i0]
            n_tot += 1

            v = v_el[i0:i1] if "codo" in clase else v_sh[i0:i1]
            tm = primer_cruce(v, UMBRAL_VEL, persist)
            reg = dict(sesion=nombre, clase=clase, inicio_etiqueta_s=i0 / FS,
                       t_mov_s=None,
                       t_emg_s_calibracion=None, retardo_ms_calibracion=None,
                       t_emg_s_reposo_previo=None,
                       retardo_ms_reposo_previo=None)
            if tm is None:
                registros.append(reg)
                continue
            n_mov += 1
            t_mov = i0 + tm
            reg["t_mov_s"] = t_mov / FS

            j0, j1 = cortes[k - 1], max(cortes[k - 1], i0 - int(FS * 0.5))
            base_reposo = None
            if j1 - j0 >= FS / 2:
                base_reposo = {c: (float(env[c][j0:j1].mean()),
                                   float(env[c][j0:j1].std()))
                               for c in CANALES}

            b0 = max(cortes[k - 1], i0 - int(FS * 0.5))
            for etiqueta, base in (("calibracion", base_sesion),
                                   ("reposo_previo", base_reposo)):
                if base is None:
                    continue
                candidatos = []
                for c in CANALES:
                    mu, sd = base[c]
                    te = inicio_de_la_activacion(env[c][b0:t_mov],
                                                 mu + K_SIGMA * sd, persist)
                    if te is not None:
                        candidatos.append(b0 + te)
                if not candidatos:
                    continue
                t_emg = min(candidatos)
                reg["t_emg_s_" + etiqueta] = t_emg / FS
                reg["retardo_ms_" + etiqueta] = (t_mov - t_emg) / FS * 1000.0
            if reg.get("retardo_ms_calibracion") is not None:
                n_ambas += 1
            if reg.get("retardo_ms_reposo_previo") is not None:
                n_respaldo += 1
            registros.append(reg)

        resumen_sesiones.append(dict(archivo=nombre, episodios=n_tot,
                                     con_t_mov=n_mov,
                                     con_t_emg_calibracion=n_ambas,
                                     con_t_emg_reposo_previo=n_respaldo,
                                     tiene_calibracion=base_sesion is not None))
        print("  %-32s %3d episodios | %3d con t_mov | t_EMG: %3d con "
              "calibración, %3d con reposo previo"
              % (nombre, n_tot, n_mov, n_ambas, n_respaldo))

    tot = sum(s["episodios"] for s in resumen_sesiones)
    con_mov = sum(s["con_t_mov"] for s in resumen_sesiones)
    con_cal = sum(s["con_t_emg_calibracion"] for s in resumen_sesiones)
    con_rep = sum(s["con_t_emg_reposo_previo"] for s in resumen_sesiones)

    print()
    print("POBLACIONES")
    print("  episodios de las cinco clases activas ................ %d" % tot)
    print("  con cruce sostenido de velocidad (t_mov) ............. %d" % con_mov)
    print("  sin cruce sostenido ................................. %d"
          % (tot - con_mov))
    print("  con t_EMG usando la calibración de la sesión ........ %d" % con_cal)
    print("  con t_EMG usando el reposo inmediatamente previo .... %d" % con_rep)
    print()

    salida = dict(
        criterios=dict(umbral_velocidad_dps=UMBRAL_VEL,
                       persistencia_ms=PERSIST_MS, k_sigma=K_SIGMA,
                       suavizado_ms=SUAVIZADO_MS,
                       busqueda_previa_ms=BUSQUEDA_PREVIA_MS),
        poblaciones=dict(episodios_clases_activas=int(tot),
                         con_t_mov=int(con_mov),
                         sin_t_mov=int(tot - con_mov),
                         con_t_emg_calibracion=int(con_cal),
                         con_t_emg_reposo_previo=int(con_rep)),
        por_sesion=resumen_sesiones,
        episodios=registros)

    for etiqueta, titulo in (("calibracion", "línea base: bloque de calibración"),
                             ("reposo_previo", "línea base: reposo previo")):
        r = np.array([x["retardo_ms_" + etiqueta] for x in registros
                      if x.get("retardo_ms_" + etiqueta) is not None])
        print("RETARDO ELECTROMECÁNICO t_mov − t_EMG, %s" % titulo)
        if len(r) < 5:
            print("  solo %d episodios determinados: no se reporta "
                  "distribución." % len(r))
            salida["retardo_" + etiqueta] = dict(n=int(len(r)))
            print()
            continue
        q1, med, q3 = np.percentile(r, [25, 50, 75])
        print("  n = %d    mediana %.0f ms    IQR de %.0f a %.0f ms"
              % (len(r), med, q1, q3))
        print("  media %.0f ms    desviación típica %.0f ms" % (r.mean(), r.std()))
        print("  la activación precede al movimiento en el %.0f %% de los "
              "episodios" % (100 * (r > 0).mean()))
        print("  en el intervalo de 50 a 100 ms de Trigili et al. (2019): "
              "%d de %d" % (int(((r >= 50) & (r <= 100)).sum()), len(r)))
        por_clase = {}
        for g in GESTOS:
            v = np.array([x["retardo_ms_" + etiqueta] for x in registros
                          if x["clase"] == g
                          and x.get("retardo_ms_" + etiqueta) is not None])
            if len(v):
                por_clase[g] = dict(
                    n=int(len(v)), mediana_ms=round(float(np.median(v))),
                    q1_ms=round(float(np.percentile(v, 25))),
                    q3_ms=round(float(np.percentile(v, 75))))
                print("    %-16s n=%2d  mediana %6.0f ms  IQR %6.0f a %6.0f"
                      % (g, len(v), np.median(v), np.percentile(v, 25),
                         np.percentile(v, 75)))
        salida["retardo_" + etiqueta] = dict(
            n=int(len(r)), mediana=round(float(med)), q1=round(float(q1)),
            q3=round(float(q3)), media=round(float(r.mean())),
            sd=round(float(r.std())),
            fraccion_anticipa=round(float((r > 0).mean()), 3),
            en_intervalo_trigili=int(((r >= 50) & (r <= 100)).sum()),
            por_clase=por_clase)
        print()

    with open("anticipacion_episodios.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=1, ensure_ascii=False)
    print("Escrito anticipacion_episodios.json")


if __name__ == "__main__":
    main()
