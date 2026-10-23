"""Presupuesto de latencia extremo a extremo.

Suma los terminos del retardo entre la activacion muscular y la disponibilidad del
comando asistivo. Escribe presupuesto_latencia.json.
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
RUTA_COMPUTO = sys.argv[2] if len(sys.argv) > 2 else "latencia_computo.json"

# parametros, identicos al pipeline
FS = 1000.0
WIN_MS = 300
HOP_MS = 10
L = 20
BP = (20.0, 450.0)
NOTCH = (60.0, 30.0)
ORDEN = 4

T_H_MS = 25.0        # periodo del supervisor, 06_cadena_supervisora.py
T_EMB_MS = 5.0       # ExoCS.ino linea 99: LOOP_INTERVAL_US = 5000
HX711_MS = 100.0     # ExoCS.ino linea 59


def retardo_de_grupo_ponderado():
    """Retardo de grupo de la cascada digital, ponderado por el espectro
    de potencia medio de la senal realmente registrada."""
    nyq = FS / 2
    b1, a1 = sg.butter(ORDEN, [BP[0] / nyq, BP[1] / nyq], btype="bandpass")
    b2, a2 = sg.iirnotch(NOTCH[0] / nyq, NOTCH[1])

    n = 4096
    w, gd1 = sg.group_delay((b1, a1), w=n, fs=FS)
    _, gd2 = sg.group_delay((b2, a2), w=n, fs=FS)
    gd_ms = (gd1 + gd2) / FS * 1000.0        # muestras -> ms

    rutas = sorted(glob.glob(os.path.join(DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    if not rutas:
        sys.exit("No hay emg_*.csv en " + DIR_DATOS)

    psd_acum = np.zeros(n)
    n_seg = 0
    for r in rutas:
        e = pd.read_csv(r, usecols=["EMG_ch2_raw_bic", "EMG_ch3_raw_tri"],
                        low_memory=False)
        for col in ("EMG_ch2_raw_bic", "EMG_ch3_raw_tri"):
            x = pd.to_numeric(e[col], errors="coerce").dropna().values
            f, p = sg.welch(x, fs=FS, nperseg=2 * (n - 1))
            psd_acum += np.interp(w, f, p)
            n_seg += 1
        print("   espectro de %s" % os.path.basename(r))
    psd = psd_acum / n_seg

    banda = (w >= BP[0]) & (w <= BP[1])
    peso = psd[banda] / psd[banda].sum()
    medio = float((gd_ms[banda] * peso).sum())

    # valores de referencia en frecuencias representativas
    ref = {}
    for f0 in (20, 50, 100, 150, 250, 450):
        i = int(np.argmin(np.abs(w - f0)))
        ref["%d Hz" % f0] = round(float(gd_ms[i]), 3)
    return medio, ref, float(w[banda][int(np.argmax(peso))])


def main():
    print("PASO 11  PRESUPUESTO DE LATENCIA EXTREMO A EXTREMO")
    print()
    print("1b. Retardo de grupo del filtrado digital causal")
    gd_medio, gd_ref, f_pico = retardo_de_grupo_ponderado()
    print()
    print("   retardo de grupo por frecuencia (ms):")
    for k, v in gd_ref.items():
        print("     %-8s %7.3f" % (k, v))
    print("   frecuencia de mayor energia: %.1f Hz" % f_pico)
    print("   RETARDO DE GRUPO PONDERADO POR EL ESPECTRO: %.3f ms" % gd_medio)
    print()

    # termino 3, del cronometraje
    if not os.path.exists(RUTA_COMPUTO):
        sys.exit("Falta %s. Corre antes 08_medir_latencia_computo.py."
                 % RUTA_COMPUTO)
    with open(RUTA_COMPUTO, encoding="utf-8") as fh:
        comp = json.load(fh)
    c = comp["ciclo_completo"]
    print("3. Computo, de %s: mediana %.3f ms, p95 %.3f, p99 %.3f"
          % (RUTA_COMPUTO, c["mediana_ms"], c["p95_ms"], c["p99_ms"]))
    print()

    # terminos deterministas
    cobertura_ms = (L - 1) * HOP_MS + WIN_MS

    terminos = [
        dict(n="1a", nombre="Adquisición analógica de la señal mioeléctrica",
             mediana=None, peor=None, origen="no instrumentado; exige medir "
             "la respuesta del circuito del sensor",
             en_la_cadena=True),
        dict(n="1b", nombre="Filtrado digital causal, retardo de grupo",
             mediana=round(gd_medio, 3), peor=round(gd_medio, 3),
             origen="calculado sobre la cascada del pipeline y ponderado por "
                    "el espectro de los cinco registros",
             en_la_cadena=True),
        dict(n="1c", nombre="Adquisición del canal de fuerza",
             mediana=HX711_MS, peor=HX711_MS,
             origen="ExoCS.ino línea 59, HX711 fijado a 10 muestras por "
                    "segundo",
             en_la_cadena=False),
        dict(n="2", nombre="Acumulación de la ventana de análisis",
             mediana=float(WIN_MS), peor=float(WIN_MS),
             origen="determinista, T_w del apartado 3.5.2",
             en_la_cadena=True),
        dict(n="2b", nombre="Cobertura temporal de la secuencia completa",
             mediana=float(cobertura_ms), peor=float(cobertura_ms),
             origen="determinista, (L-1)·T_s + T_w; cota de arranque en frío, "
                    "no latencia de operación continua",
             en_la_cadena=False),
        dict(n="3", nombre="Cómputo de un ciclo del supervisor",
             mediana=c["mediana_ms"], peor=c["p99_ms"],
             origen="medido, 2000 ciclos cronometrados",
             en_la_cadena=True),
        dict(n="4", nombre="Comunicación con el nivel embebido",
             mediana=None, peor=None,
             origen="no instrumentado; exige marcas de tiempo en ambos "
                    "extremos del enlace",
             en_la_cadena=True),
        dict(n="5", nombre="Periodo de actualización del supervisor",
             mediana=T_H_MS / 2, peor=T_H_MS,
             origen="determinista, retardo de muestreo uniforme en [0, T_H]",
             en_la_cadena=True),
        dict(n="6", nombre="Periodo del nivel embebido",
             mediana=T_EMB_MS / 2, peor=T_EMB_MS,
             origen="determinista, ExoCS.ino línea 99, 5 ms",
             en_la_cadena=True),
    ]

    print("PRESUPUESTO")
    print("%-4s %-46s %10s %10s" % ("", "término", "mediana", "peor caso"))
    for t in terminos:
        m = "%.3f" % t["mediana"] if t["mediana"] is not None else "no instr."
        p = "%.3f" % t["peor"] if t["peor"] is not None else "no instr."
        print("%-4s %-46s %10s %10s" % (t["n"], t["nombre"][:46], m, p))

    en_cadena = [t for t in terminos if t["en_la_cadena"]]
    conocidos = [t for t in en_cadena if t["mediana"] is not None]
    faltan = [t for t in en_cadena if t["mediana"] is None]
    suma_med = sum(t["mediana"] for t in conocidos)
    suma_peor = sum(t["peor"] for t in conocidos)

    regimen = [t for t in conocidos if t["n"] in ("1b", "3", "5", "6")]
    sr_med = sum(t["mediana"] for t in regimen)
    sr_peor = sum(t["peor"] for t in regimen)

    print("A. COTA DE ARRANQUE, tal como el apartado 3.8.2 define el término 2")
    print("   Suma de los términos conocidos de la cadena de decisión:")
    print("   mediana %.3f ms      peor caso %.3f ms" % (suma_med, suma_peor))
    dominante = max(conocidos, key=lambda t: t["mediana"])
    print("   Término dominante: %s, %s, con %.3f ms, el %.1f %% de la suma."
          % (dominante["n"], dominante["nombre"], dominante["mediana"],
             100 * dominante["mediana"] / suma_med))
    print()
    print("B. SOBRECOSTE DE PROCESAMIENTO EN RÉGIMEN PERMANENTE")
    print("   Términos que se pagan en cada decisión: 1b, 3, 5 y 6.")
    print("   mediana %.3f ms      peor caso %.3f ms" % (sr_med, sr_peor))
    print("   El término 2 no entra aquí: una vez la ventana está llena, el")
    print("   supervisor decide cada %d ms sobre la ventana más reciente y no"
          % HOP_MS)
    print("   vuelve a esperar. El retardo con que esa decisión responde a un")
    print("   cambio de la señal es una magnitud empírica distinta, que mide")
    print("   el análisis de anticipación del apartado 3.8.1.")
    print()
    print("   Términos sin instrumentar en ambos casos: %s"
          % ", ".join("%s (%s)" % (t["n"], t["nombre"][:34]) for t in faltan))
    print()
    print("C. CONTRASTE CON EL CRITERIO EXTERNO")
    print("   Farrell y Weir (2007) sitúan entre 100 y 125 ms el límite de lo")
    print("   percibido como natural en control mioeléctrico.")
    print("   La cota de arranque lo %s (%.1f ms)."
          % ("EXCEDE" if suma_med > 125 else "respeta", suma_med))
    print("   El sobrecoste en régimen lo %s (%.1f ms), con un margen de %.1f "
          "ms" % ("excede" if sr_med > 125 else "RESPETA", sr_med, 100 - sr_med))
    print("   sobre el extremo inferior del intervalo, antes de añadir el")
    print("   término de comunicación, que no se instrumentó.")

    salida = dict(
        nota="Presupuesto de latencia del apartado 3.8.2. Los términos 1a y 4 "
             "no se instrumentaron durante la adquisición y se declaran como "
             "tales; no se estiman.",
        retardo_de_grupo_por_frecuencia_ms=gd_ref,
        frecuencia_de_mayor_energia_hz=round(f_pico, 1),
        terminos=terminos,
        cota_de_arranque_ms=dict(mediana=round(suma_med, 3),
                                 peor_caso=round(suma_peor, 3),
                                 incluye=[t["n"] for t in conocidos]),
        sobrecoste_en_regimen_ms=dict(mediana=round(sr_med, 3),
                                      peor_caso=round(sr_peor, 3),
                                      incluye=[t["n"] for t in regimen]),
        terminos_sin_instrumentar=[t["n"] for t in faltan],
        criterio_farrell_weir_ms=[100, 125])
    with open("presupuesto_latencia.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=1, ensure_ascii=False)
    print()
    print("Escrito presupuesto_latencia.json")


if __name__ == "__main__":
    main()
