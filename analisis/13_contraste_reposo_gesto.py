"""Contraste de amplitud entre reposo y gesto, por canal y por sesion.

Escribe contraste_reposo_gesto.json.
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
SUAVIZADO_MS = 50
K_SIGMA = 3.0
GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro",
          "levantamiento"]
CANALES = {"bíceps": "EMG_ch2_raw_bic", "tríceps": "EMG_ch3_raw_tri"}


def filtrar(x):
    nyq = FS / 2
    b1, a1 = sg.butter(ORDEN, [BP[0] / nyq, BP[1] / nyq], btype="bandpass")
    b2, a2 = sg.iirnotch(NOTCH[0] / nyq, NOTCH[1])
    return sg.lfilter(b2, a2, sg.lfilter(b1, a1, x.astype(np.float64)))


def envolvente(x):
    n = int(FS * SUAVIZADO_MS / 1000)
    return np.convolve(np.abs(x), np.ones(n) / n, mode="full")[:len(x)]


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_DATOS, "**", "emg_*.csv"),
                             recursive=True))
    if not rutas:
        sys.exit("No hay emg_*.csv en " + DIR_DATOS)

    salida = {}
    for nombre, col in CANALES.items():
        print()
        print("CANAL DEL %s" % nombre.upper())
        print("%-18s %9s %9s %9s %9s %9s %10s"
              % ("sesión", "calibr.", "reposo", "gesto", "rep/cal",
                 "gesto/rep", "reposo>u"))
        filas = []
        for r in rutas:
            e = pd.read_csv(r, usecols=["label", col], low_memory=False)
            # el reposo es la AUSENCIA de etiqueta en estos registros
            lab = e["label"].fillna("reposo").astype(str).values
            E = envolvente(filtrar(pd.to_numeric(e[col], errors="coerce")
                                   .fillna(0).values))
            cal = lab == "calibracion"
            rep = lab == "reposo"
            ges = np.isin(lab, GESTOS)
            if cal.sum() < FS or rep.sum() < FS or ges.sum() < FS:
                print("%-18s datos insuficientes" % os.path.basename(r)[4:-4])
                continue
            u = E[cal].mean() + K_SIGMA * E[cal].std()
            m_cal, m_rep, m_ges = (float(np.median(E[cal])),
                                   float(np.median(E[rep])),
                                   float(np.median(E[ges])))
            frac = float((E[rep] > u).mean())
            print("%-18s %9.1f %9.1f %9.1f %9.2f %9.2f %9.1f %%"
                  % (os.path.basename(r)[4:-4], m_cal, m_rep, m_ges,
                     m_rep / m_cal, m_ges / m_rep, 100 * frac))
            filas.append(dict(sesion=os.path.basename(r)[4:-4],
                              mediana_calibracion=round(m_cal, 1),
                              mediana_reposo=round(m_rep, 1),
                              mediana_gesto=round(m_ges, 1),
                              razon_reposo_calibracion=round(m_rep / m_cal, 2),
                              razon_gesto_reposo=round(m_ges / m_rep, 2),
                              fraccion_reposo_sobre_umbral=round(frac, 3)))
        salida[nombre] = filas
        if filas:
            rgr = [f["razon_gesto_reposo"] for f in filas]
            rrc = [f["razon_reposo_calibracion"] for f in filas]
            print("   razón gesto/reposo: de %.2f a %.2f, mediana %.2f"
                  % (min(rgr), max(rgr), float(np.median(rgr))))
            print("   razón reposo/calibración: de %.2f a %.2f"
                  % (min(rrc), max(rrc)))

    print()
    print("LECTURA")
    print("  El reposo del protocolo no es reposo muscular: el participante")
    print("  sostiene la postura contra la gravedad, de modo que su amplitud")
    print("  basal supera a la del bloque de calibración, tomado con el brazo")
    print("  relajado. La consecuencia es que el contraste entre gesto y")
    print("  reposo, que es la información de la que dispone cualquier")
    print("  detector, resulta muy inferior al contraste entre gesto y")
    print("  calibración, que es el que sugeriría una lectura ingenua.")

    with open("contraste_reposo_gesto.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=1, ensure_ascii=False)
    print()
    print("Escrito contraste_reposo_gesto.json")


if __name__ == "__main__":
    main()
