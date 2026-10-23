"""Cuantifica el sesgo que introducen los bloques de calibracion ejecutados con carga.

Escribe factor_sesgo_calibracion.json.
"""
# -*- coding: utf-8 -*-
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

CARPETA = sys.argv[1] if len(sys.argv) > 1 else "resultados_R15"
CARACTERISTICA = "IAV_bic"
N_REMUESTREOS = 2000
SEMILLA = 0

SIN_CARGA = "calibracion"
CON_CARGA = "calibracion_con_carga"


def cargar(carpeta):
    rutas = sorted(glob.glob(os.path.join(carpeta, "featok_*.csv")))
    if not rutas:
        sys.exit("No hay featok_*.csv en %r" % carpeta)
    datos = {}
    for r in rutas:
        sesion = os.path.basename(r)[len("featok_"):-len(".csv")]
        datos[sesion] = pd.read_csv(r, low_memory=False)
    return datos


def bloque(df, etiqueta):
    sel = df[df["label_str"] == etiqueta][CARACTERISTICA]
    return pd.to_numeric(sel, errors="coerce").dropna().values


def ic_del_cociente(num, den, n=N_REMUESTREOS, semilla=SEMILLA):
    """Intervalo de confianza del cociente de medianas por remuestreo."""
    rng = np.random.default_rng(semilla)
    b = [np.median(rng.choice(num, len(num))) / np.median(rng.choice(den, len(den)))
         for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def main():
    datos = cargar(CARPETA)

    print("Bloques de calibracion presentes en cada sesion")
    referencia = []
    for sesion in sorted(datos):
        df = datos[sesion]
        for etiqueta in (SIN_CARGA, CON_CARGA):
            x = bloque(df, etiqueta)
            if len(x) == 0:
                continue
            print("  %s  %-22s n=%6d  mediana=%10.1f  media=%11.1f  media/mediana=%5.1f"
                  % (sesion, etiqueta, len(x), np.median(x), x.mean(),
                     x.mean() / np.median(x)))
            if etiqueta == SIN_CARGA:
                referencia.append(x)
    print()

    ref = np.concatenate(referencia)
    q1, med_ref, q3 = np.percentile(ref, [25, 50, 75])
    print("REFERENCIA, ventanas de calibracion ejecutadas sin carga")
    print("  n = %d    Q1 = %.1f    mediana = %.1f    Q3 = %.1f"
          % (len(ref), q1, med_ref, q3))
    print()

    salida = {
        "caracteristica": CARACTERISTICA,
        "referencia_sin_carga": {
            "n": int(len(ref)), "q1": round(float(q1), 1),
            "mediana": round(float(med_ref), 1), "q3": round(float(q3), 1),
        },
    }

    # Sesion de 0,85 kg: no tiene bloque propio sin carga. Comparacion entre
    # sesiones, y asi debe declararse.
    s085 = [s for s in datos if len(bloque(datos[s], CON_CARGA))
            and not len(bloque(datos[s], SIN_CARGA))]
    for s in s085:
        x = bloque(datos[s], CON_CARGA)
        f = np.median(x) / med_ref
        lo, hi = ic_del_cociente(x, ref)
        print("SESION %s, bloque integro ejecutado con la carga puesta" % s)
        print("  Esta sesion NO tiene ninguna ventana etiquetada %r." % SIN_CARGA)
        print("  n = %d    mediana = %.1f" % (len(x), np.median(x)))
        print("  FACTOR frente a la referencia = %.1f   IC95 %.1f a %.1f"
              % (f, lo, hi))
        print("  Comparacion ENTRE SESIONES: el factor recoge a la vez el efecto")
        print("  de la carga y la variacion de colocacion de los electrodos.")
        print()
        salida["entre_sesiones"] = {
            "sesion": s, "n": int(len(x)), "mediana": round(float(np.median(x)), 1),
            "factor": round(float(f), 1),
            "ic95": [round(float(lo), 1), round(float(hi), 1)],
        }

    # Sesiones con ambos bloques: contraste interno, misma colocacion.
    for s in sorted(datos):
        a, b = bloque(datos[s], SIN_CARGA), bloque(datos[s], CON_CARGA)
        if len(a) == 0 or len(b) == 0:
            continue
        f = np.median(b) / np.median(a)
        lo, hi = ic_del_cociente(b, a)
        print("SESION %s, contraste INTERNO" % s)
        print("  tramo valido  n=%5d  mediana = %.1f" % (len(a), np.median(a)))
        print("  con carga     n=%5d  mediana = %.1f" % (len(b), np.median(b)))
        print("  FACTOR = %.1f   IC95 %.1f a %.1f" % (f, lo, hi))
        print("  Misma sesion y misma colocacion: el factor aisla el efecto de")
        print("  la carga sobre la linea base.")
        print()
        salida["intra_sesion"] = {
            "sesion": s, "n_valido": int(len(a)), "mediana_valido": round(float(np.median(a)), 1),
            "n_con_carga": int(len(b)), "mediana_con_carga": round(float(np.median(b)), 1),
            "factor": round(float(f), 1),
            "ic95": [round(float(lo), 1), round(float(hi), 1)],
        }

    with open("factor_sesgo_calibracion.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=1, ensure_ascii=False)
    print("Escrito factor_sesgo_calibracion.json")


if __name__ == "__main__":
    main()
