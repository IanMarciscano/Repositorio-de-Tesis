"""Comprueba el umbral de inicio mecanico frente a la velocidad articular observada.
"""
# -*- coding: utf-8 -*-
import glob, json, os
import numpy as np
import pandas as pd

R = "/mnt/user-data/uploads/outputs/entrenamiento de la red/resultados_R15"
UMBRAL = 5.0
GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro",
          "levantamiento", "bajada"]

filas = []
for f in sorted(glob.glob(os.path.join(R, "featok_*.csv"))):
    df = pd.read_csv(f, low_memory=False)
    ses = os.path.basename(f)[7:-4]
    # el reposo son los tramos iniciales de cada bloque etiquetado, antes de
    # que el gesto comience: se identifican por el tiempo desde la etiqueta
    if "t_since_label_s" not in df.columns:
        continue
    for recorte in (0.0, 2.0, 3.0, 4.0):
        m = (df["label_str"].isin(GESTOS)) & (df["t_since_label_s"] >= recorte) \
            & (df["t_since_label_s"] < 8.0)
        v = df.loc[m, "dq_el"].abs()
        if len(v) == 0:
            continue
        filas.append(dict(sesion=ses, recorte_s=recorte, n=int(len(v)),
                          supera=round(100.0 * float((v > UMBRAL).mean()), 2),
                          p95=round(float(np.percentile(v, 95)), 2),
                          maximo=round(float(v.max()), 2)))
t = pd.DataFrame(filas)
print("Velocidad angular del codo dentro de los bloques etiquetados,")
print("por recorte inicial aplicado. Umbral de inicio: %.0f grados por segundo.\n" % UMBRAL)
for rec in sorted(t.recorte_s.unique()):
    s = t[t.recorte_s == rec]
    print("recorte %.0f s -> supera el umbral el %.2f %% de las muestras "
          "(mediana entre sesiones), p95 %.2f, maximo %.2f"
          % (rec, s.supera.median(), s.p95.median(), s.maximo.max()))
print()
print(t.to_string(index=False))
json.dump(filas, open("/home/claude/work/umbral_inicio.json", "w"),
          ensure_ascii=False, indent=1)
