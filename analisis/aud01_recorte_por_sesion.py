"""Comprueba la causa del recorte por saturacion, sesion por sesion.
"""
# -*- coding: utf-8 -*-
import glob, json, os
import numpy as np
import pandas as pd

RUTA = "/mnt/user-data/uploads/outputs/entrenamiento de la red/resultados_R15"
EMG = ["IAV_bic","SSI_bic","WL_bic","LOG_bic","IAV_tri","SSI_tri","WL_tri","LOG_tri"]
SAT = 10.0
EXCL = ["calibracion","calibracion_con_carga","bajada","STOPPED"]

filas = []
for f in sorted(glob.glob(os.path.join(RUTA, "featok_*.csv"))):
    df = pd.read_csv(f)
    ses = os.path.basename(f)[7:-4]
    cal = df[df["label_str"] == "calibracion"]
    if len(cal) == 0:
        cal = df[df["label_str"] == "calibracion_con_carga"]
    mu = cal[EMG].mean().values
    sd = cal[EMG].std(ddof=0).values
    act = df[~df["label_str"].isin(EXCL)]
    z = (act[EMG].values - mu) / sd
    fuera = np.abs(z) > SAT
    filas.append(dict(sesion=ses, carga=df["load_kg"].iloc[0],
                      sigma_IAV=round(float(sd[0]), 1),
                      mu_IAV=round(float(mu[0]), 1),
                      valores=round(100.0 * fuera.mean(), 2),
                      ventanas=round(100.0 * fuera.any(1).mean(), 1),
                      zmax=round(float(np.abs(z).max()), 0)))
t = pd.DataFrame(filas)
print(t.to_string(index=False))
print()
print("razon entre la sigma mayor y la menor: %.1f"
      % (t.sigma_IAV.max() / t.sigma_IAV.min()))
print("la fraccion de ventanas recortadas va del %.1f al %.1f por ciento"
      % (t.ventanas.min(), t.ventanas.max()))
json.dump(filas, open("/home/claude/work/recorte_por_sesion.json","w"),
          ensure_ascii=False, indent=1)
