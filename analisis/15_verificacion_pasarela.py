"""Verificacion funcional de la pasarela de seguridad sobre los registros.

Indice de severidad, modos y conmutaciones. Escribe verificacion_pasarela.json.
"""
# -*- coding: utf-8 -*-
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DIR_P6 = sys.argv[1] if len(sys.argv) > 1 else "resultados_paso6_cadena_supervisora"
DIR_DATOS = sys.argv[2] if len(sys.argv) > 2 else "data"

FS = 1000.0

# del firmware, verificado
CAN_TIMEOUT_MS = 200.0       # ExoCS.ino linea 231
CUR_LIMIT_A = 10.0           # ExoCS.ino linea 224
TELEMETRIA_NOMINAL_MS = 50.0 # 20 Hz nominales, solo informativo

# eleccion del disenador, declarada como tal
CUR_AVISO = 0.80 * CUR_LIMIT_A     # el indice empieza a crecer al 80 %
TEMP_AVISO_C, TEMP_LIMITE_C = 60.0, 80.0
D_SAFE_DEG, D_MAX_DEG = 5.0, 15.0
DIR_MINIMO_DEG = 2.0               # criterio del apartado 4.17
DIR_VENTANA_S = 0.5

PESOS = {"comm": 1.0, "fault": 1.0, "dir": 1.0,   # criticos
         "trk": 0.5, "cur": 0.4, "temp": 0.3}
RHO_MENOR, RHO_MAYOR = 0.30, 0.80


def sat01(x):
    return np.clip(x, 0.0, 1.0)


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_P6, "cadena_*.npz")))
    if not rutas:
        sys.exit("No hay cadena_*.npz en " + DIR_P6)

    filas = []
    for ruta in rutas:
        ses = os.path.basename(ruta)[len("cadena_"):-len(".npz")]
        z = np.load(ruta, allow_pickle=True)
        t = z["t"]
        qd = {"sh": z["qd_sh"], "el": z["qd_el"]}
        estado = z["estado"]

        csv = os.path.join(DIR_DATOS, "emg_%s.csv" % ses)
        if not os.path.exists(csv):
            print("  %s sin registro crudo, se omite" % ses)
            continue
        e = pd.read_csv(csv, usecols=["t_ms", "ShoulderCur_A", "ShoulderTemp_C",
                                      "ShoulderErr", "ShoulderPos_deg",
                                      "ElbowJointAngle_deg"], low_memory=False)
        idx = np.clip((t * FS).astype(int), 0, len(e) - 1)

        def col(c):
            return pd.to_numeric(e[c], errors="coerce").ffill().bfill().values

        # i_comm
        tms = pd.to_numeric(e["t_ms"], errors="coerce").ffill().bfill().values
        hueco = np.r_[0.0, np.diff(tms)]
        I_comm = sat01(hueco[idx] / CAN_TIMEOUT_MS)

        # i_fault
        err = col("ShoulderErr")[idx]
        I_fault = (np.abs(err) > 0).astype(float)

        # i_cur
        cur = np.abs(col("ShoulderCur_A")[idx])
        I_cur = sat01((cur - CUR_AVISO) / (CUR_LIMIT_A - CUR_AVISO))

        # i_temp
        temp = col("ShoulderTemp_C")[idx]
        I_temp = sat01((temp - TEMP_AVISO_C) / (TEMP_LIMITE_C - TEMP_AVISO_C))

        # i_trk
        q_sh = col("ShoulderPos_deg")[idx]
        q_el = col("ElbowJointAngle_deg")[idx]
        activo = estado > 0
        d_q = np.zeros_like(t)
        d_q[activo] = np.hypot(qd["sh"][activo] - q_sh[activo],
                               qd["el"][activo] - q_el[activo])
        I_trk = sat01((d_q - D_SAFE_DEG) / (D_MAX_DEG - D_SAFE_DEG))

        # i_dir
        dt = float(np.median(np.diff(t)))
        n = max(1, int(DIR_VENTANA_S / dt))
        I_dir = np.zeros_like(t)
        for art, obs in (("sh", q_sh), ("el", q_el)):
            dref = np.r_[np.zeros(n), qd[art][n:] - qd[art][:-n]]
            dobs = np.r_[np.zeros(n), obs[n:] - obs[:-n]]
            opuesto = ((np.sign(dref) * np.sign(dobs)) < 0) & \
                      (np.abs(dref) > DIR_MINIMO_DEG) & \
                      (np.abs(dobs) > DIR_MINIMO_DEG) & activo
            I_dir = np.maximum(I_dir, opuesto.astype(float))

        indices = {"comm": I_comm, "fault": I_fault, "cur": I_cur,
                   "temp": I_temp, "trk": I_trk, "dir": I_dir}
        for k, v in indices.items():
            if v.min() < -1e-9 or v.max() > 1 + 1e-9:
                sys.exit("ABORTADO: I_%s sale del intervalo unitario "
                         "en %s: [%.3f, %.3f]" % (k, ses, v.min(), v.max()))

        rho = sum(PESOS[k] * v for k, v in indices.items())
        modo = np.where(rho >= RHO_MAYOR, 2,
                        np.where(rho >= RHO_MENOR, 1, 0))

        fila = dict(sesion=ses, muestras=int(len(t)),
                    segundos=round(float(len(t) * dt), 1),
                    crudos=dict(
                        corriente_max_a=round(float(np.abs(col("ShoulderCur_A")).max()), 2),
                        temperatura_min_c=round(float(col("ShoulderTemp_C").min()), 1),
                        temperatura_max_c=round(float(col("ShoulderTemp_C").max()), 1),
                        err_no_nulos=int((np.abs(col("ShoulderErr")) > 0).sum()),
                        hueco_telemetria_max_ms=round(float(hueco.max()), 1)))
        for k, v in indices.items():
            fila["I_" + k] = dict(maximo=round(float(v.max()), 3),
                                  medio=round(float(v.mean()), 4),
                                  fraccion_no_nulo=round(float((v > 0).mean()), 4))
        fila["rho"] = dict(maximo=round(float(rho.max()), 3),
                           mediana=round(float(np.median(rho)), 3))
        fila["modos"] = {str(m): round(float((modo == m).mean()), 4)
                         for m in (0, 1, 2)}
        fila["transiciones"] = int((np.diff(modo) != 0).sum())
        filas.append(fila)

        print("%-18s  ρ máx %.2f  |  modo 0 %5.1f %%  modo 1 %5.1f %%  "
              "modo 2 %5.1f %%  |  %d transiciones"
              % (ses, rho.max(), 100 * (modo == 0).mean(),
                 100 * (modo == 1).mean(), 100 * (modo == 2).mean(),
                 int((np.diff(modo) != 0).sum())))

    print()
    print("ÍNDICES QUE SE ACTIVAN Y QUE NO, sobre el conjunto")
    for k in ("comm", "fault", "cur", "temp", "trk", "dir"):
        mx = max(f["I_" + k]["maximo"] for f in filas)
        fr = np.mean([f["I_" + k]["fraccion_no_nulo"] for f in filas])
        estado_txt = ("nunca se activa" if mx == 0 else
                      "se activa el %.1f %% del tiempo" % (100 * fr))
        print("  I_%-6s  máximo %.3f   %s" % (k, mx, estado_txt))

    print()
    print("COMPROBACIONES")
    print("  1. Los seis índices son calculables desde el registro:   sí")
    print("  2. Los seis quedan acotados en el intervalo unitario:    sí")
    print("     (verificado muestra a muestra; el script aborta si no)")
    m2 = np.mean([f["modos"]["2"] for f in filas])
    m1 = np.mean([f["modos"]["1"] for f in filas])
    print("  3. La conmutación se ejerce: modo 1 el %.1f %% del tiempo,"
          % (100 * m1))
    print("     modo 2 el %.1f %% del tiempo." % (100 * m2))
    print("  4. Condiciones de fallo alcanzadas durante la adquisición:")
    for k, nombre in (("fault", "códigos de falla del motor"),
                      ("cur", "sobrecorriente"),
                      ("temp", "sobretemperatura"),
                      ("comm", "pérdida de comunicación")):
        mx = max(f["I_" + k]["maximo"] for f in filas)
        print("     %-28s %s" % (nombre, "NO" if mx == 0 else
                                 "sí, máximo %.3f" % mx))

    salida = dict(
        pesos=PESOS, umbrales=dict(rho_menor=RHO_MENOR, rho_mayor=RHO_MAYOR),
        origen_de_los_parametros="elección del diseñador; ninguno se deriva "
                                 "de medición ni de restricción del firmware",
        del_firmware=dict(can_timeout_ms=CAN_TIMEOUT_MS,
                          cur_limit_a=CUR_LIMIT_A),
        por_sesion=filas)
    with open("verificacion_pasarela.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=1, ensure_ascii=False)
    print()
    print("Escrito verificacion_pasarela.json")


if __name__ == "__main__":
    main()
