"""Falsos positivos previos al movimiento y efecto de la histeresis.

Escribe falsos_positivos.json.
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
JSON_P12 = sys.argv[3] if len(sys.argv) > 3 else "anticipacion_episodios.json"

FS = 1000.0
THETA_ON = 0.60          # 06_cadena_supervisora.py
TOLERANCIA_S = 0.490     # (L-1)*T_s + T_w
GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro",
          "levantamiento"]


def flancos(mascara):
    """Índices donde la máscara pasa de falso a verdadero."""
    m = np.asarray(mascara).astype(bool)
    return np.flatnonzero(m & ~np.r_[False, m[:-1]])


def main():
    if not os.path.exists(JSON_P12):
        sys.exit("Falta %s. Corre antes 12_anticipacion_episodios.py."
                 % JSON_P12)
    with open(JSON_P12, encoding="utf-8") as fh:
        p12 = json.load(fh)
    t_mov_por_sesion = {}
    for ep in p12["episodios"]:
        if ep["t_mov_s"] is not None:
            t_mov_por_sesion.setdefault(ep["sesion"], []).append(ep["t_mov_s"])

    rutas = sorted(glob.glob(os.path.join(DIR_P6, "cadena_*.npz")))
    if not rutas:
        sys.exit("No hay cadena_*.npz en " + DIR_P6)

    filas = []
    print("%-18s %8s %9s %9s %9s %9s"
          % ("sesión", "reposo", "cruda", "por min", "tras MEF", "por min"))
    for ruta in rutas:
        ses = os.path.basename(ruta)[len("cadena_"):-len(".npz")]
        z = np.load(ruta, allow_pickle=True)
        t, p_int, estado = z["t"], z["p_int"], z["estado"]

        csv = os.path.join(DIR_DATOS, "emg_%s.csv" % ses)
        if not os.path.exists(csv):
            print("  %s sin registro crudo, se omite" % ses)
            continue
        e = pd.read_csv(csv, usecols=["label"], low_memory=False)
        lab = e["label"].fillna("reposo").astype(str).values

        # etiqueta de cada instante del supervisor
        idx = np.clip((t * FS).astype(int), 0, len(lab) - 1)
        etiqueta = lab[idx]
        en_reposo = etiqueta == "reposo"
        dt = float(np.median(np.diff(t)))
        seg_reposo = float(en_reposo.sum() * dt)

        t_movs = np.array(sorted(t_mov_por_sesion.get("emg_%s.csv" % ses, [])))

        def espurias(mascara_activa):
            """Activaciones que nacen en reposo y no van seguidas de un
            inicio mecánico dentro de la ventana de tolerancia."""
            ini = flancos(mascara_activa)
            ini = ini[en_reposo[ini]]
            if len(t_movs) == 0:
                return ini, ini
            legitimas = []
            for i in ini:
                ventana = (t_movs >= t[i]) & (t_movs <= t[i] + TOLERANCIA_S)
                legitimas.append(bool(ventana.any()))
            legitimas = np.array(legitimas, dtype=bool)
            return ini, ini[~legitimas]

        ini_cruda, esp_cruda = espurias(p_int >= THETA_ON)
        ini_mef, esp_mef = espurias(estado > 0)

        # proporción de episodios precedidos por al menos una espuria
        cortes = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        cortes = np.r_[cortes, len(lab)]
        eps = 0
        con_prev = {"cruda": 0, "mef": 0}
        for k in range(1, len(cortes) - 1):
            i0 = cortes[k]
            if lab[i0] not in GESTOS:
                continue
            eps += 1
            a, b = cortes[k - 1] / FS, i0 / FS      # tramo de reposo previo
            for nombre, esp in (("cruda", esp_cruda), ("mef", esp_mef)):
                if len(esp) and ((t[esp] >= a) & (t[esp] < b)).any():
                    con_prev[nombre] += 1

        fila = dict(
            sesion=ses,
            segundos_de_reposo=round(seg_reposo, 1),
            fraccion_reposo_sobre_umbral=round(float(
                (p_int[en_reposo] >= THETA_ON).mean()), 3) if en_reposo.any() else None,
            cruda=dict(activaciones_en_reposo=int(len(ini_cruda)),
                       espurias=int(len(esp_cruda)),
                       espurias_por_minuto=round(
                           60 * len(esp_cruda) / seg_reposo, 2) if seg_reposo else None),
            tras_maquina_de_estados=dict(
                activaciones_en_reposo=int(len(ini_mef)),
                espurias=int(len(esp_mef)),
                espurias_por_minuto=round(
                    60 * len(esp_mef) / seg_reposo, 2) if seg_reposo else None),
            episodios=int(eps),
            episodios_precedidos_cruda=int(con_prev["cruda"]),
            episodios_precedidos_mef=int(con_prev["mef"]))
        filas.append(fila)
        print("%-18s %7.0f s %9d %9.2f %9d %9.2f"
              % (ses, seg_reposo, len(esp_cruda),
                 60 * len(esp_cruda) / seg_reposo, len(esp_mef),
                 60 * len(esp_mef) / seg_reposo))

    rep = sum(f["segundos_de_reposo"] for f in filas)
    ec = sum(f["cruda"]["espurias"] for f in filas)
    em = sum(f["tras_maquina_de_estados"]["espurias"] for f in filas)
    eps = sum(f["episodios"] for f in filas)
    pc = sum(f["episodios_precedidos_cruda"] for f in filas)
    pm = sum(f["episodios_precedidos_mef"] for f in filas)

    print()
    print("CONJUNTO DE LAS %d SESIONES SIMULADAS" % len(filas))
    print("  tiempo total de reposo ......................... %.0f s (%.1f min)"
          % (rep, rep / 60))
    print()
    print("  SEÑAL CRUDA de la cabeza auxiliar")
    print("    activaciones espurias ........................ %d" % ec)
    print("    por minuto de reposo ......................... %.2f" % (60 * ec / rep))
    print("    episodios precedidos de al menos una .......... %d de %d (%.0f %%)"
          % (pc, eps, 100 * pc / eps))
    print()
    print("  TRAS LA MÁQUINA DE ESTADOS")
    print("    activaciones espurias ........................ %d" % em)
    print("    por minuto de reposo ......................... %.2f" % (60 * em / rep))
    print("    episodios precedidos de al menos una .......... %d de %d (%.0f %%)"
          % (pm, eps, 100 * pm / eps))
    print()
    if ec:
        print("  REDUCCIÓN que aporta la exigencia de persistencia:")
        print("    %.1f %% de las activaciones espurias,"
              % (100 * (1 - em / ec)))
        print("    y %.1f %% de los episodios precedidos."
              % (100 * (1 - pm / pc) if pc else 0.0))
    fr = np.mean([f["fraccion_reposo_sobre_umbral"] for f in filas
                  if f["fraccion_reposo_sobre_umbral"] is not None])
    print()
    print("  Contexto: durante los tramos de reposo la probabilidad de")
    print("  intención supera el umbral de activación el %.0f %% del tiempo."
          % (100 * fr))

    salida = dict(
        definicion=dict(theta_on=THETA_ON, tolerancia_s=TOLERANCIA_S,
                        senal="cabeza auxiliar de intención"),
        por_sesion=filas,
        global_=dict(
            segundos_de_reposo=round(rep, 1),
            episodios=int(eps),
            cruda=dict(espurias=int(ec),
                       por_minuto=round(60 * ec / rep, 2),
                       episodios_precedidos=int(pc),
                       proporcion_episodios=round(pc / eps, 3)),
            tras_maquina_de_estados=dict(
                espurias=int(em), por_minuto=round(60 * em / rep, 2),
                episodios_precedidos=int(pm),
                proporcion_episodios=round(pm / eps, 3)),
            reduccion_por_persistencia=round(1 - em / ec, 3) if ec else None,
            fraccion_reposo_sobre_umbral=round(float(fr), 3)))
    with open("falsos_positivos.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=1, ensure_ascii=False)
    print()
    print("Escrito falsos_positivos.json")


if __name__ == "__main__":
    main()
