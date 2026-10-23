"""Ejecuta la cadena supervisora completa en lazo abierto sobre los registros.

Maquina de estados de doble umbral, generador de trayectoria de minima sacudida y
ley de asistencia segun necesidad. Escribe resumen_cadena.json.
"""
# -*- coding: utf-8 -*-
import os, glob, json
import numpy as np
import pandas as pd

SEMILLA = 42
np.random.seed(SEMILLA)

# parametros del supervisor
FS        = 1000.0
T_H       = 0.025
THETA_ON  = 0.60
THETA_OFF = 0.42
T_ON, T_OFF = 0.100, 0.300
N_ON  = int(round(T_ON  / T_H))
N_OFF = int(round(T_OFF / T_H))
T_MOV = 4.6
TAU   = 2.0

EPS_Q = {"sh": 1.9, "el": 4.1}
EPS_V = {"sh": 3.1, "el": 7.8}
EPS_F = 427.0

CAP_V   = {"sh": 15.0, "el": 30.0}
ROM_SH  = (-40.0, 30.0)
SERVO   = (-30.0, 50.0)
ESCALAR_ALCANCE = True
PLANO_MAX  = 0.50   # s; en las sesiones sanas el tramo plano no supera 0.17 s
R_MIN_IMU  = 0.90   # correlacion servo vs angulo absoluto de la IMU

W_E, W_F     = 0.2, 0.8
ETA_0, ETA_1 = 0.10, 0.60

CLASES = ["reposo", "flex_codo", "ext_codo", "flex_hombro",
          "ext_hombro", "levantamiento"]
GESTOS = CLASES[1:]
DQ_F = {"flex_codo":     {"sh":   0.0, "el": -37.2},
        "ext_codo":      {"sh":   0.0, "el":  49.1},
        "flex_hombro":   {"sh": -33.5, "el":   0.0},
        "ext_hombro":    {"sh":  33.0, "el":   0.0},
        "levantamiento": {"sh": -11.2, "el": -58.2}}
CARGA_SESION = {"20260806_182858": 0.00, "20260806_193017": 0.00,
                "20260807_150646": 0.00, "20260807_170550": 0.85,
                "20260811_122212": 0.50}


# capa de atencion del modelo del paso 4
def definir_atencion():
    import keras, tensorflow as tf
    from keras import layers

    @keras.saving.register_keras_serializable()
    class AtencionAditiva(layers.Layer):
        def __init__(self, unidades=32, **kw):
            super().__init__(**kw)
            self.unidades = unidades
            self.W = layers.Dense(unidades, use_bias=True)
            self.v = layers.Dense(1, use_bias=False)

        def call(self, h):
            a = tf.nn.softmax(self.v(tf.tanh(self.W(h))), axis=1)
            return tf.reduce_sum(a * h, axis=1)

        def get_config(self):
            c = super().get_config()
            c["unidades"] = self.unidades
            return c

    return AtencionAditiva


def reconstruir_modelo():
    """Arquitectura del modelo del paso 4, variante con cabeza auxiliar."""
    import keras
    from keras import layers
    A = definir_atencion()
    e = layers.Input((20, 15))
    x = layers.Conv1D(32, 3, padding="same", name="conv1d")(e)
    x = layers.BatchNormalization(name="bn0")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, padding="same", name="conv1d_1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(48, return_sequences=True, name="lstm")(x)
    x = A(32, name="aten")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu", name="d32")(x)
    c = layers.Dense(6, activation="softmax", name="clase")(x)
    i = layers.Dense(1, activation="sigmoid", name="intencion")(x)
    return keras.Model(e, [c, i])


def cargar_pesos_directo(ruta):
    """Respaldo para cuando la version de Keras no coincide con la que guardo
    el modelo: reconstruye la arquitectura y asigna los pesos del archivo."""
    import zipfile, tempfile, h5py
    m = reconstruir_modelo()
    m(np.zeros((1, 20, 15), np.float32))
    tmp = tempfile.mkdtemp()
    zipfile.ZipFile(ruta).extract("model.weights.h5", tmp)
    f = h5py.File(os.path.join(tmp, "model.weights.h5"), "r")
    g = lambda p: np.array(f[p])
    m.get_layer("conv1d").set_weights([g("layers/conv1d/vars/0"),
                                       g("layers/conv1d/vars/1")])
    m.get_layer("conv1d_1").set_weights([g("layers/conv1d_1/vars/0"),
                                         g("layers/conv1d_1/vars/1")])
    for n, h in (("bn0", "batch_normalization"), ("bn1", "batch_normalization_1")):
        m.get_layer(n).set_weights([g("layers/%s/vars/%d" % (h, k)) for k in range(4)])
    m.get_layer("lstm").set_weights([g("layers/lstm/cell/vars/%d" % k) for k in range(3)])
    at = m.get_layer("aten")
    at.W.set_weights([g("layers/atencion_aditiva/W/vars/0"),
                      g("layers/atencion_aditiva/W/vars/1")])
    at.v.set_weights([g("layers/atencion_aditiva/v/vars/0")])
    m.get_layer("d32").set_weights([g("layers/dense/vars/0"), g("layers/dense/vars/1")])
    m.get_layer("clase").set_weights([g("layers/dense_1/vars/0"), g("layers/dense_1/vars/1")])
    m.get_layer("intencion").set_weights([g("layers/dense_2/vars/0"), g("layers/dense_2/vars/1")])
    f.close()
    return m


def cargar_modelo(ruta):
    """Carga directa; si la version de Keras no lo admite, reconstruye."""
    import keras
    A = definir_atencion()
    try:
        return keras.models.load_model(ruta, compile=False,
                                       custom_objects={"AtencionAditiva": A})
    except Exception as exc:
        print("   carga directa no disponible ({}), se reconstruye desde los pesos"
              .format(type(exc).__name__))
        return cargar_pesos_directo(ruta)


# estimacion fusionada del angulo de codo
def filtro_complementario(servo, omega, q0, tau=TAU, fs=FS):
    """Fusiona el angulo del servo, que no deriva, con la integracion de la
    velocidad angular de la IMU, que sigue el movimiento real. La frecuencia
    de cruce 1/(2*pi*tau) queda por debajo de la banda del movimiento."""
    dt = 1.0 / fs
    a = tau / (tau + dt)
    sv = pd.Series(servo).ffill().bfill().values
    om = np.nan_to_num(omega)
    q = np.empty(len(sv))
    q[0] = q0
    for i in range(1, len(q)):
        q[i] = a * (q[i - 1] + om[i] * dt) + (1 - a) * sv[i]
    return q


def estado_imu(e):
    """Diagnostico del estado de la IMU, con el mismo criterio del paso 1:
    duracion del tramo plano mas largo de la coordenada articular y
    correlacion del angulo del servo con el angulo absoluto de la IMU.
    En las sesiones sanas el tramo plano no supera 0.17 s, que corresponde
    a la tasa de refresco de unos 18.5 Hz del flujo de la IMU."""
    q = e["ElbowJointAngle_deg"].values
    d = np.diff(q) == 0
    mx = c = 0
    for v in d:
        c = c + 1 if v else 0
        if c > mx:
            mx = c
    plano = mx / FS
    a = e["ElbowAngle_deg"].values
    b = e["ElbowIMUAngle_deg"].values
    m = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 else float("nan")
    return plano, r


def calidad_estimacion(e):
    """Determina que senal se emplea como realimentacion de posicion del codo
    en el lazo de control, y devuelve esa senal.

    La decision del paso 1 fue conservar la sesion de IMU congelada en el
    entrenamiento del modelo, porque el sistema debe seguir clasificando
    cuando un sensor falla. Esa razon no se traslada al lazo de control: el
    criterio de llegada y la guarda de rango se evaluan sobre la posicion
    articular, y una senal congelada los vuelve inoperantes. En esas
    sesiones se emplea por tanto el angulo del servo como realimentacion,
    conforme a la nota de registro de la sesion.

    En las sesiones con IMU sana se emplea la estimacion fusionada, que
    corrige la deriva de la integracion sin perder el detalle del
    movimiento."""
    plano, r = estado_imu(e)
    a = e["ElbowAngle_deg"].values
    b = e["ElbowJointAngle_deg"].values
    sana = bool(plano <= PLANO_MAX and (not np.isfinite(r) or r >= R_MIN_IMU))
    off = float(np.nanmean(b - a)) if sana else 0.0
    if sana:
        q = filtro_complementario(a + off, e["ElbowIMUOmega_dps"].values,
                                  float(np.nanmean(b[:500])))
        sigma = float(np.nanstd(q - (a + off)))
        fuente = "fusion servo-IMU"
    else:
        q = pd.Series(a).ffill().bfill().values
        sigma = 0.0
        fuente = "angulo del servo (IMU degradada)"
    return {"plano_max_s": float(plano), "r_servo_imu": r, "offset": off,
            "sigma": sigma, "imu_sana": sana, "fuente_q_el": fuente,
            "rango_el": [SERVO[0] + off, SERVO[1] + off]}, q


def quintico(A, T, t):
    s = np.clip(t / T, 0.0, 1.0)
    q = A * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    v = A / T * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4)
    a = A / T ** 2 * (60 * s - 180 * s ** 2 + 120 * s ** 3)
    return q, v, a


def sat(x):
    return float(np.clip(x, 0.0, 1.0))


def simular(e, q_el, t_k, p_int, clase_k, rom, calidad):
    """Maquina de estados, trayectoria y ley AAN a lo largo de la sesion."""
    q = {"sh": e["ShoulderPos_deg"].values, "el": q_el}
    v = {"sh": np.abs(e["ShoulderSpd_dps"].values),
         "el": np.abs(e["ElbowIMUOmega_dps"].values)}
    F = {"sh": e["LoadS_g"].values, "el": e["LoadE_g"].values}
    lab = e["label"].fillna("NULL").values
    idx = np.clip((t_k * FS).astype(int), 0, len(q["sh"]) - 1)

    estado, c_on, c_off = "REPOSO", 0, 0
    q0, clase, t_s, k0 = {}, None, None, 0
    act = []
    traza = {"alpha_sh": np.zeros(len(t_k)), "alpha_el": np.zeros(len(t_k)),
             "qd_sh": np.full(len(t_k), np.nan),
             "qd_el": np.full(len(t_k), np.nan),
             "estado": np.zeros(len(t_k), dtype=np.int8)}
    n_vel = 0

    for k in range(len(t_k)):
        i = idx[k]
        if estado == "REPOSO":
            c_on = c_on + 1 if p_int[k] >= THETA_ON else 0
            if c_on >= N_ON:
                estado, c_on, c_off, t_s, k0 = "MOVIMIENTO", 0, 0, t_k[k], k
                q0 = {j: q[j][i] for j in ("sh", "el")}
                clase = clase_k[k]
                esc = 1.0
                if ESCALAR_ALCANCE:
                    for j in ("sh", "el"):
                        d0 = DQ_F[clase][j]
                        if abs(d0) < 1e-9:
                            continue
                        lo, hi = rom[j]
                        margen = (hi - q0[j]) if d0 > 0 else (q0[j] - lo)
                        esc = 0.0 if margen <= 0 else min(
                            esc, float(np.clip(margen / abs(d0), 0.0, 1.0)))
                act.append({"t_s": float(t_s), "clase": str(clase),
                            "escala": float(esc), "recorte": False,
                            "origen": "gesto" if lab[i] in GESTOS else "reposo"})
        else:
            traza["estado"][k] = 1
            tau_t = t_k[k] - t_s
            dq = {j: DQ_F[clase][j] * act[-1]["escala"] for j in ("sh", "el")}
            for j in ("sh", "el"):
                qq, vv, _ = quintico(dq[j], T_MOV, tau_t)
                obj = q0[j] + qq
                lo, hi = rom[j]
                if obj < lo or obj > hi:
                    obj = min(max(obj, lo), hi)
                    act[-1]["recorte"] = True
                if abs(vv) > CAP_V[j]:
                    n_vel += 1
                traza["qd_%s" % j][k] = obj
                err = abs(obj - q[j][i])
                eta = W_E * min(err / 20.0, 1.0) + W_F * sat(
                    (abs(F[j][i]) - EPS_F) / (3000.0 - EPS_F))
                traza["alpha_%s" % j][k] = p_int[k] * sat(
                    (eta - ETA_0) / (ETA_1 - ETA_0))
            # llegada: contra el OBJETIVO y solo tras completar la trayectoria
            if tau_t >= T_MOV:
                llegada = all(abs(q[j][i] - (q0[j] + dq[j])) <= EPS_Q[j]
                              and v[j][i] <= EPS_V[j] for j in ("sh", "el")) \
                          and abs(F["el"][i]) <= EPS_F
            else:
                llegada = False
            c_off = c_off + 1 if (llegada or p_int[k] <= THETA_OFF) else 0
            if c_off >= N_OFF or tau_t > 2.0 * T_MOV:
                a = act[-1]
                a["t_f"] = float(t_k[k])
                a["duracion"] = float(t_k[k] - t_s)
                a["motivo"] = ("llegada" if llegada else
                               ("perdida" if p_int[k] <= THETA_OFF
                                else "tiempo maximo"))
                # oposicion: la referencia se movio en contra del usuario
                if k - k0 > 8:
                    op = tot = 0
                    for j in ("sh", "el"):
                        dr = traza["qd_%s" % j][k] - q0[j]
                        du = q[j][i] - q[j][idx[k0]]
                        if not np.isfinite(dr) or not np.isfinite(du):
                            continue
                        if abs(dr) < 2 or abs(du) < 2:
                            continue
                        tot += 1
                        op += int(np.sign(dr) != np.sign(du))
                    a["seg_dir"], a["seg_opuestos"] = tot, op
                estado, c_off = "REPOSO", 0
    for a in act:
        a.setdefault("t_f", float(t_k[-1]))
        a.setdefault("duracion", 0.0)
        a.setdefault("motivo", "fin de sesion")
        a.setdefault("seg_dir", 0)
        a.setdefault("seg_opuestos", 0)
    return act, traza, n_vel


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.dirname(aqui)
    datos = os.path.join(raiz, "data")
    if not os.path.isdir(datos):
        datos = aqui
    salida = os.path.join(raiz, "resultados")
    if not os.path.isdir(salida):
        salida = aqui

    def buscar(patron):
        for d in (salida, aqui, raiz):
            h = glob.glob(os.path.join(d, patron))
            if h:
                return sorted(h)[0]
        return None

    rutas = sorted(glob.glob(os.path.join(datos, "**", "emg_*.csv"),
                             recursive=True))
    if not rutas:
        raise FileNotFoundError("No hay emg_*.csv en " + datos)

    print("PASO 6  CADENA SUPERVISORA EN LAZO ABIERTO")
    print("theta_on {:.2f}  theta_off {:.2f}  N_on {}  N_off {}  T_mov {:.1f} s"
          .format(THETA_ON, THETA_OFF, N_ON, N_OFF, T_MOV))
    print("P_int: cabeza auxiliar del modelo  |  clase: cabeza de clase")

    # 1. calidad de la estimacion articular, antes de todo resultado
    print("\n1. ESTADO DE LA IMU Y FUENTE DE REALIMENTACION DEL CODO\n")
    print("   {:<18}{:>11}{:>8}{:>9}   {}".format(
        "sesion", "plano max", "r", "offset", "fuente de q_el"))
    print("   " + "-" * 76)
    CAL, QEL, SES = {}, {}, {}
    for r in rutas:
        k = os.path.basename(r)[4:-4]
        e = pd.read_csv(r, low_memory=False)
        c, q = calidad_estimacion(e)
        CAL[k], QEL[k], SES[k] = c, q, e
        print("   {:<18}{:>10.2f}s{:>8.3f}{:>8.1f}°   {}".format(
            k, c["plano_max_s"], c["r_servo_imu"], c["offset"], c["fuente_q_el"]))
    sanas = [k for k in CAL if CAL[k]["imu_sana"]]
    print("   " + "-" * 76)
    print("   IMU sana en {} de {} sesiones".format(len(sanas), len(CAL)))

    # 2. inferencia con el modelo del paso 4
    print("\n2. INFERENCIA CON EL MODELO ENTRENADO\n")
    modelos = {}
    for c in ("0.00kg", "0.50kg", "0.85kg"):
        f = buscar("modelo_{}_auxint.keras".format(c))
        if f is None:
            raise FileNotFoundError("Falta modelo_{}_auxint.keras".format(c))
        modelos[c] = cargar_modelo(f)
        print("   cargado {}".format(os.path.basename(f)))

    RES, GLOB = {}, {"act": 0, "gesto": 0, "eps": 0, "cub": 0, "rec": 0,
                     "vel": 0, "dir": 0, "opo": 0}
    print("\n3. SIMULACION POR SESION\n")
    for k in CAL:
        carga = "{:.2f}kg".format(CARGA_SESION.get(k, 0.0))
        pl = buscar("pliegue_{}.npz".format(carga))
        if pl is None:
            print("   {} sin pliegue correspondiente, se omite".format(k))
            continue
        z = np.load(pl, allow_pickle=True)
        ses_test = [str(s) for s in np.atleast_1d(z["sesiones_test"])]
        if k not in " ".join(ses_test):
            print("   {} no pertenece al conjunto de prueba de {}, se omite"
                  .format(k, carga))
            continue
        # Solo las secuencias de esta sesion. X_test puede contener varias.
        sel = np.asarray([str(x) for x in z["sesion_test"]]) == k
        if not sel.any():
            print("   {} no aparece en sesion_test de {}, se omite".format(k, carga))
            continue
        X = z["X_test"][sel].astype("float32")
        pc, pi = modelos[carga].predict(X, batch_size=2048, verbose=0)
        p_int = pi.ravel()
        cls = np.array(CLASES)[1 + pc[:, 1:].argmax(1)]

        e = SES[k]
        n = len(e)
        lab_s = e["label"].fillna("NULL").values
        cortes = np.flatnonzero(np.r_[True, lab_s[1:] != lab_s[:-1]])
        cortes = np.r_[cortes, len(lab_s)]
        b = z["blk_test"][sel].astype(np.int64)
        b = b - (b.min() // 100000) * 100000
        b = np.clip(b, 0, len(cortes) - 1)
        t_v = cortes[b] / FS + z["t_since_label_test"][sel]
        orden = np.argsort(t_v)
        t_v, p_int, cls = t_v[orden], p_int[orden], cls[orden]
        t_k = np.arange(t_v[0], t_v[-1], T_H)
        pk = np.interp(t_k, t_v, p_int)
        ck = cls[np.searchsorted(t_v, t_k).clip(0, len(cls) - 1)]
        rom = {"sh": ROM_SH, "el": tuple(CAL[k]["rango_el"])}
        act, traza, nvel = simular(e, QEL[k], t_k, pk, ck, rom, CAL[k])

        lab = e["label"].fillna("NULL").values
        cor = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        eps = [(c / FS, (cor[j + 1] if j + 1 < len(cor) else n) / FS)
               for j, c in enumerate(cor) if lab[c] in GESTOS]
        g = sum(1 for a in act if a["origen"] == "gesto")
        cub = sum(1 for a0, a1 in eps
                  if any(a0 <= a["t_s"] <= a1 for a in act))
        rec = sum(1 for a in act if a["recorte"])
        sd = sum(a["seg_dir"] for a in act)
        so = sum(a["seg_opuestos"] for a in act)
        RES[k] = {"imu_sana": CAL[k]["imu_sana"], "carga": carga,
                  "activaciones": len(act), "en_gesto": g,
                  "episodios": len(eps), "cubiertos": cub, "recortes": rec,
                  "exc_velocidad": nvel, "seg_dir": sd, "seg_opuestos": so,
                  "duracion_mediana": float(np.median([a["duracion"] for a in act])) if act else 0.0,
                  "escala_media": float(np.mean([a["escala"] for a in act])) if act else 1.0,
                  "frac_asistiendo": float((traza["estado"] == 1).mean()),
                  "alpha_medio": float(np.mean(traza["alpha_el"][traza["estado"] == 1]))
                                 if (traza["estado"] == 1).any() else 0.0,
                  "motivos": {m: sum(1 for a in act if a["motivo"] == m)
                              for m in set(a["motivo"] for a in act)}}
        for a, b in (("act", "activaciones"), ("gesto", "en_gesto"),
                     ("eps", "episodios"), ("cub", "cubiertos"),
                     ("rec", "recortes"), ("vel", "exc_velocidad"),
                     ("dir", "seg_dir"), ("opo", "seg_opuestos")):
            GLOB[a] += RES[k][b]
        np.savez(os.path.join(salida, "cadena_{}.npz".format(k)),
                 t=t_k, p_int=pk, **traza)
        print("   {}  act {:>3}  en gesto {:>3}  cubiertos {:>2}/{:<2}  "
              "recortes {:>3}  exc.vel {}  oposicion {}/{}"
              .format(k, len(act), g, cub, len(eps), rec, nvel, so, sd))

    print("\n" + "=" * 76)
    if GLOB["act"]:
        print("GLOBAL  activaciones {}   precision {:.1%}   cobertura {}/{} = {:.1%}"
              .format(GLOB["act"], GLOB["gesto"] / GLOB["act"],
                      GLOB["cub"], GLOB["eps"], GLOB["cub"] / max(GLOB["eps"], 1)))
        print("        recortes {} ({:.1%})   excesos de velocidad {}"
              .format(GLOB["rec"], GLOB["rec"] / GLOB["act"], GLOB["vel"]))
        if GLOB["dir"]:
            print("        oposicion direccional {}/{} = {:.1%}"
                  .format(GLOB["opo"], GLOB["dir"], GLOB["opo"] / GLOB["dir"]))
    out = {"parametros": {"T_H": T_H, "theta_on": THETA_ON,
                          "theta_off": THETA_OFF, "N_on": N_ON, "N_off": N_OFF,
                          "T_mov": T_MOV, "tau": TAU, "eps_q": EPS_Q,
                          "eps_v": EPS_V, "eps_F": EPS_F, "Dq_f": DQ_F,
                          "escalado_alcance": ESCALAR_ALCANCE,
                          "w_e": W_E, "w_F": W_F, "eta_0": ETA_0, "eta_1": ETA_1},
           "calidad": CAL, "por_sesion": RES, "global": GLOB}
    with open(os.path.join(salida, "resumen_cadena.json"), "w",
              encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nEscrito resumen_cadena.json y cadena_<sesion>.npz en {}".format(salida))


if __name__ == "__main__":
    main()
