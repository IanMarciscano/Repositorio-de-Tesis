"""Pruebas de validez sobre la linea base B: conglomerados y series subrogadas.
"""
# -*- coding: utf-8 -*-
import importlib.util, json, sys
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

spec = importlib.util.spec_from_file_location("a4", "/mnt/user-data/outputs/aud04_ic_anticipacion.py")
a4 = importlib.util.module_from_spec(spec)
sys.argv = ["aud06"]
a4.__name__ = "a4"
spec.loader.exec_module(a4)          # define trayectorias, conglomerados, BASE, REJILLA
p5 = a4.p5
REJILLA, BASE = a4.REJILLA, a4.BASE
print("linea base empleada:", BASE, "ms")

CACHE = "/home/claude/work/P_trayectorias.npy"

def matriz_probabilidades():
    import os
    if os.path.exists(CACHE):
        print("matriz de probabilidades leida de cache")
        return np.load(CACHE)
    datos = a4.trayectorias()
    X, y, g, info = p5.construir()
    prob_modelo = {}
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = RandomForestClassifier(300, random_state=p5.SEMILLA, n_jobs=-1)
        m.fit(X[tr], y[tr])
        for e in np.unique(g[te]):
            prob_modelo[e] = m
    P = [prob_modelo[ep].predict_proba(F)[:, 1] for ep, F in datos if ep in prob_modelo]
    P = np.array(P)
    np.save(CACHE, P)
    return P

def masas_de(t, umbral):
    fuera = np.abs(t) > umbral
    salida, i = [], 0
    while i < len(fuera):
        if fuera[i]:
            j = i
            while j + 1 < len(fuera) and fuera[j + 1]:
                j += 1
            salida.append((i, j, float(np.abs(t[i:j + 1]).sum())))
            i = j + 1
        else:
            i += 1
    return salida

def estadistico(M):
    ne = M.shape[0]
    return M.mean(0) / np.maximum(M.std(0, ddof=1) / np.sqrt(ne), 1e-12)

P = matriz_probabilidades()
ne = P.shape[0]
print("trayectorias fuera de muestra: %d x %d" % P.shape)
base = (REJILLA >= BASE[0]) & (REJILLA < BASE[1])
D = P - P[:, base].mean(1, keepdims=True)
umbral = stats.t.ppf(0.975, ne - 1)

obs = sorted(masas_de(estadistico(D), umbral), key=lambda x: -x[2])
i0, i1, masa = obs[0]
divergencia = int(REJILLA[i0])
print("conglomerado mayor: %d a %d ms, masa %.1f" % (REJILLA[i0], REJILLA[i1], masa))

# 1. permutacion de signo
rng = np.random.default_rng(20260917)
nulo_perm = np.zeros(2000)
for r in range(2000):
    m = masas_de(estadistico(D * rng.choice([-1.0, 1.0], size=(ne, 1))), umbral)
    nulo_perm[r] = max([x[2] for x in m]) if m else 0.0
p_perm = (1.0 + (nulo_perm >= masa).sum()) / (1.0 + len(nulo_perm))
print("permutacion de signo: p95 %.1f  max %.1f  p = %.4f"
      % (np.percentile(nulo_perm, 95), nulo_perm.max(), p_perm))

# 2. sustitutos por desplazamiento circular
rng = np.random.default_rng(20260923)
nT = P.shape[1]
nulo_sus = np.zeros(1000)
for r in range(1000):
    S = np.array([np.roll(P[e], rng.integers(1, nT)) for e in range(ne)])
    Ds = S - S[:, base].mean(1, keepdims=True)
    m = masas_de(estadistico(Ds), umbral)
    nulo_sus[r] = max([x[2] for x in m]) if m else 0.0
p_sus = (1.0 + (nulo_sus >= masa).sum()) / (1.0 + len(nulo_sus))
print("sustitutos circulares: mediana %.1f  p95 %.1f  max %.1f  p = %.4f"
      % (np.median(nulo_sus), np.percentile(nulo_sus, 95), nulo_sus.max(), p_sus))

# 3. remuestreo de episodios
rng = np.random.default_rng(7)
inicios = []
for _ in range(2000):
    M = D[rng.integers(0, ne, ne)]
    t = estadistico(M)
    u = stats.t.ppf(0.975, M.shape[0] - 1)
    m = masas_de(t, u)
    if not m:
        continue
    j0 = max(m, key=lambda x: x[2])[0]
    inicios.append(REJILLA[j0])
inicios = np.array(inicios)
ic = [int(np.percentile(inicios, 2.5)), int(np.percentile(inicios, 97.5))]
frac = float((inicios < 0).mean())
print("remuestreo: IC 95 %% [%d, %d] ms, fraccion que anticipa %.3f" % (ic[0], ic[1], frac))

json.dump(dict(linea_base="B [-1000,-500] ms", episodios=int(ne),
               divergencia_ms=divergencia, masa=round(float(masa), 1),
               perm_p95=round(float(np.percentile(nulo_perm, 95)), 1),
               perm_max=round(float(nulo_perm.max()), 1), perm_p=round(float(p_perm), 4),
               sus_mediana=round(float(np.median(nulo_sus)), 1),
               sus_p95=round(float(np.percentile(nulo_sus, 95)), 1),
               sus_max=round(float(nulo_sus.max()), 1), sus_p=round(float(p_sus), 4),
               ic=ic, fraccion_anticipa=round(frac, 3), replicas_sus=1000),
          open("/home/claude/work/validez_lineaB.json", "w"), indent=1, ensure_ascii=False)
print("\nlisto -> validez_lineaB.json")
