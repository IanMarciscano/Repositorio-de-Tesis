"""Intervalo de confianza del area bajo la curva de la puerta, por remuestreo de episodios.
"""
# -*- coding: utf-8 -*-
import importlib.util, json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("p5", "/home/claude/work/05_puerta_intencion.py")
p5 = importlib.util.module_from_spec(spec)
sys.argv = ["05"]
spec.loader.exec_module(p5)
p5.DIR_DATOS = "/mnt/user-data/uploads/outputs/entrenamiento de la red/data"

X, y, g, info = p5.construir()
print("ventanas %d, episodios %d, caracteristicas %d"
      % (len(y), len(np.unique(g)), X.shape[1]))

prob = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    m = RandomForestClassifier(300, random_state=p5.SEMILLA, n_jobs=-1)
    m.fit(X[tr], y[tr])
    prob[te] = m.predict_proba(X[te])[:, 1]
auc = roc_auc_score(y, prob)
print("AUC agregada reproducida: %.4f  (la tesis reporta 0,6344)" % auc)

eps = np.unique(g)
rng = np.random.default_rng(20260917)
reps = []
for _ in range(2000):
    sel = rng.choice(eps, size=len(eps), replace=True)
    idx = np.concatenate([np.where(g == e)[0] for e in sel])
    if len(np.unique(y[idx])) < 2:
        continue
    reps.append(roc_auc_score(y[idx], prob[idx]))
reps = np.array(reps)
lo, hi = np.percentile(reps, [2.5, 97.5])
print("IC 95 %% por remuestreo de episodios: [%.3f, %.3f]  (n=%d réplicas)"
      % (lo, hi, len(reps)))
print("fracción de réplicas por debajo de 0,5: %.4f" % (reps < 0.5).mean())
json.dump(dict(auc=round(float(auc),4), ic=[round(float(lo),3), round(float(hi),3)],
               p_bootstrap=round(float((reps<0.5).mean()),4),
               n_episodios=int(len(eps)), n_ventanas=int(len(y)), replicas=len(reps)),
          open("/home/claude/work/ic_puerta.json","w"), ensure_ascii=False, indent=1)
