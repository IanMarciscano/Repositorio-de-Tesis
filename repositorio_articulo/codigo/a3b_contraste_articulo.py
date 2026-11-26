"""Auditoría A3b. Contraste de A3 con el artículo asociado al conjunto (Toro-Ossaba et al., 2024), que declara
Uso: python -I a3b_contraste_articulo.py DIR_CONJUNTO SALIDA.json"""
import glob, json, os, sys
import numpy as np
from scipy import signal as sg
D, SAL = sys.argv[1:3]
arch = sorted(f for f in glob.glob(os.path.join(D, "subject *", "*_flex_*.txt")))
Pxy = Pxx = 0; per, npk, lon = [], [], []
for f in arch:
    A = np.loadtxt(f)
    for c, d in ((0, 2), (1, 3)):
        fr, pxy = sg.csd(A[:, c], A[:, d], fs=1.0, nperseg=16384); _, pxx = sg.welch(A[:, c], fs=1.0, nperseg=16384)
        Pxy, Pxx = Pxy + pxy, Pxx + pxx
    ang = sg.savgol_filter(A[:, 4], 201, 2); pk, _ = sg.find_peaks(ang, prominence=30, distance=400)
    npk.append(len(pk)); lon.append(len(A)); per += list(np.diff(pk))
m = np.abs(Pxy / Pxx); i = np.flatnonzero((fr > 0.005) & (fr < 0.015)); f3 = float(np.interp(1 / np.sqrt(2), m[i], fr[i]))
per = np.array(per)
res = dict(punto_menos3dB_ciclos_muestra=f3, fs_si_10Hz_es_menos3dB=10 / f3, ganancia_a_500Hz_si_fs1024=float(m[np.argmin(abs(fr - 500 / 1024))]),
           archivos=len(arch), picos_por_archivo_mediana=float(np.median(npk)), periodo_muestras_mediana=float(np.median(per)),
           periodo_muestras_p25_p75=[float(np.percentile(per, 25)), float(np.percentile(per, 75))],
           segundos_por_ciclo={"1024": float(np.median(per) / 1024), "780.2": float(np.median(per) / 780.2)},
           duracion_archivo_mediana_s={"1024": float(np.median(lon) / 1024), "780.2": float(np.median(lon) / 780.2)})
print(res); json.dump(res, open(SAL, "w"), indent=1)
