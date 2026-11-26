"""Auditoría A3. Frecuencia de muestreo del conjunto público, deducida de sus propios datos.
Uso: python -I a3_frecuencia_muestreo.py DIR_CONJUNTO SALIDA.json"""
import glob, json, os, sys
import numpy as np
from scipy import signal as sg
from scipy.optimize import minimize, minimize_scalar
D, SAL = sys.argv[1:3]
arch = sorted(f for f in glob.glob(os.path.join(D, "subject *", "*.txt")) if "info" not in f)
res = {}
Pxy = Pxx = Pyy = 0
for f in arch:
    A = np.loadtxt(f)
    for c, d in ((0, 2), (1, 3)):
        fr, pxy = sg.csd(A[:, c], A[:, d], fs=1.0, nperseg=16384); _, pxx = sg.welch(A[:, c], fs=1.0, nperseg=16384); _, pyy = sg.welch(A[:, d], fs=1.0, nperseg=16384)
        Pxy, Pxx, Pyy = Pxy + pxy, Pxx + pxx, Pyy + pyy
H = Pxy / Pxx; coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
s = (fr > 0.003) & (fr < 0.4999) & (coh > 0.95); f, Hm = fr[s], H[s]
def L(v):
    w1, w2, rp = v
    if not (0.003 < w1 < 0.03 and 0.45 < w2 < 0.49999 and 0.001 < rp < 6): return 1e3
    b, a = sg.cheby1(4, rp, [w1, w2], btype="bandpass", fs=1.0); _, h = sg.freqz(b, a, worN=f, fs=1.0)
    return float(np.mean(np.abs(h - Hm) ** 2)) * 1e4
r = min((minimize(L, x0, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-12, maxiter=6000)) for x0 in
         ([0.01, 0.49, 0.05], [0.012, 0.498, 0.5], [0.008, 0.47, 0.05])), key=lambda r: r.fun)
w1, w2, rp = r.x
res["transferencia"] = dict(borde_inferior_ciclos_muestra=w1, borde_superior_ciclos_muestra=w2, rizado_dB=rp, ecm=r.fun / 1e4,
                            coherencia_min_banda=float(coh[s].min()), fase_grados_0_02=float(np.degrees(np.angle(H[np.argmin(abs(fr - 0.02))]))),
                            fs_si_inferior_10Hz=10 / w1, fs_si_superior_500Hz=500 / w2, fs_si_superior_nyquist_menos_1Hz=1 / (0.5 - w2))
print(res["transferencia"])
def err(A, c, d, fs, rp):
    b, a = sg.cheby1(4, rp, [10, min(500, fs / 2 - 1)], btype="bandpass", fs=fs); y = sg.lfilter(b, a, A[:, c])
    sl = slice(4000, None); e = A[sl, d] - y[sl]; return float(np.sqrt((e ** 2).mean()) / A[sl, d].std())
def mejor_rp(A, c, d, fs):
    o = minimize_scalar(lambda rp: err(A, c, d, fs, rp), bounds=(0.001, 3), method="bounded", options=dict(xatol=1e-5)); return o.fun, o.x
cands = [744.7, 762.0, 780.2, 799.2, 819.2, 1000.0, 1024.0]
tabla = {str(fs): [] for fs in cands}; fs_opt = []
for f in arch:
    A = np.loadtxt(f)
    for fs in cands: tabla[str(fs)].append(mejor_rp(A, 0, 2, fs)[0])
    o = minimize(lambda v: err(A, 0, 2, v[0], v[1]), [790, 0.04], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-10)); fs_opt.append(o.x[0])
res["reconstruccion"] = {fs: dict(mediana=float(np.median(v)), maximo=float(np.max(v))) for fs, v in tabla.items()}
res["fs_optima_por_archivo"] = dict(mediana=float(np.median(fs_opt)), p5=float(np.percentile(fs_opt, 5)), p95=float(np.percentile(fs_opt, 95)), n=len(fs_opt))
res["nota"] = "Shimmer3 genera fs = 32768/N Hz: 744,7 (N=44), 762,0 (43), 780,2 (42), 799,2 (41), 819,2 (40), 1024 (32)."
print(res["reconstruccion"]); print(res["fs_optima_por_archivo"])
json.dump(res, open(SAL, "w"), indent=1)
