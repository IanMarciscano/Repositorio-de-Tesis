"""Construye el conjunto de la puerta de intención con la misma regla que"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import signal as sg

FS = 1000.0; WIN_MS = 300; HOP_MS = 10; ANCHO_PRE = 300; SEP_REPOSO = 1000
UMBRAL_VEL = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
PERSIST_MS = 50
BP = (20.0, 450.0); NOTCH = (60.0, 30.0); ORDEN = 4
GESTOS = ["flex_codo", "ext_codo", "flex_hombro", "ext_hombro", "levantamiento"]
T_PRE, T_POST = 2000, 600

def filtrar(x):
    nyq = FS / 2
    b1, a1 = sg.butter(ORDEN, [BP[0] / nyq, BP[1] / nyq], btype="bandpass")
    b2, a2 = sg.iirnotch(NOTCH[0] / nyq, NOTCH[1])
    return sg.lfilter(b2, a2, sg.lfilter(b1, a1, x.astype(np.float64)))

def ventanas(x, n, h):
    x = np.ascontiguousarray(x); nv = (len(x) - n) // h + 1; s = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (nv, n), (h * s, s), writeable=False)

def caracteristicas(W):
    R = np.abs(W); fr = np.fft.rfftfreq(W.shape[1], 1 / FS)
    P = np.abs(np.fft.rfft(W, axis=1)) ** 2
    return np.c_[R.sum(1), (R * R).sum(1), np.abs(np.diff(R, axis=1)).sum(1),
                 np.exp(np.log(R + 1.0).mean(1)),
                 (np.diff(np.sign(W), axis=1) != 0).sum(1),
                 (np.diff(np.sign(np.diff(W, axis=1)), axis=1) != 0).sum(1),
                 (P * fr).sum(1) / np.maximum(P.sum(1), 1e-12)]

def primer_cruce(x, umbral, persist):
    s = (x > umbral).astype(int)
    if s.sum() < persist: return None
    c = np.convolve(s, np.ones(persist), "valid"); i = np.flatnonzero(c >= persist)
    return int(i[0]) if len(i) else None

def main(dir_datos, destino):
    rutas = sorted(glob.glob(os.path.join(dir_datos, "emg_*.csv")))
    n = int(FS * WIN_MS / 1000); h = int(FS * HOP_MS / 1000)
    X, y, g, ses, clase = [], [], [], [], []
    TR, tr_ep, tr_ses, tr_clase, tr_tetq = [], [], [], [], []
    stats = {}
    for si, r in enumerate(rutas):
        e = pd.read_csv(r, low_memory=False, usecols=["label", "EMG_ch2_raw_bic", "EMG_ch3_raw_tri", "ElbowIMUOmega_dps", "ShoulderSpd_dps"])
        lab = e["label"].fillna("NULL").values
        F = np.c_[caracteristicas(ventanas(filtrar(e["EMG_ch2_raw_bic"].values), n, h)),
                  caracteristicas(ventanas(filtrar(e["EMG_ch3_raw_tri"].values), n, h))]
        F = np.sign(F) * np.log1p(np.abs(F))
        fin = np.arange(len(F)) * h + n - 1
        v_el = np.abs(e["ElbowIMUOmega_dps"].values); v_sh = np.abs(e["ShoulderSpd_dps"].values)
        cortes = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        stats[si] = (F[100:].mean(0), F[100:].std(0))
        n_ep = 0
        for k, i0 in enumerate(cortes):
            if lab[i0] not in GESTOS or k == 0: continue
            i1 = cortes[k + 1] if k + 1 < len(cortes) else len(lab)
            v = v_el[i0:i1] if "codo" in lab[i0] else v_sh[i0:i1]
            tm = primer_cruce(v, UMBRAL_VEL, PERSIST_MS)
            if tm is None or tm < ANCHO_PRE + n: continue
            t0 = i0 + tm
            m = (fin >= t0 - ANCHO_PRE) & (fin < t0)
            j1 = i0 - SEP_REPOSO; j0 = max(cortes[k - 1] + n, j1 - ANCHO_PRE)
            if j1 - j0 < n or m.sum() == 0: continue
            mr = (fin >= j0) & (fin < j1)
            if mr.sum() == 0: continue
            ep = si * 1000 + k
            for mm, lab_bin in ((m, 1), (mr, 0)):
                X.append(F[mm]); y.append(np.full(mm.sum(), lab_bin)); g.append(np.full(mm.sum(), ep))
                ses.append(np.full(mm.sum(), si)); clase.append(np.full(mm.sum(), lab[i0], dtype=object))
            objetivos = t0 + np.arange(-T_PRE, T_POST + 1, HOP_MS)
            idx = np.round((objetivos - (n - 1)) / h).astype(int)
            ok = (idx >= 0) & (idx < len(F))
            T = np.full((len(objetivos), F.shape[1]), np.nan); T[ok] = F[idx[ok]]
            TR.append(T); tr_ep.append(ep); tr_ses.append(si); tr_clase.append(lab[i0]); tr_tetq.append((i0 - t0))
            n_ep += 1
        print(os.path.basename(r), n_ep, "episodios", flush=True)
    np.savez_compressed(destino, X=np.concatenate(X), y=np.concatenate(y).astype(int), g=np.concatenate(g),
                        ses=np.concatenate(ses), clase=np.concatenate(clase).astype(str),
                        TR=np.array(TR), tr_ep=np.array(tr_ep), tr_ses=np.array(tr_ses), tr_clase=np.array(tr_clase).astype(str),
                        tr_tetq=np.array(tr_tetq), t_rel=np.arange(-T_PRE, T_POST + 1, HOP_MS),
                        mu_ses=np.array([stats[i][0] for i in sorted(stats)]), sd_ses=np.array([stats[i][1] for i in sorted(stats)]),
                        sesiones=np.array([os.path.basename(r)[4:-4] for r in rutas]))
main(sys.argv[1], sys.argv[2])
