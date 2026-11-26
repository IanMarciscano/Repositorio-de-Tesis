"""Genera las figuras del informe a partir de los JSON de resultados.
Uso: python figuras.py DIR_DATOS DIR_RESULTADOS DIR_FIGURAS"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

D, R, F = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False})
AZUL, NARANJA, VERDE, GRIS, ROJO = "#2f6fb3", "#d9822b", "#3a8f5a", "#7a7a7a", "#b33a3a"
J = lambda n: json.load(open(os.path.join(R, n)))
ETQ_S = ["S1", "S2", "S3", "S4", "S5"]


def fig_protocolo():
    s = "20260807_150646"; d = cargar_sesion(D, s); d = d[d.etiqueta != "calibracion"]
    t = (d.idx - d.idx.iloc[0]) / 100.0
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(hspace=0.12))
    es_g = d.etiqueta.isin(GESTOS).to_numpy()
    for a in ax:
        ini = None
        for i in range(len(d)):
            if es_g[i] and ini is None: ini = t.iloc[i]
            if ini is not None and (not es_g[i] or i == len(d) - 1):
                a.axvspan(ini, t.iloc[i], color=NARANJA, alpha=0.16, lw=0); ini = None
    ax[0].plot(t, d.q_el, color=AZUL, lw=0.8); ax[0].set_ylabel("Codo (°)")
    ax[1].plot(t, d.q_sh, color=VERDE, lw=0.8); ax[1].set_ylabel("Hombro (°)")
    ax[2].semilogy(t, d.IAV_bic.rolling(50, min_periods=1).median(), color=GRIS, lw=0.8)
    ax[2].set_ylabel("IAV bíceps\n(cuentas)"); ax[2].set_xlabel("Tiempo en la sesión (s)")
    ax[0].set_title("Sesión S3 (0 kg): el movimiento ocurre dentro del bloque de gesto y el reposo conserva la postura alcanzada")
    ax[1].text(0.005, 0.9, "Sombreado: bloque etiquetado como gesto. Sin sombrear: reposo.", transform=ax[1].transAxes, ha="left", fontsize=7.5, color="#8a4f12")
    fig.savefig(os.path.join(F, "fig1_protocolo.png")); plt.close(fig)


def fig_escalera():
    r = J("e1_N1.json")["resultados"]
    conf = [("EMG14|HGB", "sEMG, refuerzo por gradiente", AZUL, "o"), ("EMG14|LDA", "sEMG, discriminante lineal", AZUL, "s"),
            ("MEC7|HGB", "Mecánicas, refuerzo por gradiente", NARANJA, "o"), ("FUS15|HGB", "Fusión, refuerzo por gradiente", VERDE, "o")]
    conf = [c for c in conf if c[0] in r]
    esc = [("A_aleatoria", "Aleatoria\npor ventanas"), ("B_bloque", "Agrupada\npor bloque"),
           ("C_sesion", "Sesión\nexcluida"), ("D_carga", "Carga\nexcluida")]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for k, (clave, nombre, col, mk) in enumerate(conf):
        xs = np.arange(len(esc)) + (k - (len(conf) - 1) / 2) * 0.16
        ys = [np.mean(r[clave][e].get("auc_repeticiones", [r[clave][e]["auc"]])) for e, _ in esc]
        lo = [r[clave][e]["ic95"][0] for e, _ in esc]; hi = [r[clave][e]["ic95"][1] for e, _ in esc]
        ax.errorbar(xs, ys, yerr=[np.maximum(0, np.array(ys) - lo), np.maximum(0, np.array(hi) - ys)], fmt=mk, color=col,
                    mfc=col if mk == "o" else "white", ms=5.5, lw=1, capsize=2, label=nombre)
    ax.axhline(0.5, color="k", lw=0.7, ls=":"); ax.text(len(esc) - 0.55, 0.507, "azar", fontsize=7.5)
    ax.set_xticks(range(len(esc))); ax.set_xticklabels([n for _, n in esc]); ax.set_ylabel("AUC, reposo frente a gesto")
    ax.set_ylim(0.3, 1.02); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("El mismo modelo y las mismas características bajo cuatro particiones")
    fig.savefig(os.path.join(F, "fig2_escalera.png")); plt.close(fig)


def fig_solapamiento():
    r = J("e2_N1.json"); b = r["barrido_paso"]
    pasos = [10, 30, 50, 100, 150, 300]; sol = [100 * (1 - p / 300) for p in pasos]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    for a, lista, titulo in [(ax[0], [("EMG14|HGB", "Refuerzo por gradiente", AZUL), ("EMG14|LDA", "Discriminante lineal", GRIS)], "sEMG"),
                             (ax[1], [("MEC7|HGB", "Mecánicas", NARANJA), ("FUS15|HGB", "Fusión", VERDE)], "Variables mecánicas y fusión")]:
        for clave, nombre, col in lista:
            A = [np.mean([v["A"] for v in b[clave][str(p)].values()]) for p in pasos]
            B = [np.mean([np.mean(v["B"]) for v in b[clave][str(p)].values()]) for p in pasos]
            a.plot(sol, A, "-o", color=col, ms=4, label=f"{nombre}, aleatoria")
            a.plot(sol, B, "--s", color=col, ms=3.5, mfc="white", label=f"{nombre}, por bloque")
        a.axhline(0.5, color="k", lw=0.7, ls=":"); a.set_xlabel("Solapamiento entre ventanas (%)"); a.set_title(titulo)
        a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); a.set_xlim(-4, 100)
    ax[0].set_ylabel("AUC dentro de sesión"); ax[0].set_ylim(0.38, 1.02)
    fig.savefig(os.path.join(F, "fig3_solapamiento.png")); plt.close(fig)


def fig_geometria():
    r = J("e2_N1.json"); e1 = J("e1_N1.json")["resultados"]
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    conf = [("EMG14|HGB", "sEMG", AZUL), ("FUS15|HGB", "Fusión", VERDE)]
    etqs = ["Aleatoria", "Aleatoria,\netiquetas\npermutadas", "Por bloque", "Por bloque,\netiquetas\npermutadas", "Purga ±1", "Purga ±2", "Vecino\ntemporal"]
    for k, (clave, nombre, col) in enumerate(conf):
        if clave not in r["purga_adyacentes"]: continue
        pu, pe = r["purga_adyacentes"][clave], r["permutacion"][clave]
        vals = [np.mean([v["A"] for v in r["barrido_paso"][clave]["10"].values()]),
                np.mean([np.mean(v["A"]) for v in pe.values()]),
                np.mean([np.mean(v["0"]) for v in pu.values()]),
                np.mean([np.mean(v["B"]) for v in pe.values()]),
                np.mean([np.mean(v["1"]) for v in pu.values()]),
                np.mean([np.mean(v["2"]) for v in pu.values()])]
        x = np.arange(6) + (k - 0.5) * 0.36
        ax.bar(x, vals, 0.34, color=col, label=nombre)
        for xi, v in zip(x, vals): ax.text(xi, v + 0.012, f"{v:.2f}".replace(".", ","), ha="center", fontsize=7)
    vt = np.mean([np.mean(v) for v in r["vecino_temporal"].values()])
    ax.bar([6], [vt], 0.34, color=GRIS); ax.text(6, vt + 0.012, f"{vt:.2f}".replace(".", ","), ha="center", fontsize=7)
    ax.axhline(0.5, color="k", lw=0.7, ls=":"); ax.set_xticks(range(7)); ax.set_xticklabels(etqs, fontsize=7.5)
    ax.set_ylabel("AUC dentro de sesión"); ax.set_ylim(0, 1.1); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Dentro de una sesión, la estimación depende de la geometría de los pliegues")
    fig.savefig(os.path.join(F, "fig3b_geometria.png")); plt.close(fig)


def fig_transferencia():
    r = J("e3_N1.json")["matrices"]
    claves = [k for k in ["EMG14|HGB", "MEC7|HGB"] if k in r]
    fig, ax = plt.subplots(1, len(claves), figsize=(3.5 * len(claves), 3.3))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, claves):
        M = np.array(r[k]); im = a.imshow(M, vmin=0.3, vmax=0.7, cmap="RdBu_r")
        for i in range(5):
            for j in range(5):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                       fontweight="bold" if i == j else "normal")
        a.set_xticks(range(5)); a.set_xticklabels(ETQ_S); a.set_yticks(range(5)); a.set_yticklabels(ETQ_S)
        a.set_xlabel("Sesión de prueba"); a.set_title("sEMG" if k.startswith("EMG") else "Variables mecánicas")
    ax[0].set_ylabel("Sesión de entrenamiento")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="AUC")
    fig.savefig(os.path.join(F, "fig4_transferencia.png")); plt.close(fig)


def fig_puerta():
    r = J("e4.json"); t = np.arange(-2000, 601, 10)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
    for clave, nombre, col in [("episodio", "Agrupada por episodio", AZUL), ("sesion", "Sesión excluida", ROJO)]:
        m = np.array(r[clave]["trayectoria"]["curva_media"]); e = np.array(r[clave]["trayectoria"]["curva_et"])
        ax[0].plot(t, m, color=col, lw=1.3, label=nombre); ax[0].fill_between(t, m - e, m + e, color=col, alpha=0.18, lw=0)
        ini = r[clave]["trayectoria"]["B"]["inicio_ms"]
        if ini is not None and r[clave]["trayectoria"]["B"]["p"] < 0.05:
            ax[0].axvline(ini, color=col, lw=0.8, ls="--")
    ax[0].axvline(0, color="k", lw=0.7); ax[0].axvspan(-1000, -500, color=GRIS, alpha=0.12, lw=0)
    ax[0].set_xlabel("Tiempo respecto del inicio mecánico (ms)"); ax[0].set_ylabel("Probabilidad de intención")
    ax[0].legend(fontsize=7.5, loc="upper left"); ax[0].set_title("Trayectoria de la puerta de intención")
    c = r.get("curva_calibracion", {})
    if c:
        K = sorted(int(k) for k in next(iter(c.values())))
        tr = np.array([[c[s][str(k)]["transfer_media"] for k in K] for s in c])
        so = np.array([[c[s][str(k)]["solo_media"] if c[s][str(k)]["solo_media"] is not None else np.nan for k in K] for s in c])
        ax[1].plot(K, tr.mean(0), "-o", color=AZUL, ms=4, label="Otras sesiones + k episodios")
        ax[1].plot(K, np.nanmean(so, 0), "--s", color=GRIS, ms=4, mfc="white", label="Solo k episodios")
        for fila in tr: ax[1].plot(K, fila, color=AZUL, lw=0.5, alpha=0.35)
        ax[1].axhline(0.5, color="k", lw=0.7, ls=":"); ax[1].set_xticks(K)
        ax[1].set_xlabel("Episodios etiquetados de la sesión nueva (k)"); ax[1].set_ylabel("AUC en el resto de la sesión")
        ax[1].legend(fontsize=7, loc="lower right"); ax[1].set_title("Calibración de sesión")
    fig.savefig(os.path.join(F, "fig5_puerta.png")); plt.close(fig)


def fig_normalizacion():
    modos = [("N1", "Calibración\n(tesis)"), ("N0", "Solo log"), ("N2", "Acumulada\ncausal"), ("N3", "Exponencial\ncausal"), ("N4", "Mediana/IQR\n(no causal)")]
    modos = [(m, n) for m, n in modos if os.path.exists(os.path.join(R, f"e1_{m}.json"))]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for a, clave, titulo in [(ax[0], "EMG14|HGB", "sEMG, refuerzo por gradiente"), (ax[1], "EMG14|LDA", "sEMG, discriminante lineal")]:
        for e, nombre, col, mk in [("A_aleatoria", "Aleatoria", GRIS, "^"), ("B_bloque", "Por bloque", NARANJA, "s"), ("C_sesion", "Sesión excluida", AZUL, "o")]:
            ys, lo, hi = [], [], []
            for m, _ in modos:
                x = J(f"e1_{m}.json")["resultados"][clave][e]
                ys.append(np.mean(x.get("auc_repeticiones", [x["auc"]]))); lo.append(x["ic95"][0]); hi.append(x["ic95"][1])
            ys = np.array(ys)
            a.errorbar(range(len(modos)), ys, yerr=[np.maximum(0, ys - lo), np.maximum(0, np.array(hi) - ys)], fmt="-" + mk, color=col, ms=4.5, lw=0.9, capsize=2, label=nombre)
        a.axhline(0.5, color="k", lw=0.7, ls=":"); a.set_xticks(range(len(modos))); a.set_xticklabels([n for _, n in modos], fontsize=7.5); a.set_title(titulo)
    ax[0].set_ylabel("AUC, reposo frente a gesto"); ax[0].legend(fontsize=7.5)
    fig.savefig(os.path.join(F, "fig6_normalizacion.png")); plt.close(fig)


def fig_n_fijo():
    r = J("e6.json"); ds = [10, 30, 50, 100, 150, 300]; x = [100 * (1 - d / 300) for d in ds]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for clave, nombre, col, mk in [("EMG14|HGB", "sEMG, refuerzo por gradiente", AZUL, "o"), ("EMG14|LDA", "sEMG, discriminante lineal", GRIS, "s"),
                                   ("MEC7|HGB", "Mecánicas, refuerzo por gradiente", NARANJA, "o")]:
        M = np.array([[np.mean(r[clave][str(d)][s]) for d in ds] for s in SESIONES])
        ax[0].errorbar(x, M.mean(0), yerr=M.std(0), fmt="-" + mk, color=col, ms=4, lw=1.1, capsize=2, label=nombre)
    ax[0].axhline(0.5, color="k", lw=0.7, ls=":"); ax[0].set_xlabel("Muestras compartidas con la ventana\nde entrenamiento más próxima (%)")
    ax[0].set_ylabel("AUC"); ax[0].set_title("Datos de la tesis (5 sesiones)"); ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_ylim(0.45, 1.02)
    if os.path.exists(os.path.join(R, "e10.json")):
        e = J("e10.json")["sujetos"]; M = np.array([[v["solapamiento_HGB"][str(d)] for d in ds] for v in e.values()])
        for fila in M: ax[1].plot(x, fila, color=VERDE, lw=0.5, alpha=0.35)
        ax[1].errorbar(x, M.mean(0), yerr=M.std(0), fmt="-o", color=VERDE, ms=4, lw=1.3, capsize=2, label="sEMG, refuerzo por gradiente")
        g = np.mean([v["HGB"]["auc_registro"] for v in e.values()]); ax[1].axhline(g, color=ROJO, lw=0.9, ls="--"); ax[1].text(2, g + 0.008, "agrupada por registro", fontsize=7, color=ROJO)
        ax[1].set_xlabel("Muestras compartidas con la ventana\nde entrenamiento más próxima (%)"); ax[1].set_title("Conjunto público (10 sujetos)"); ax[1].legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(F, "fig7_solapamiento_n_fijo.png")); plt.close(fig)


def fig_simulacion():
    r = J("e7.json"); e2 = J("e2_N1.json")
    fig, ax = plt.subplots(figsize=(7.0, 3.2)); esq = ["aleatoria", "bloque", "purga1", "purga2"]
    series = [("alternadas|tau20|HGB", "Simulación, etiquetas alternadas", NARANJA), ("azar|tau20|HGB", "Simulación, etiquetas al azar por bloque", GRIS)]
    for k, (clave, nombre, col) in enumerate(series):
        m = [np.mean(r[clave][e]) for e in esq]; s = [np.std(r[clave][e]) for e in esq]
        ax.bar(np.arange(4) + (k - 1) * 0.27, m, 0.25, yerr=s, color=col, capsize=2, label=nombre, error_kw=dict(lw=0.8))
    pu = e2["purga_adyacentes"]["EMG14|HGB"]
    real = [np.mean([v["A"] for v in e2["barrido_paso"]["EMG14|HGB"]["10"].values()])] + [np.mean([np.mean(v[m]) for v in pu.values()]) for m in "012"]
    sd = [np.std([v["A"] for v in e2["barrido_paso"]["EMG14|HGB"]["10"].values()])] + [np.std([np.mean(v[m]) for v in pu.values()]) for m in "012"]
    ax.bar(np.arange(4) + 0.27, real, 0.25, yerr=sd, color=AZUL, capsize=2, label="Datos reales, sEMG (5 sesiones)", error_kw=dict(lw=0.8))
    ax.axhline(0.5, color="k", lw=0.7, ls=":"); ax.set_xticks(range(4)); ax.set_xticklabels(["Aleatoria", "Por bloque", "Purga ±1", "Purga ±2"])
    ax.set_ylabel("AUC"); ax.set_ylim(0, 1.12); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Sin información de clase, la alternancia de etiquetas reproduce el patrón")
    fig.savefig(os.path.join(F, "fig8_simulacion.png")); plt.close(fig)


def fig_cnn():
    r = J("e9.json"); esq = [("aleatoria", "Aleatoria"), ("bloque", "Por bloque"), ("sesion", "Sesión\nexcluida"), ("carga", "Carga\nexcluida")]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True, gridspec_kw=dict(wspace=0.06))
    for a, c, titulo in [(ax[0], "C", "Fusión (15 entradas)"), (ax[1], "A", "sEMG sola (8 entradas)"), (ax[2], "B", "Mecánicas solas (7 entradas)")]:
        if not any(k.startswith(c + "|") for k in r): a.set_title(titulo); continue
        for j, (met, nombre, col) in enumerate([("exactitud", "Exactitud", AZUL), ("f1_macro", "F1 macro", NARANJA), ("cond_gesto", "Condicionada al gesto", VERDE)]):
            ms, ss = [], []
            for e, _ in esq:
                ks = [k for k in r if k == f"{c}|{e}" or k.startswith(f"{c}|{e}|s")]
                if e in ("sesion", "carga"): v = [p[met] for p in r[f"{c}|{e}"]["pliegues"].values()] if f"{c}|{e}" in r else [np.nan]
                else: v = [r[k]["media"][met] for k in ks if "media" in r[k]] or [np.nan]
                ms.append(np.mean(v)); ss.append(np.std(v) if len(v) > 1 else 0)
            a.bar(np.arange(4) + (j - 1) * 0.27, ms, 0.25, yerr=ss, color=col, capsize=2, label=nombre, error_kw=dict(lw=0.8))
        a.axhline(0.563, color="k", lw=0.8, ls="--"); a.axhline(0.2, color=VERDE, lw=0.7, ls=":")
        a.set_xticks(range(4)); a.set_xticklabels([n for _, n in esq], fontsize=8); a.set_title(titulo)
    ax[0].text(3.45, 0.575, "clase mayoritaria", fontsize=7, ha="right"); ax[0].set_ylabel("Seis clases"); ax[0].set_ylim(0, 1.05); ax[-1].legend(fontsize=7.5, loc="upper right")
    fig.savefig(os.path.join(F, "fig9_cnn_lstm.png")); plt.close(fig)


def fig_replica():
    e = J("e10.json")["sujetos"]; fig, ax = plt.subplots(figsize=(4.6, 3.3))
    a = [v["HGB"]["auc_aleatoria"] for v in e.values()]; p = [np.mean(v["permutacion_HGB_auc_aleatoria"]) for v in e.values()]; g = [v["HGB"]["auc_registro"] for v in e.values()]
    for i in range(len(a)): ax.plot([0, 1, 2], [a[i], p[i], g[i]], "-o", color=GRIS, lw=0.6, ms=3, alpha=0.7)
    ax.plot([0, 1, 2], [np.mean(a), np.mean(p), np.mean(g)], "-o", color=AZUL, lw=1.6, ms=6)
    ax.axhline(0.5, color="k", lw=0.7, ls=":"); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Aleatoria", "Aleatoria,\netiquetas\nreasignadas", "Agrupada\npor registro"], fontsize=8)
    ax.set_ylabel("AUC macro, carga (3 clases)"); ax.set_ylim(0.45, 1.02); ax.set_xlim(-0.3, 2.3); ax.set_title("Conjunto público, 10 sujetos")
    fig.savefig(os.path.join(F, "fig10_replica.png")); plt.close(fig)


for f in (fig_protocolo, fig_escalera, fig_solapamiento, fig_geometria, fig_transferencia, fig_puerta, fig_normalizacion, fig_n_fijo, fig_simulacion, fig_cnn, fig_replica):
    try:
        f(); print("ok", f.__name__)
    except Exception as ex:
        print("FALLO", f.__name__, repr(ex))
