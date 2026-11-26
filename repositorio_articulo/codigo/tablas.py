"""Vuelca a Markdown las tablas del informe directamente desde los JSON de resultados,"""
import json, os, sys
import numpy as np

R, SAL = sys.argv[1], sys.argv[2]
J = lambda n: json.load(open(os.path.join(R, n)))
SES = ["20260806_182858", "20260806_193017", "20260807_150646", "20260807_170550", "20260811_122212"]
ET = ["S1", "S2", "S3", "S4", "S5"]
NOM = {"EMG8": "sEMG, amplitud (8)", "EMG14": "sEMG, amplitud y espectro (14)", "MEC7": "Mecánicas (7)", "FUS15": "Fusión (15)"}
MOD = {"HGB": "Refuerzo por gradiente", "LDA": "Discriminante lineal"}
f3 = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}".replace(".", ",")
f4 = lambda v: f"{v:.4f}".replace(".", ",")
out = []


def tabla(titulo, cab, filas):
    out.append(f"\n**{titulo}**\n")
    out.append("| " + " | ".join(cab) + " |"); out.append("|" + "|".join(["---"] * len(cab)) + "|")
    for f in filas: out.append("| " + " | ".join(str(x) for x in f) + " |")


if os.path.exists(os.path.join(R, "e0_tabla421.json")):
    r = J("e0_tabla421.json")
    tabla("Tabla R0. Réplica de la Tabla 4.21 de la tesis (fusión de 15 componentes, refuerzo por gradiente, dentro de sesión)",
          ["Sesión", "Bloques", "Ventanas", "Aleatoria", "Por bloque (una asignación)"],
          [[ET[SES.index(s)], v["bloques"], v["ventanas"], f4(v["aleatoria"]), f3(v["por_bloque"])] for s, v in r.items()])

r = J("e1_N1.json"); res = r["resultados"]
filas = []
for k, v in res.items():
    cj, mod = k.split("|")
    def celda(e):
        if e not in v: return "—"
        x = v[e]; m = np.mean(x.get("auc_repeticiones", [x["auc"]]))
        return f"{f3(float(m))} [{f3(x['ic95'][0])}; {f3(x['ic95'][1])}]"
    filas.append([NOM[cj], MOD[mod], celda("A_aleatoria"), celda("B_bloque"), celda("C_sesion"), celda("D_carga")])
tabla("Tabla R1. Escalera de particiones, tarea reposo/gesto, sesiones mezcladas. AUC [IC 95 % por remuestreo de bloques]. "
      + f"{r['n_ventanas']:,}".replace(",", ".") + f" ventanas, {r['n_bloques']} bloques.",
      ["Características", "Modelo", "A. Aleatoria", "B. Por bloque", "C. Sesión excluida", "D. Carga excluida"], filas)

filas = []
for k, v in res.items():
    cj, mod = k.split("|")
    for e, n in [("A_aleatoria", "A"), ("B_bloque", "B"), ("C_sesion", "C"), ("D_carga", "D")]:
        if e in v:
            filas.append([NOM[cj], MOD[mod], n] + [f3(v[e]["auc_por_sesion"].get(s)) for s in SES] + [f3(v[e]["bal_acc"])])
tabla("Tabla R2. Escalera de particiones: AUC evaluada dentro de cada sesión de prueba y exactitud balanceada agregada",
      ["Características", "Modelo", "Esquema"] + ET + ["Exactitud balanceada"], filas)

filas = []
for k, v in res.items():
    cj, mod = k.split("|")
    a = [v["intra_A"][s]["auc"] for s in SES]; b = [float(np.mean(v["intra_B"][s]["auc_reps"])) for s in SES]
    sd = [float(np.std(v["intra_B"][s]["auc_reps"])) for s in SES]
    filas.append([NOM[cj], MOD[mod], f4(float(np.mean(a))), f"{f3(min(a))} a {f3(max(a))}", f3(float(np.mean(b))), f"{f3(min(b))} a {f3(max(b))}", f3(float(np.mean(sd)))])
tabla("Tabla R3. Dentro de sesión: aleatoria frente a agrupada por bloque (media de las cinco sesiones; la agrupada, sobre 5 asignaciones de bloques a pliegues)",
      ["Características", "Modelo", "Aleatoria", "Rango entre sesiones", "Por bloque", "Rango entre sesiones", "DE entre asignaciones"], filas)

r2 = J("e2_N1.json"); b = r2["barrido_paso"]; pasos = [10, 30, 50, 100, 150, 300]
filas = []
for k, v in b.items():
    cj, mod = k.split("|")
    for esq in ("A", "B"):
        filas.append([NOM[cj], MOD[mod], "Aleatoria" if esq == "A" else "Por bloque"] +
                     [f3(float(np.mean([np.mean(x[esq]) for x in v[str(p)].values()]))) for p in pasos])
tabla("Tabla R4. Barrido del paso entre ventanas, dentro de sesión (AUC media de las cinco sesiones)",
      ["Características", "Modelo", "Partición"] + [f"{p} ms ({100*(1-p/300):.1f} %)".replace(".", ",") for p in pasos], filas)

filas = []
for k in r2["purga_adyacentes"]:
    cj, mod = k.split("|"); v = r2["purga_adyacentes"][k]
    perm = r2["permutacion"].get(k)
    filas.append([NOM[cj], MOD[mod],
                  f3(float(np.mean([np.mean(x['A']) for x in perm.values()]))) if perm else "—",
                  f3(float(np.mean([np.mean(x['B']) for x in perm.values()]))) if perm else "—",
                  f3(float(np.mean([np.mean(x['0']) for x in v.values()]))),
                  f3(float(np.mean([np.mean(x['1']) for x in v.values()]))),
                  f3(float(np.mean([np.mean(x['2']) for x in v.values()]))),
                  f3(r2["division_intra_bloque"][k] and float(np.mean(list(r2["division_intra_bloque"][k].values()))))])
tabla("Tabla R5. Controles dentro de sesión (AUC media de las cinco sesiones). Etiquetas permutadas a nivel de bloque; "
      "agrupada por bloque en 10 pliegues con purga de los bloques adyacentes al de prueba; división temporal dentro del bloque",
      ["Características", "Modelo", "Aleatoria, etiquetas permutadas", "Por bloque, etiquetas permutadas", "Por bloque, sin purga", "Purga ±1", "Purga ±2", "60/40 dentro del bloque"], filas)
vt = r2.get("vecino_temporal", {})
if vt:
    out.append(f"\nReferencia del vecino temporal (asignar la etiqueta del bloque de entrenamiento más cercano en el tiempo): AUC media "
               f"{f3(float(np.mean([np.mean(v) for v in vt.values()])))}, rango {f3(float(min(min(v) for v in vt.values())))} a {f3(float(max(max(v) for v in vt.values())))}.\n")

for arch, etq, num in [("e3_N1.json", "normalización de la tesis", "R6a"), ("e3_N2.json", "normalización acumulada causal", "R6c")]:
    if not os.path.exists(os.path.join(R, arch)): continue
    r3 = J(arch)["matrices"]; filas = []
    for k, M in r3.items():
        cj, mod = k.split("|"); M = np.array(M); fuera = M[~np.eye(5, dtype=bool)]
        cero = [M[i, j] for i in range(3) for j in range(3) if i != j]
        dia_mismo = [M[i, j] for i, j in [(0, 1), (1, 0), (2, 3), (3, 2)]]
        filas.append([NOM[cj], MOD[mod], f3(float(np.diag(M).mean())), f3(float(fuera.mean())), f"{f3(float(fuera.min()))} a {f3(float(fuera.max()))}",
                      f3(float(np.mean(cero))), f3(float(np.mean(dia_mismo)))])
    tabla(f"Tabla {num}. Transferencia entre sesiones, {etq}: entrenar en una sesión y probar en otra (AUC)",
          ["Características", "Modelo", "Diagonal (por bloque, intra)", "Fuera de la diagonal", "Rango", "Entre sesiones de 0 kg", "Mismo día"], filas)
    for k in ("EMG14|HGB",):
        if k in r3:
            M = np.array(r3[k])
            tabla(f"Tabla {num[:-1]}{chr(ord(num[-1])+1)}. Matriz completa, {NOM['EMG14']}, refuerzo por gradiente, {etq} (filas: entrenamiento; columnas: prueba)",
                  [""] + ET, [[ET[i]] + [f3(float(M[i, j])) for j in range(5)] for i in range(5)])

if os.path.exists(os.path.join(R, "e4.json")):
    r4 = J("e4.json"); filas = []
    for k, n in [("episodio", "Agrupada por episodio"), ("sesion", "Sesión excluida"),
                 ("episodio_tipificada", "Por episodio, tipificada por sesión"), ("sesion_tipificada", "Sesión excluida, tipificada por sesión")]:
        if k not in r4: continue
        x = r4[k]; t = x["trayectoria"]["B"]
        filas.append([n, f"{f3(x['auc_media'])} ± {f3(x['auc_desv'])}", f3(x["auc_agregada"]), f"{f3(x['ic95'][0])} a {f3(x['ic95'][1])}",
                      " / ".join(f3(x["auc_por_sesion"][s]) for s in SES),
                      "—" if t["inicio_ms"] is None else f"{t['inicio_ms']:+d}", f"{t['masa']:.1f}".replace(".", ","), f4(t["p"])])
    tabla(f"Tabla R7. Puerta de intención ({r4['n_episodios']} episodios, {r4['n_ventanas']} ventanas, bosque aleatorio). "
          "Trayectoria: línea base B, de −1000 a −500 ms",
          ["Validación", "AUC por pliegue", "AUC agregada", "IC 95 % (episodios)", "AUC por sesión S1 / S2 / S3 / S4 / S5", "Inicio de divergencia (ms)", "Masa", "p"], filas)
    filas = []
    for k, n in [("episodio", "Agrupada por episodio"), ("sesion", "Sesión excluida")]:
        for lb in "ABC":
            t = r4[k]["trayectoria"][lb]
            filas.append([n, lb, "—" if t["inicio_ms"] is None else f"{t['inicio_ms']:+d}", f"{t['masa']:.1f}".replace(".", ","), f4(t["p"]),
                          f"{t.get('nulo_p95', float('nan')):.1f}".replace(".", ","), f"{t.get('nulo_max', float('nan')):.1f}".replace(".", ",")])
    tabla("Tabla R8. Sensibilidad de la divergencia a la línea base (A: −2000 a −1000 ms; B: −1000 a −500 ms; C: −900 a −400 ms)",
          ["Validación", "Línea base", "Inicio (ms)", "Masa", "p", "Nulo p95", "Nulo máx."], filas)
    c = r4.get("curva_calibracion")
    if c:
        K = sorted(int(k) for k in next(iter(c.values())))
        filas = [[ET[SES.index(s)], r4["episodios_por_sesion"][s]] + [f"{f3(c[s][str(k)]['transfer_media'])} ± {f3(c[s][str(k)]['transfer_desv'])}" for k in K] for s in c]
        filas.append(["Media", ""] + [f3(float(np.mean([c[s][str(k)]["transfer_media"] for s in c]))) for k in K])
        filas.append(["Solo k episodios (media)", ""] + [f3(float(np.mean([c[s][str(k)]["solo_media"] for s in c]))) if k > 0 else "—" for k in K])
        tabla(f"Tabla R9. Curva de calibración de sesión: AUC sobre los episodios de prueba de la sesión excluida al devolverle k episodios etiquetados "
              f"({r4['curva_calibracion_n_sorteos']} sorteos; media ± DE entre sorteos)", ["Sesión", "Episodios"] + [f"k = {k}" for k in K], filas)

if os.path.exists(os.path.join(R, "e4b.json")):
    b4 = J("e4b.json"); filas = []
    for lb in "ABC":
        x = b4[lb]
        filas.append([lb, f"{x['episodio']['puntual']:+d} [{x['episodio']['ic'][0]:+.0f}; {x['episodio']['ic'][2]:+.0f}]", f"{100*x['episodio']['frac_antes']:.0f} %",
                      f"{x['sesion']['puntual']:+d} [{x['sesion']['ic'][0]:+.0f}; {x['sesion']['ic'][2]:+.0f}]", f"{100*x['sesion']['frac_antes']:.0f} %",
                      f"{x['diferencia']['ic'][1]:+.0f} [{x['diferencia']['ic'][0]:+.0f}; {x['diferencia']['ic'][2]:+.0f}]", f"{100*x['diferencia']['frac_positiva']:.0f} %"])
    tabla("Tabla R8b. Incertidumbre del instante de divergencia por remuestreo pareado de episodios (2000 réplicas). Inicio en ms [IC 95 %]; "
          "proporción de réplicas con inicio anterior al movimiento; diferencia sesión − episodio (mediana [IC 95 %]) y proporción de réplicas en que es positiva",
          ["Línea base", "Por episodio", "Antes de 0", "Sesión excluida", "Antes de 0", "Diferencia (ms)", "Diferencia > 0"], filas)
if os.path.exists(os.path.join(R, "e4c.json")):
    c4 = J("e4c.json"); filas = []
    def fila(n, x): return [n, f3(x["auc_episodio"]), f3(x["auc_sesion"]), f"{x['delta']:+.3f}".replace(".", ","), f"{x['ic95'][0]:+.3f} a {x['ic95'][1]:+.3f}".replace(".", ","), f"{100*x['frac_positiva']:.0f} %"]
    filas.append(fila("Los 61 episodios", c4["todas"]))
    for i, s in enumerate(SES): filas.append(fila(f"Solo {ET[i]}", c4["solo_" + s]))
    for i, s in enumerate(SES): filas.append(fila(f"Sin {ET[i]}", c4["sin_" + s]))
    tabla("Tabla R8c. Diferencia de AUC de la puerta entre validación por episodio y sesión excluida, con remuestreo pareado de episodios",
          ["Episodios evaluados", "AUC por episodio", "AUC sesión excluida", "Diferencia", "IC 95 %", "Réplicas con diferencia > 0"], filas)
if os.path.exists(os.path.join(R, "e5b.json")):
    b5 = J("e5b.json"); filas = []
    for cj, v in b5.items():
        filas.append([NOM[cj]] + [f"{f3(v[s]['con_S1'])} / {f3(v[s]['sin_S1'])}" for s in SES[1:]] +
                     [f"{f3(float(np.mean([v[s]['con_S1'] for s in SES[1:]])))} / {f3(float(np.mean([v[s]['sin_S1'] for s in SES[1:]])))}"])
    tabla("Tabla R11. Sesión excluida con y sin la sesión de IMU congelada (S1) en el entrenamiento. AUC con S1 / sin S1, refuerzo por gradiente",
          ["Características"] + ET[1:] + ["Media"], filas)

modos = [("N1", "Calibración de sesión, recorte ±10 (tesis)"), ("N0", "Solo log1p"), ("N2", "Acumulada causal"), ("N3", "Exponencial causal, 60 s"), ("N4", "Mediana/IQR de sesión (no causal)")]
filas = []
for m, n in modos:
    a = os.path.join(R, f"e1_{m}.json")
    if not os.path.exists(a): continue
    rr = J(f"e1_{m}.json")["resultados"]
    for k in ("EMG14|HGB", "EMG14|LDA"):
        if k not in rr: continue
        v = rr[k]
        filas.append([n, MOD[k.split("|")[1]], f3(v["A_aleatoria"]["auc"]), f3(float(np.mean(v["B_bloque"]["auc_repeticiones"]))),
                      f"{f3(v['C_sesion']['auc'])} [{f3(v['C_sesion']['ic95'][0])}; {f3(v['C_sesion']['ic95'][1])}]",
                      f3(v["C_sesion"]["auc_media_sesiones"]), f3(v["D_carga"]["auc"]),
                      f3(float(np.mean([v['intra_A'][s]['auc'] for s in SES]))), f3(float(np.mean([np.mean(v['intra_B'][s]['auc_reps']) for s in SES])))])
if filas:
    tabla("Tabla R10. Efecto de la normalización de la sEMG (14 características) sobre cada peldaño (AUC)",
          ["Normalización", "Modelo", "A. Aleatoria", "B. Por bloque", "C. Sesión excluida [IC 95 %]", "C, media por sesión", "D. Carga excluida", "Intra, aleatoria", "Intra, por bloque"], filas)

if os.path.exists(os.path.join(R, "e6.json")):
    r6 = J("e6.json"); ds = [10, 30, 50, 100, 150, 300]; filas = []
    for k, v in r6.items():
        cj, mod = k.split("|")
        filas.append([NOM[cj], MOD[mod]] + [f3(float(np.mean([np.mean(x) for kk, x in v[str(d)].items()]))) for d in ds])
    tabla("Tabla R12. Solapamiento con el número de ventanas igualado (E6). Entrenamiento fijo, una ventana cada 600 ms por bloque; "
          "prueba desplazada. AUC media de las cinco sesiones y de seis rejillas de entrenamiento, según la fracción de muestras compartidas",
          ["Características", "Modelo"] + [f"{100*(1-d/300):.1f} %".replace(".", ",") for d in ds], filas)
if os.path.exists(os.path.join(R, "e7.json")):
    r7 = J("e7.json"); filas = []
    ms = lambda v: f"{f3(float(np.mean(v)))} ± {f3(float(np.std(v)))}"
    for k, v in r7.items():
        et, tau, mod = k.split("|")
        filas.append(["Alternadas" if et == "alternadas" else "Al azar por bloque", tau[3:] + " s", MOD[mod], ms(v["aleatoria"]), ms(v["bloque"]), ms(v["purga1"]), ms(v["purga2"])])
    tabla("Tabla R13. Simulación (E7): características sin información de clase, AUC verdadera 0,5. Media ± DE de 10 sesiones sintéticas",
          ["Etiquetas", "Constante de la deriva", "Modelo", "Aleatoria", "Por bloque", "Purga ±1", "Purga ±2"], filas)
if os.path.exists(os.path.join(R, "e8.json")):
    r8 = J("e8.json"); b = r8["base_mayoritaria"]; filas = []
    for k, v in r8["resultados"].items():
        cj, mod = k.split("|")
        for e, n in zip("ABCD", ["A. Aleatoria", "B. Por bloque", "C. Sesión excluida", "D. Carga excluida"]):
            x = v[e]
            filas.append([NOM[cj], MOD[mod], n, f"{f3(x['exactitud'])} [{f3(x['ic_exactitud'][0])}; {f3(x['ic_exactitud'][1])}]",
                          f"{f3(x['f1_macro'])} [{f3(x['ic_f1'][0])}; {f3(x['ic_f1'][1])}]", f3(x["cond_gesto"])])
    tabla(f"Tabla R14. Escalera en seis clases (E8), cuatro sesiones completas, {r8['n_ventanas']} ventanas y {r8['n_bloques']} bloques. "
          f"Línea base de la clase mayoritaria: exactitud {f3(b['exactitud'])}, F1 macro {f3(b['f1_macro'])}, condicionada al gesto 0,200. IC 95 % por remuestreo de bloques",
          ["Características", "Modelo", "Esquema", "Exactitud", "F1 macro", "Condicionada al gesto"], filas)
if os.path.exists(os.path.join(R, "e9.json")):
    r9 = J("e9.json"); filas = []
    NC = {"C": "Fusión (15)", "A": "sEMG, amplitud (8)", "B": "Mecánicas (7)"}
    NE = {"aleatoria": "Aleatoria por secuencias", "bloque": "Agrupada por bloque", "sesion": "Sesión excluida", "carga": "Carga excluida"}
    for c in "CAB":
        for e in ("aleatoria", "bloque", "sesion", "carga"):
            k = f"{c}|{e}"
            if k not in r9 or "media" not in r9[k]: continue
            if e in ("aleatoria", "bloque"):
                ks = [q for q in r9 if (q == k or q.startswith(k + "|s")) and "media" in r9[q]]
                M = {q: [r9[z]["media"][q] for z in ks] for q in ("exactitud", "f1_macro", "cond_gesto", "base_mayoritaria", "f1_base")}
                pm = lambda q: f3(float(np.mean(M[q]))) + (f" ± {f3(float(np.std(M[q])))}" if len(ks) > 1 else "")
                filas.append([NC[c], NE[e], f"1 división × {len(ks)} semilla" + ("s" if len(ks) > 1 else ""), pm("exactitud"), pm("f1_macro"), pm("cond_gesto"),
                              f3(float(np.mean(M["base_mayoritaria"]))), f3(float(np.mean(M["f1_base"]))),
                              " / ".join(str(r9[z]["pliegues"]["unico"]["epocas"]) for z in ks)])
                continue
            m, d, pl = r9[k]["media"], r9[k]["desv"], r9[k]["pliegues"]
            pm = lambda q: f3(m[q]) + (f" ± {f3(d[q])}" if len(pl) > 1 else "")
            filas.append([NC[c], NE[e], f"{len(pl)} pliegues", pm("exactitud"), pm("f1_macro"), pm("cond_gesto"), f3(m["base_mayoritaria"]), f3(m["f1_base"]),
                          " / ".join(str(p["epocas"]) for p in pl.values())])
    tabla("Tabla R15. CNN-LSTM de referencia de la tesis (E9) bajo cuatro particiones. Media ± DE entre pliegues (sesión y carga) o entre semillas de inicialización y de división (aleatoria y por bloque)",
          ["Entradas", "Partición", "Repeticiones", "Exactitud", "F1 macro", "Condicionada al gesto", "Base mayoritaria", "F1 de la base", "Épocas"], filas)
if os.path.exists(os.path.join(R, "e10.json")):
    r10 = J("e10.json"); filas = []; ds = ["10", "30", "50", "100", "150", "300"]
    for s, v in r10["sujetos"].items():
        filas.append([s, v["n_ventanas"], f3(v["HGB"]["auc_aleatoria"]), f3(v["HGB"]["auc_registro"]), f3(float(np.mean(v["permutacion_HGB_auc_aleatoria"]))),
                      f3(v["LDA"]["auc_aleatoria"]), f3(v["LDA"]["auc_registro"]), f3(v["solapamiento_HGB"]["10"]), f3(v["solapamiento_HGB"]["300"])])
    S = r10["sujetos"]; med = lambda f: f3(float(np.mean([f(v) for v in S.values()])))
    filas.append(["Media", "", med(lambda v: v["HGB"]["auc_aleatoria"]), med(lambda v: v["HGB"]["auc_registro"]), med(lambda v: np.mean(v["permutacion_HGB_auc_aleatoria"])),
                  med(lambda v: v["LDA"]["auc_aleatoria"]), med(lambda v: v["LDA"]["auc_registro"]), med(lambda v: v["solapamiento_HGB"]["10"]), med(lambda v: v["solapamiento_HGB"]["300"])])
    tabla("Tabla R16. Réplica en el conjunto público de codo (E10), fs = " + str(r10['fs']).replace(".", ",") + " Hz: clasificación de la carga, AUC macro por sujeto. "
          "Aleatoria por ventanas frente a agrupada por registro; etiquetas reasignadas entre registros; solapamiento con N igualado",
          ["Sujeto", "Ventanas", "RG aleatoria", "RG por registro", "RG aleatoria, etiquetas reasignadas", "DL aleatoria", "DL por registro", "RG, 96,7 % compartido", "RG, 0 % compartido"], filas)
    for mod, n in (("HGB", "Refuerzo por gradiente"), ("LDA", "Discriminante lineal")):
        x = r10["resumen_" + mod]
        out.append(f"\n{n}: aleatoria {f3(x['aleatoria_media'])} ± {f3(x['aleatoria_desv'])}; por registro {f3(x['registro_media'])} ± {f3(x['registro_desv'])}; "
                   f"diferencia mediana {f3(x['diferencia_mediana'])}; aleatoria mayor en {x['sujetos_con_aleatoria_mayor']} de 10 sujetos; Wilcoxon unilateral p = {x['wilcoxon_p_unilateral']:.4f}.".replace("p = 0.", "p = 0,"))
    out.append("\nRG: refuerzo por gradiente. DL: discriminante lineal.\n")


if os.path.exists(os.path.join(R, "a2.json")):
    a2 = J("a2.json"); filas = []
    NCL = ["Reposo", "Flex. codo", "Ext. codo", "Flex. hombro", "Ext. hombro", "Levantamiento"]
    for k, n in [("EMG14|A", "sEMG (14), aleatoria"), ("EMG14|C", "sEMG (14), sesión excluida"), ("FUS15|A", "Fusión (15), aleatoria"), ("FUS15|C", "Fusión (15), sesión excluida")]:
        M = np.array(a2[k]["por_fila"]); filas.append([n] + [f3(M[i, i]) for i in range(6)])
    tabla("Tabla R17. Sensibilidad por clase (diagonal de la matriz de confusión normalizada por fila), seis clases, refuerzo por gradiente (E8). Azar: 0,167",
          ["Configuración"] + NCL, filas)
    for k, n in [("EMG14|C", "sEMG (14), sesión excluida"), ("FUS15|C", "Fusión (15), sesión excluida")]:
        M = np.array(a2[k]["por_fila"])
        tabla(f"Tabla R17{'a' if k.startswith('EMG') else 'b'}. Matriz de confusión normalizada por fila, {n}. Filas: clase real; columnas: predicha",
              ["Real \\ Predicha"] + NCL, [[NCL[i]] + [f3(M[i, j]) for j in range(6)] for i in range(6)])
if os.path.exists(os.path.join(R, "a1.json")):
    a1 = J("a1.json")
    tabla("Tabla R18. Error de Monte Carlo de los IC por remuestreo de bloques (sEMG 14, discriminante lineal, sesión excluida; 4 semillas del remuestreo)",
          ["Réplicas", "DE del límite inferior", "DE del límite superior"], [[n, f3(a1[n]["de_limite_inferior"]), f3(a1[n]["de_limite_superior"])] for n in ("500", "2000")])


if os.path.exists(os.path.join(R, "e10_fs780.json")) and os.path.exists(os.path.join(R, "e10.json")):
    filas = []
    for n, f in (("1024 Hz (declarada en el artículo, principal)", "e10.json"), ("780,2 Hz (sensibilidad)", "e10_fs780.json")):
        q = J(f); S = q["sujetos"]; m = lambda g: f3(float(np.mean([g(v) for v in S.values()])))
        filas.append([n, q["ventana_muestras"], q["paso_muestras"], f3(q["resumen_HGB"]["aleatoria_media"]), f3(q["resumen_HGB"]["registro_media"]), q["resumen_HGB"]["sujetos_con_aleatoria_mayor"],
                      f3(q["resumen_LDA"]["aleatoria_media"]), f3(q["resumen_LDA"]["registro_media"]), m(lambda v: np.mean(v["permutacion_HGB_auc_aleatoria"])),
                      m(lambda v: v["solapamiento_HGB"]["10"]), m(lambda v: v["solapamiento_HGB"]["300"])])
    tabla("Tabla R19. Sensibilidad de E10 a la frecuencia de muestreo. AUC macro, media de 10 sujetos",
          ["fs", "Ventana (muestras)", "Paso (muestras)", "HGB aleatoria", "HGB por registro", "Sujetos con aleatoria > registro", "LDA aleatoria", "LDA por registro",
           "HGB etiquetas reasignadas", "HGB 96,7 % compartido", "HGB 0 % compartido"], filas)
if os.path.exists(os.path.join(R, "a3.json")) and os.path.exists(os.path.join(R, "a3b.json")):
    a3 = J("a3.json"); t = a3["transferencia"]; b3 = J("a3b.json"); c = lambda x, d=0: f"{x:.{d}f}".replace(".", ",")
    tabla("Tabla R20. Frecuencia de muestreo del conjunto público: lo que declara el artículo asociado frente a lo que muestran los datos (A3 y A3b)",
          ["Evidencia", "Resultado", "Lectura"],
          [["Artículo asociado (Toro-Ossaba et al., 2024, sección 2.5)", "Shimmer3 configurado a 1024 Hz para sEMG e IMU", "Fuente primaria"],
           ["Punto de −3 dB del pasaaltos, medido entre columna cruda y filtrada", f"{c(b3['punto_menos3dB_ciclos_muestra'], 5)} ciclos/muestra → {c(b3['fs_si_10Hz_es_menos3dB'])} Hz si 10 Hz es el punto de media potencia", "Compatible con 1024 Hz"],
           ["Borde de rizado del ajuste Chebyshev (convención de scipy)", f"{c(t['borde_inferior_ciclos_muestra'], 5)} ciclos/muestra → {c(t['fs_si_inferior_10Hz'])} Hz", "Depende de la convención de corte; descartado"],
           ["Atenuación a 500 Hz si fs = 1024 Hz", f"Ganancia {c(b3['ganancia_a_500Hz_si_fs1024'], 3)}: el pasabajos de 500 Hz no tiene efecto visible", "Sin consecuencia: E10 filtra la señal cruda"],
           ["Fase a 0,02 ciclos/muestra", f"{c(t['fase_grados_0_02'])}°", "Filtro causal"],
           ["Repeticiones por archivo (ángulo)", f"Mediana {c(b3['picos_por_archivo_mediana'])} picos en {b3['archivos']} archivos", "Coincide con series de diez repeticiones"],
           ["Periodo del ciclo (mediana de muestras)", f"{c(b3['periodo_muestras_mediana'])} muestras: {c(b3['segundos_por_ciclo']['1024'], 2)} s a 1024 Hz; {c(b3['segundos_por_ciclo']['780.2'], 2)} s a 780,2 Hz", "El artículo indica unos 2 s/ciclo: más cercano a 1024 Hz"]])
if os.path.exists(os.path.join(R, "a4.json")):
    a4 = J("a4.json"); filas = []
    ic = lambda v: f"[{f3(v[0])}; {f3(v[1])}]"
    for k, v in a4["casos"].items():
        cj, mod, e = k.split("|")
        filas.append([NOM[cj], MOD[mod], {"C": "Sesión excluida", "D": "Carga excluida"}[e], f3(v["auc"]), ic(v["ic_independiente_E1"]),
                      ic(v["ic_moviles_L2"]), ic(v["ic_moviles_L4"]), ic(v["ic_moviles_L8"]), f3(v["autocorr_rezago1_media"])])
    tabla(f"Tabla R21. Bloques vecinos (A4): IC 95 % del AUC con bloques independientes (método de E1) frente a tramos móviles de L bloques consecutivos; {a4['n_rep']} réplicas",
          ["Características", "Modelo", "Partición", "AUC", "Independientes", "L = 2", "L = 4", "L = 8", "Autocorrelación de rezago 1"], filas)

open(SAL, "w", encoding="utf8").write("\n".join(out) + "\n")
print("tablas escritas:", SAL, len(out), "líneas")
