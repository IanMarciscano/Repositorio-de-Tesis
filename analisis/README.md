# Análisis

Scripts independientes. Leen los `.npz` y `.keras` de los pasos anteriores y los
`emg_*.csv` originales, y escriben un JSON cada uno.

| Script | Qué responde | Tesis |
|---|---|---|
| `08_medir_latencia_computo` | cuánto tarda un ciclo del supervisor | 4.9 |
| `09_factor_sesgo_calibracion` | sesgo de los bloques de calibración con carga | 4.2 |
| `10_barrido_ancho_ventana` | por qué la ventana de transición son 300 ms | 4.7 |
| `11_presupuesto_latencia` | retardo extremo a extremo, término a término | 4.9 |
| `12_anticipacion_episodios` | marcas de inicio mecánico y población de episodios | 4.8 |
| `13_contraste_reposo_gesto` | la razón 1,22 entre gesto y reposo | 4.1 |
| `14_falsos_positivos` | efecto de la histéresis sobre las activaciones falsas | 4.14 |
| `15_verificacion_pasarela` | severidad, modos y conmutaciones | 4.19 |

## Comprobaciones de robustez

Se añadieron durante la revisión del manuscrito, con los resultados a la vista.
**No son criterios preespecificados**, y la tesis lo declara así en el apartado
3.7.4. El criterio declarado de antemano era el AUC con intervalo que excluyera
0,5 y la mediana del margen por episodio mayor que cero.

| Script | Qué comprueba |
|---|---|
| `aud01_recorte_por_sesion` | la causa real del recorte por saturación |
| `aud02_ic_puerta` | intervalo de confianza del AUC, 2000 réplicas |
| `aud03_umbral_inicio` | el umbral de inicio frente a la velocidad observada |
| `aud04_ic_anticipacion` | intervalo de confianza del margen de anticipación |
| `aud05_barrido_umbral` | sensibilidad del margen frente al umbral de inicio |
| `aud06_sustitutos_B` | conglomerados y series subrogadas sobre la línea base B |

Sobre `aud06`: las series subrogadas se construyen por desplazamiento circular de
la traza de cada episodio, lo que conserva su forma y su autocorrelación y
destruye solo el anclaje temporal al inicio del movimiento. Un control previo por
permutación del eje temporal resultó inválido por construcción y se descartó; el
apartado 4.8 lo documenta.
