# Procesamiento de la señal

## 01_extraer_features.py

Lee los `emg_*.csv` de cada sesión y escribe un `featok_*.csv`.

**Filtrado causal.** Butterworth de cuarto orden entre 20 y 450 Hz más rechazo
de banda a 60 Hz, aplicado con `lfilter`. La causalidad no es
un detalle de implementación: un filtro de fase cero mira muestras futuras y
anula cualquier afirmación sobre el instante de detección. El coste medido es un
retardo de grupo de 11,1 ms, ponderado por el espectro de los cinco registros,
que entra en el presupuesto de latencia del apartado 4.9.

**Descriptores por canal.** Cuatro de amplitud sobre la señal rectificada —IAV,
SSI, WL y LOG, según Hudgins et al. (1993)— y tres espectrales —ZC, SSC y MNF,
según Phinyomark et al. (2012)—. WL se calcula sobre la señal rectificada y no
sobre la filtrada sin rectificar: el apartado 3.5.3 de la tesis documenta la
comprobación que justifica la sustitución y sus dos salvedades.

**Comprobaciones que el script hace y no debe silenciar.** Frecuencia real de
1000,00 Hz y cero huecos mayores de 2 ms en las cinco sesiones. La IMU congelada
de la sesión 20260806_182858 y las 43 filas con NaN interpoladas de esa misma
sesión son avisos esperados. Si aparece una sesión que no está en
`REGISTRO_SESIONES`, el script se detiene en vez de adivinar la carga.

Tesis: apartados 3.5.1 a 3.5.4.

## 02_construir_secuencias.py

Lee los `featok_*.csv` y escribe `pliegue_0.00kg.npz`, `pliegue_0.50kg.npz`,
`pliegue_0.85kg.npz`, `informe_paso2.json` y `sesiones_parciales.npz`.

**Ventaneo.** Ventana de 300 ms con salto de 10 ms y secuencias de L = 20
vectores, lo que da una cobertura temporal efectiva de 490 ms. Los 300 ms
proceden de Trigili et al. (2019) y se confirmaron por medición propia en
`analisis/10_barrido_ancho_ventana.py`.

**Partición por condición excluida.** Cada condición de carga se reserva entera
como conjunto de prueba, por turnos. La unidad de partición es el bloque, nunca
la ventana: dos ventanas consecutivas comparten el 97 % de sus muestras y una
partición aleatoria produciría fuga de información. La magnitud de esa fuga está
cuantificada en el apartado 4.10 de la tesis.

**Normalización.** La electromiografía se tipifica con los estadísticos del
bloque de calibración de cada sesión, que no contiene etiquetas de gesto. Las
variables mecánicas se tipifican solo con el entrenamiento, para no filtrar
información de la condición excluida. La sesión de 0,85 kg no tiene bloque de
calibración sin carga, y esa desviación se declara en el apartado 3.4.2.

**Recorte de saturación.** Tras normalizar, ψ se recorta al intervalo [−10, +10].
El efecto por variable y por pliegue está en la figura 4.3.

## Las dos configuraciones del vector

El vector ψ se arma en este archivo, en la constante `PSI_COLS`:

    EMG_COLS = IAV, SSI, WL, LOG, ZC, SSC, MNF   por cada canal   -> 14
    MEC_COLS = q_sh, q_el, dq_sh, dq_el, i_sh, F_el, F_sh          ->  7
    PSI_COLS = EMG_COLS + MEC_COLS                                 -> 21

Tal como está, produce la configuración R21. Para reproducir la configuración
principal de la tesis, ψ ∈ ℝ¹⁵ con 31.142 parámetros, hay que dejar en
`EMG_COLS` solo los cuatro descriptores de amplitud por canal y quitar ZC, SSC y
MNF. `modelos/clasificador_gesto/03_entrenar.py` comprueba el número de
parámetros y aborta si no coincide con el de la configuración elegida.

Tesis: apartados 3.5.2, 3.5.5, 3.5.6 y 3.7.1.
