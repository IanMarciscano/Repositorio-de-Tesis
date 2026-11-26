# Plan de análisis fijado antes de correr los experimentos

Fecha: 16 de noviembre de 2026. Escrito antes de ejecutar E1–E5. Cualquier análisis
que se añada después de ver resultados se declarará como exploratorio en el artículo.

## Pregunta

¿Cuánto del desempeño aparente de un decodificador sEMG de intención sobrevive cuando
se controla, por turno, cada dependencia entre entrenamiento y prueba: muestras
compartidas entre ventanas solapadas, pertenencia al mismo bloque, pertenencia a la
misma sesión y condición de carga?

## Datos

Cinco sesiones de un participante (20260806_182858, 20260806_193017, 20260807_150646
a 0 kg; 20260807_170550 a 0,85 kg; 20260811_122212 a 0,50 kg). Características de la
tesis (featok_*.csv, ventana de 300 ms, paso de 10 ms). No se adquiere ningún dato.

## Tarea principal (T1)

Binaria, reposo frente a gesto, por ventana. Gesto: cualquiera de las cinco clases
activas. Reposo: tramos sin etiqueta entre bloques. Se excluyen la calibración y la
bajada, y toda ventana que no esté contenida por completo en un segmento de etiqueta
(t_since_label < 0,3 s). Bloque: segmento continuo de una misma etiqueta. La sesión
20260806_193017 participa, porque la tarea binaria no requiere la clase que le falta.

## Factores fijos

- Conjuntos de características: EMG8 (amplitud), EMG14 (amplitud y espectro),
  MEC7 (variables mecánicas), FUS15 (EMG8 + MEC7, el vector de la tesis).
- Modelos: refuerzo por gradiente basado en histogramas (el de la reconstrucción de
  la tesis) y análisis discriminante lineal con contracción (referencia del campo).
  Hiperparámetros por defecto; no se ajusta ninguno. Semilla 42.
- Normalización de referencia (N1): la de la tesis. sEMG tipificada con el bloque de
  calibración de la sesión y recorte |z| <= 10; mecánicas tipificadas con el
  entrenamiento de cada pliegue.

## E1. Escalera de particiones

A. Aleatoria por ventanas, sesiones mezcladas (5 pliegues estratificados).
B. Agrupada por bloque, sesiones mezcladas (5 pliegues).
C. Una sesión completa excluida (5 pliegues).
D. Una condición de carga excluida (3 pliegues).
Réplica de la Tabla 4.21: A y B dentro de cada sesión.

Métrica primaria: AUC. Secundaria: exactitud balanceada con umbral 0,5.
Incertidumbre: remuestreo de bloques del conjunto de prueba, 1000 réplicas, IC 95 %.
La unidad es el bloque, no la ventana.

## E2. Descomposición de la fuga (dentro de sesión)

- Barrido del paso entre ventanas: 10, 30, 50, 100, 150 y 300 ms
  (solapamiento 96,7; 90; 83,3; 66,7; 50 y 0 %), bajo A y B.
- División temporal dentro del bloque: entrenar con el primer 60 % de cada bloque y
  probar con el último 40 %, con 300 ms de separación.
- Control por permutación de etiquetas a nivel de bloque (20 permutaciones): si la
  partición aleatoria conserva un AUC alto con etiquetas sin sentido, el
  clasificador reconoce el bloque.

Predicción registrada de antemano: con 0 % de solapamiento la partición aleatoria
seguirá por encima de 0,9. Si no ocurre, el mensaje del artículo es el solapamiento.

## E3. Transferencia entre sesiones

Matriz 5 x 5: entrenar en la sesión i, probar en la j. Diagonal: validación agrupada
por bloque dentro de la sesión. Lectura a carga constante: las tres sesiones de 0 kg.

## E4. Puerta de intención

Construcción idéntica a 05_puerta_intencion.py (61 episodios, 14 características
sEMG, bosque aleatorio de 300 árboles). Primero se reproduce el AUC de 0,632 de la
tesis como comprobación del entorno.
- Agrupada por episodio frente a sesión excluida, con IC por remuestreo de episodios.
- Curva de calibración: a la sesión excluida se le devuelven k episodios etiquetados
  (k = 0, 2, 4, 6, 8; 30 repeticiones aleatorias) y se evalúa en los restantes.
  Comparador: entrenar solo con esos k episodios.

## E5. Mecanismos

Se repiten los peldaños B y C con cuatro normalizaciones alternativas: sin
normalización de sesión (solo log1p), acumulada causal, exponencial causal de 60 s y
mediana/IQR de sesión (no causal, cota). Se repite además sin la sesión de IMU
congelada para MEC7 y FUS15.

## Criterio de lectura

Ningún contraste de significación sobre pliegues de sesión o de carga (n = 5 y n = 3).
Se reportan magnitudes, intervalos por remuestreo de bloques o episodios y el valor de
cada sesión por separado.

---

# Adenda: experimentos E6 a E10

Escrita el 16 de noviembre de 2026, después de ver E1–E5 y antes de correr E6–E10.

## E6. Solapamiento con el número de ventanas igualado

En el barrido de E2, reducir el solapamiento reduce también el número de ventanas.
Diseño que lo separa: el conjunto de entrenamiento es siempre el mismo (una ventana cada
600 ms dentro de cada bloque, sin solapamiento entre sí). El de prueba tiene una ventana
por cada una de entrenamiento, desplazada 10, 30, 50, 100, 150 o 300 ms, de modo que
comparte con ella el 96,7; 90; 83,3; 66,7; 50 o 0 % de sus muestras. N y cobertura
temporal son constantes; solo cambia la fracción de muestras compartidas.
Predicción: en sEMG el AUC crece con la fracción compartida; en las mecánicas es alto
en todos los casos.

## E7. Simulación del diseño de bloques alternados

Sesiones sintéticas de 40 bloques con etiquetas alternadas y 14 características SIN
información de clase (AUC verdadera 0,5): un proceso autorregresivo lento más ruido
suavizado sobre 30 pasos, que imita ventanas de 300 ms con paso de 10 ms. Se varía la
constante de tiempo de la deriva (2, 20 y 200 s) y se compara con etiquetas asignadas al
azar por bloque. Mismos esquemas que en E2: aleatoria, por bloque, purga ±1 y ±2.
Predicción: con etiquetas alternadas y deriva lenta, por bloque queda bajo 0,5 y purga
±1 sobre 0,5; con etiquetas al azar, por bloque queda en 0,5.

## E8. Escalera en seis clases

Mismas cuatro particiones de E1 con las seis clases de la tesis, sobre las cuatro
sesiones completas (sin S2, como en la tesis). Métricas: exactitud, F1 macro, línea base
de la clase mayoritaria con su F1 macro y exactitud condicionada al gesto. Modelos de
E1; el refuerzo por gradiente con ponderación de clases balanceada.

## E9. CNN-LSTM de la tesis

Arquitectura de referencia de la tesis (31.142 parámetros con 15 entradas) y su
configuración de entrenamiento, sin ajustar nada. Secuencias de 20 pasos construidas como
en 02_construir_secuencias.py. Particiones: aleatoria por secuencias (una división
80/20), sesión excluida (4 pliegues) y carga excluida (3 pliegues, para comprobar que se
reproduce el resultado de la tesis). Configuraciones: fusión (15) y sEMG sola (8).
Si el tiempo de cómputo no alcanza, se recorta primero la configuración de sEMG sola y
se declara.

## E10. Réplica en datos ajenos

Conjunto público "EMG elbow dataset" (Toro-Ossaba y Tejada, Zenodo, 2023,
DOI 10.5281/zenodo.7946782, CC BY 4.0): diez sujetos, sEMG de bíceps y tríceps, flexión
y extensión de codo con tres cargas (0, 3 y 5 lb) y dos registros independientes por
carga. Tarea: clasificar la carga por ventana, por sujeto. Particiones: aleatoria por
ventanas frente a agrupada por registro (entrenar con un registro de cada carga y probar
con el otro, en ambos sentidos). Mismas características, ventana, paso y modelos.
Controles: barrido de solapamiento con el diseño de E6 y permutación de etiquetas entre
registros. Unidad de inferencia: el sujeto (n = 10), con prueba de Wilcoxon pareada.
Predicción: la partición aleatoria supera a la agrupada por registro en los diez sujetos.

---

# Nota de auditoría (25 de noviembre de 2026, corregida el mismo día)

E10 se corrió primero con fs = 1024 Hz. A3 dedujo de los datos una fs de 760–800 Hz usando la convención de
scipy para el corte de un Chebyshev (borde de rizado), y E10 se repitió con 780,2 Hz. El artículo asociado al
conjunto (Toro-Ossaba et al., 2024) declara 1024 Hz, y A3b muestra que los datos son compatibles con ese valor
cuando el corte de 10 Hz se lee como punto de −3 dB (fs = 1026 Hz) y con el ritmo de unos 2 s por ciclo.
La corrida principal vuelve a ser la de 1024 Hz; la de 780,2 Hz queda como sensibilidad. A4 (bloques vecinos)
se añadió después de ver los resultados y es exploratorio.
