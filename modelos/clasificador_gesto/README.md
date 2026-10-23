# Etapa 2 · Clasificador de gesto

## 03_entrenar.py

Entrena nueve modelos: las tres configuraciones del estudio de ablación por cada
uno de los tres pliegues. El orden es deliberado, C primero, para que el
resultado principal esté disponible antes que la ablación.

| Configuración | Entrada |
|---|---|
| C, fusión multimodal | sEMG más estado del exoesqueleto |
| A, solo sEMG | únicamente las características mioeléctricas |
| B, solo exoesqueleto | únicamente las variables mecánicas |

El script comprueba el número de parámetros entrenables y se detiene si no
coincide con el esperado de la configuración elegida: 31.142 para ψ ∈ ℝ¹⁵ y
31.718 para ψ ∈ ℝ²¹.

**Arquitectura.** Conv1D de 32 filtros con núcleo 3, Conv1D de 64, submuestreo
por máximo con factor 2, LSTM de 48 unidades que devuelve solo su estado final,
densa de 32 y salida Softmax de seis clases. Abandono de 0,2 tras la primera
convolución y de 0,3 tras la segunda y tras la recurrente.

**Entrenamiento.** Adam con tasa inicial 10⁻³, entropía cruzada categórica
dispersa con clases ponderadas de forma inversa a su frecuencia, lotes de 512,
semilla fija y parada temprana a las 15 épocas sin mejora sobre un techo de 100.

**Cómo leer el resultado.** La exactitud se compara con la línea base de la clase
mayoritaria, 56,3 %, y no con el azar entre seis clases, 16,7 %. Ninguna
configuración supera la primera.

## 04_entrenar_mejorado.py

Variantes evaluadas para recuperar generalización: cabeza auxiliar de intención,
pérdida focal, jitter en la amplitud, CORAL y readaptación de las estadísticas de
normalización por lotes.

CORAL y la readaptación usan datos de entrada de la condición excluida sin sus
etiquetas. Es adaptación de dominio no supervisada, no fuga de etiquetas, y por
eso aparecen marcadas aparte en la figura 4.9. Ninguna de las dos es la variante
que el trabajo reporta como mejor.

## 05_pruebas_mejora.py

Compara las variantes entre sí y con la configuración base. Escribe
`resumen_pruebas_mejora.json`.

La mejora observada con la cabeza auxiliar es compatible con un efecto
regularizador, pero el diseño no permite afirmarlo como causa.

Tesis: apartados 3.6.5, 4.3 a 4.6. Figuras 4.4, 4.5, 4.7 y 4.9.
