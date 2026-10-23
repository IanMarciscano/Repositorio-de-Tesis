# Resultados

Solo los resúmenes en JSON. Los `.npz` de predicciones y los `.keras` de los
modelos no se versionan: son 374 MB y se regeneran ejecutando el pipeline.

| Carpeta | Configuración | Parámetros de C | Exactitud de C |
|---|---|---|---|
| `R15/` | ψ ∈ ℝ¹⁵, cuatro descriptores de amplitud por canal | 31.142 | 33,9 % |
| `R21/` | ψ ∈ ℝ²¹, con los tres espectrales añadidos | 31.718 | 27,6 % |

**R15 es la configuración principal de la tesis.** R21 es la variante que
aparece en la figura 4.9 como «C con psi en R21».

En el equipo del autor existen además `resultados - copia` y `resultados codigo
mejorado`, que son copias idénticas de R21 y R15 respectivamente, y
`resultados_paso6_cadena_supervisora_DESCARTADO_fallo_alineacion_temporal`, que
se descartó por el fallo que su propio nombre indica. Ninguna se versiona.

Contra la línea base de la clase mayoritaria, 56,3 %, ninguna configuración la
alcanza. El azar entre seis clases es 16,7 %.
