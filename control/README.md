# Capa de control

`06_cadena_supervisora.py` ejecuta la cadena supervisora completa sobre los
registros y escribe `resumen_cadena.json`.

**Lazo abierto.** La cadena se ejecuta entera y genera comandos, pero sin mover
el actuador con el usuario puesto. Ninguna de las cifras de esta carpeta procede
de actuación física sobre una persona.

**Máquina de estados de doble umbral.** Un umbral único sobre una señal ruidosa
produce parpadeo del estado. Se usan dos umbrales y una exigencia de
persistencia: para arrancar, la probabilidad debe superar el umbral alto durante
varias ventanas seguidas; para detenerse, caer por debajo del bajo y mantenerse.
El efecto sobre los falsos positivos está medido en
`analisis/14_falsos_positivos.py`.

**Generador de trayectoria de mínima sacudida.** Polinomio de quinto orden, que
es el grado mínimo que permite fijar posición, velocidad y aceleración en ambos
extremos. La trayectoria arranca y termina en reposo, sin discontinuidades de
aceleración, y se recorta a los límites del actuador: 15 °/s en el hombro y
30 °/s en el codo.

**Ley de asistencia según necesidad.** El coeficiente combina el error de
seguimiento y la fuerza de interacción, y se modula por la probabilidad de
intención. La ponderación adoptada, 0,2 para el error y 0,8 para la fuerza, **es
un resultado y no una elección previa**: procede de la medida de selectividad del
apartado 4.18, donde la fuerza distingue gesto de reposo con razón 1,48 frente al
1,08 del error. Los umbrales de necesidad asistiva, 0,10 y 0,60, sí son elección
del diseñador y se declaran como tal.

**Pasarela de seguridad.** Su lógica se verifica aparte, en
`analisis/15_verificacion_pasarela.py`, sobre los registros y sin inyección de
fallas. No está integrada físicamente.

Tesis: apartados 3.9 y 3.10. Figuras 4.13 y 4.15.
