# Etapa 1 · Puerta de intención

`05_puerta_intencion.py` entrena y evalúa la primera etapa. Escribe
`puerta_intencion.json` y `puerta_intencion.npz`.

**Por qué las variables del exoesqueleto quedan fuera.** Las siete variables
mecánicas describen el estado del exoesqueleto en el instante de la ventana y,
por tanto, reflejan movimiento ya iniciado. Son legítimas para identificar de qué
gesto se trata, pero no pueden sostener una afirmación de anticipación: un
clasificador que las emplee para decidir si el usuario va a moverse estaría
observando el movimiento, no prediciéndolo. Su exclusión en esta etapa es
condición necesaria para que la afirmación de anticipación sea válida.

**Definición de la ventana.** El instante de inicio mecánico es el primer cruce
sostenido durante 50 ms del umbral de 5 °/s en la velocidad de la articulación
principal del gesto. La clase positiva son las ventanas cuyo instante final cae
en los 300 ms previos a ese instante; la negativa, ventanas de igual duración
tomadas al menos un segundo antes del cambio de etiqueta.

**Por qué dos clasificadores.** La unidad estadística son los episodios de
movimiento, no las ventanas, y hay 61 episodios independientes. Con tan pocas
unidades se entrenan y se reportan dos modelos, un bosque aleatorio y una red
convolucional-recurrente reducida, en vez de suponer cuál generaliza mejor. El
resultado, en este conjunto, es 0,634 frente a 0,582.

**Validación.** Agrupada por episodio: ninguna ventana de un mismo episodio
aparece a la vez en entrenamiento y en prueba.

**Métrica.** Área bajo la curva característica operativa, independiente del
umbral, de modo que el punto de operación de la máquina de estados pueda fijarse
después con criterios de seguridad. El script reporta además el barrido del
umbral.

Tesis: apartados 3.6.6 y 4.7. Figura 4.10.
