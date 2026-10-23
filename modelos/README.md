# Modelos

La solución final es híbrida y conviene nombrarla sin ambigüedad.

| Etapa | Entrada | Modelo | Qué sostiene |
|---|---|---|---|
| Puerta de intención | 14 características, solo sEMG, sobre la ventana de transición | bosque aleatorio de 300 árboles | el análisis de anticipación |
| Clasificador de gesto | ψ multimodal, sEMG más estado del exoesqueleto | CNN-LSTM | la identificación del gesto y la capa de control |

Dos límites que el propio trabajo declara y que conviene no perder de vista al
leer el código:

- **La puerta no gobierna la máquina de estados.** La señal que gobierna el
  supervisor sale de la cabeza auxiliar binaria del clasificador de gesto, no de
  la puerta. La razón es el dominio sobre el que cada salida es válida: la puerta
  se entrenó solo con la ventana de transición previa al inicio y con reposo
  anterior.
- **Las dos etapas se evaluaron por separado.** El encadenamiento en cascada no
  forma parte de lo verificado.

Tesis: apartados 3.6.2, 3.6.3 y 4.7.
