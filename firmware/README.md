# Firmware

## emg_teensy · ProgPresEMg02.02.ino

Teensy 4.0 de la vía mioeléctrica. Cuatro canales analógicos a 12 bits
muestreados por temporizador a 1 kHz, sin promediado, y enviados por USB a
2 Mbaud.

Mapeo de pines:

| Pin | Canal | Señal |
|---|---|---|
| 17 | A3 | RECT bíceps |
| 16 | A2 | RECT tríceps |
| 15 | A1 | RAW bíceps |
| 14 | A0 | RAW tríceps |

El análisis usa los canales RAW: `EMG_ch2_raw_bic` y `EMG_ch3_raw_tri`.

**Formato de trama.** Buffer circular de 2048 muestras servido fuera de la
interrupción. Cada muestra sale como una estructura empaquetada de 19 bytes:

    0xA5  0x5A  0x01     cabecera y tipo        3 bytes
    t_us                 microsegundos desde START   4 bytes, little endian
    seq                  numero de secuencia         4 bytes
    c0 c1 c2 c3          cuatro canales, 12 bits     8 bytes

Comandos por línea: `START` reinicia el reloj y el contador y empieza a emitir;
`STOP` detiene la emisión. El firmware tolera el `\r` que añade Windows.

El número de secuencia permite detectar muestras perdidas, y el buffer descarta
en silencio si el consumidor no lee a tiempo: conviene vigilar los saltos de
`seq` en la adquisición.

## exoesqueleto_teensy · PruebaExo2.1.ino

Control del exoesqueleto a 200 Hz. Control PD de fuerza a velocidad en ambas
articulaciones, media móvil exponencial sobre las celdas de carga, actuador
AK70-10 del hombro por bus CAN, servomotor del codo por ancho de pulso, y
registro CSV a 20 Hz. Limita el hombro a −40° / +30°.

**Este firmware pertenece a un trabajo de tesis distinto** y se incluye como
documentación de la plataforma sobre la que se hizo el aporte