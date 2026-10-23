# Datos

Cinco sesiones de adquisición, 1.973.531 muestras y 32,9 minutos útiles.

| Archivo | Contenido |
|---|---|
| `emg_<sesion>.csv` | señal mioeléctrica cruda y telemetría sincronizada, 1 kHz |
| `exo_<sesion>.csv` | telemetría del exoesqueleto |
| `feat_<sesion>.csv` | características por ventana, primera versión del pipeline |

Sesiones y condición de carga:

| Sesión | Carga |
|---|---|
| 20260806_182858 | 0,0 kg · IMU congelada, 43 filas con NaN interpoladas |
| 20260806_193017 | 0,0 kg |
| 20260807_150646 | 0,0 kg |
| 20260807_170550 | 0,85 kg |
| 20260811_122212 | 0,5 kg · calibración ejecutada con la carga puesta |

El pipeline solo necesita los `emg_*.csv`: el paso 1 recalcula las
características desde el EMG crudo y las variables del exoesqueleto ya vienen
dentro del propio archivo.

**Advertencia.** La señal mioeléctrica es un dato personal de carácter
biométrico y queda comprendida en la Ley 81 de 2019 de Panamá. El apartado 3.11
de la tesis declara que los registros se conservan bajo custodia del equipo de
investigación y se emplean exclusivamente para los fines del trabajo. Este
repositorio es privado por esa razón. Si alguna vez se hiciera público, estos
archivos tendrían que salir primero.
