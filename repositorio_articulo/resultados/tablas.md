
**Tabla R0. Réplica de la Tabla 4.21 de la tesis (fusión de 15 componentes, refuerzo por gradiente, dentro de sesión)**

| Sesión | Bloques | Ventanas | Aleatoria | Por bloque (una asignación) |
|---|---|---|---|---|
| S1 | 38 | 23776 | 1,0000 | 0,496 |
| S2 | 31 | 19081 | 1,0000 | 0,419 |
| S3 | 45 | 36384 | 0,9999 | 0,364 |
| S4 | 43 | 34506 | 0,9999 | 0,308 |
| S5 | 45 | 36261 | 0,9998 | 0,524 |

**Tabla R1. Escalera de particiones, tarea reposo/gesto, sesiones mezcladas. AUC [IC 95 % por remuestreo de bloques]. 150.008 ventanas, 202 bloques.**

| Características | Modelo | A. Aleatoria | B. Por bloque | C. Sesión excluida | D. Carga excluida |
|---|---|---|---|---|---|
| sEMG, amplitud (8) | Refuerzo por gradiente | 0,743 [0,709; 0,774] | 0,495 [0,460; 0,546] | 0,561 [0,512; 0,610] | 0,549 [0,486; 0,616] |
| sEMG, amplitud (8) | Discriminante lineal | 0,589 [0,534; 0,640] | 0,528 [0,474; 0,589] | 0,547 [0,485; 0,611] | 0,553 [0,488; 0,621] |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | 0,804 [0,778; 0,827] | 0,501 [0,459; 0,546] | 0,568 [0,512; 0,626] | 0,542 [0,490; 0,598] |
| sEMG, amplitud y espectro (14) | Discriminante lineal | 0,624 [0,580; 0,671] | 0,565 [0,516; 0,614] | 0,569 [0,512; 0,628] | 0,553 [0,490; 0,620] |
| Mecánicas (7) | Refuerzo por gradiente | 0,977 [0,971; 0,983] | 0,556 [0,529; 0,606] | 0,612 [0,552; 0,672] | 0,594 [0,535; 0,658] |
| Mecánicas (7) | Discriminante lineal | 0,573 [0,507; 0,649] | 0,507 [0,439; 0,576] | 0,547 [0,480; 0,622] | 0,547 [0,482; 0,619] |
| Fusión (15) | Refuerzo por gradiente | 0,981 [0,976; 0,985] | 0,532 [0,508; 0,595] | 0,625 [0,564; 0,684] | 0,591 [0,532; 0,656] |
| Fusión (15) | Discriminante lineal | 0,613 [0,558; 0,671] | 0,528 [0,467; 0,584] | 0,553 [0,491; 0,623] | 0,567 [0,503; 0,630] |

**Tabla R2. Escalera de particiones: AUC evaluada dentro de cada sesión de prueba y exactitud balanceada agregada**

| Características | Modelo | Esquema | S1 | S2 | S3 | S4 | S5 | Exactitud balanceada |
|---|---|---|---|---|---|---|---|---|
| sEMG, amplitud (8) | Refuerzo por gradiente | A | 0,745 | 0,711 | 0,722 | 0,806 | 0,702 | 0,636 |
| sEMG, amplitud (8) | Refuerzo por gradiente | B | 0,607 | 0,512 | 0,522 | 0,405 | 0,469 | 0,491 |
| sEMG, amplitud (8) | Refuerzo por gradiente | C | 0,633 | 0,551 | 0,585 | 0,539 | 0,520 | 0,526 |
| sEMG, amplitud (8) | Refuerzo por gradiente | D | 0,622 | 0,552 | 0,511 | 0,539 | 0,520 | 0,535 |
| sEMG, amplitud (8) | Discriminante lineal | A | 0,680 | 0,578 | 0,581 | 0,561 | 0,547 | 0,550 |
| sEMG, amplitud (8) | Discriminante lineal | B | 0,580 | 0,564 | 0,486 | 0,519 | 0,471 | 0,522 |
| sEMG, amplitud (8) | Discriminante lineal | C | 0,599 | 0,557 | 0,526 | 0,561 | 0,525 | 0,525 |
| sEMG, amplitud (8) | Discriminante lineal | D | 0,662 | 0,557 | 0,514 | 0,561 | 0,525 | 0,544 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | A | 0,844 | 0,713 | 0,846 | 0,830 | 0,743 | 0,701 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | B | 0,587 | 0,473 | 0,575 | 0,379 | 0,481 | 0,499 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | C | 0,651 | 0,539 | 0,599 | 0,545 | 0,551 | 0,548 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | D | 0,621 | 0,514 | 0,514 | 0,545 | 0,551 | 0,534 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | A | 0,682 | 0,583 | 0,677 | 0,572 | 0,583 | 0,584 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | B | 0,620 | 0,549 | 0,573 | 0,535 | 0,526 | 0,548 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | C | 0,626 | 0,559 | 0,588 | 0,557 | 0,526 | 0,542 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | D | 0,679 | 0,528 | 0,534 | 0,557 | 0,526 | 0,536 |
| Mecánicas (7) | Refuerzo por gradiente | A | 0,984 | 0,982 | 0,978 | 0,975 | 0,977 | 0,902 |
| Mecánicas (7) | Refuerzo por gradiente | B | 0,660 | 0,547 | 0,590 | 0,442 | 0,605 | 0,528 |
| Mecánicas (7) | Refuerzo por gradiente | C | 0,639 | 0,763 | 0,678 | 0,657 | 0,628 | 0,596 |
| Mecánicas (7) | Refuerzo por gradiente | D | 0,571 | 0,631 | 0,636 | 0,657 | 0,628 | 0,571 |
| Mecánicas (7) | Discriminante lineal | A | 0,615 | 0,543 | 0,564 | 0,537 | 0,617 | 0,552 |
| Mecánicas (7) | Discriminante lineal | B | 0,565 | 0,504 | 0,430 | 0,485 | 0,536 | 0,531 |
| Mecánicas (7) | Discriminante lineal | C | 0,584 | 0,546 | 0,519 | 0,532 | 0,564 | 0,537 |
| Mecánicas (7) | Discriminante lineal | D | 0,577 | 0,557 | 0,523 | 0,532 | 0,564 | 0,541 |
| Fusión (15) | Refuerzo por gradiente | A | 0,986 | 0,979 | 0,987 | 0,978 | 0,979 | 0,915 |
| Fusión (15) | Refuerzo por gradiente | B | 0,663 | 0,547 | 0,562 | 0,389 | 0,630 | 0,522 |
| Fusión (15) | Refuerzo por gradiente | C | 0,596 | 0,730 | 0,690 | 0,657 | 0,642 | 0,595 |
| Fusión (15) | Refuerzo por gradiente | D | 0,587 | 0,658 | 0,620 | 0,657 | 0,642 | 0,585 |
| Fusión (15) | Discriminante lineal | A | 0,674 | 0,608 | 0,614 | 0,566 | 0,626 | 0,581 |
| Fusión (15) | Discriminante lineal | B | 0,606 | 0,579 | 0,472 | 0,486 | 0,520 | 0,538 |
| Fusión (15) | Discriminante lineal | C | 0,564 | 0,582 | 0,541 | 0,562 | 0,537 | 0,548 |
| Fusión (15) | Discriminante lineal | D | 0,651 | 0,559 | 0,536 | 0,562 | 0,537 | 0,558 |

**Tabla R3. Dentro de sesión: aleatoria frente a agrupada por bloque (media de las cinco sesiones; la agrupada, sobre 5 asignaciones de bloques a pliegues)**

| Características | Modelo | Aleatoria | Rango entre sesiones | Por bloque | Rango entre sesiones | DE entre asignaciones |
|---|---|---|---|---|---|---|
| sEMG, amplitud (8) | Refuerzo por gradiente | 0,8297 | 0,776 a 0,870 | 0,461 | 0,437 a 0,483 | 0,016 |
| sEMG, amplitud (8) | Discriminante lineal | 0,6151 | 0,566 a 0,677 | 0,466 | 0,369 a 0,536 | 0,033 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | 0,9167 | 0,876 a 0,957 | 0,469 | 0,396 a 0,515 | 0,016 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | 0,6560 | 0,591 a 0,722 | 0,498 | 0,393 a 0,567 | 0,023 |
| Mecánicas (7) | Refuerzo por gradiente | 0,9999 | 1,000 a 1,000 | 0,455 | 0,320 a 0,583 | 0,033 |
| Mecánicas (7) | Discriminante lineal | 0,6316 | 0,561 a 0,690 | 0,454 | 0,315 a 0,585 | 0,025 |
| Fusión (15) | Refuerzo por gradiente | 0,9999 | 1,000 a 1,000 | 0,449 | 0,309 a 0,563 | 0,032 |
| Fusión (15) | Discriminante lineal | 0,6677 | 0,582 a 0,718 | 0,457 | 0,347 a 0,517 | 0,028 |

**Tabla R4. Barrido del paso entre ventanas, dentro de sesión (AUC media de las cinco sesiones)**

| Características | Modelo | Partición | 10 ms (96,7 %) | 30 ms (90,0 %) | 50 ms (83,3 %) | 100 ms (66,7 %) | 150 ms (50,0 %) | 300 ms (0,0 %) |
|---|---|---|---|---|---|---|---|---|
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | Aleatoria | 0,917 | 0,866 | 0,823 | 0,750 | 0,702 | 0,647 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | Por bloque | 0,476 | 0,482 | 0,482 | 0,467 | 0,465 | 0,501 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | Aleatoria | 0,656 | 0,654 | 0,652 | 0,646 | 0,642 | 0,632 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | Por bloque | 0,503 | 0,523 | 0,510 | 0,507 | 0,500 | 0,528 |
| Mecánicas (7) | Refuerzo por gradiente | Aleatoria | 1,000 | 0,999 | 0,998 | 0,993 | 0,984 | 0,945 |
| Mecánicas (7) | Refuerzo por gradiente | Por bloque | 0,462 | 0,445 | 0,460 | 0,462 | 0,465 | 0,483 |
| Fusión (15) | Refuerzo por gradiente | Aleatoria | 1,000 | 1,000 | 0,998 | 0,992 | 0,979 | 0,930 |
| Fusión (15) | Refuerzo por gradiente | Por bloque | 0,458 | 0,441 | 0,455 | 0,459 | 0,465 | 0,483 |

**Tabla R5. Controles dentro de sesión (AUC media de las cinco sesiones). Etiquetas permutadas a nivel de bloque; agrupada por bloque en 10 pliegues con purga de los bloques adyacentes al de prueba; división temporal dentro del bloque**

| Características | Modelo | Aleatoria, etiquetas permutadas | Por bloque, etiquetas permutadas | Por bloque, sin purga | Purga ±1 | Purga ±2 | 60/40 dentro del bloque |
|---|---|---|---|---|---|---|---|
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | 0,920 | 0,487 | 0,474 | 0,615 | 0,543 | 0,575 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | — | — | 0,517 | 0,627 | 0,543 | 0,568 |
| Fusión (15) | Refuerzo por gradiente | 1,000 | 0,490 | 0,463 | 0,719 | 0,683 | 0,465 |

Referencia del vecino temporal (asignar la etiqueta del bloque de entrenamiento más cercano en el tiempo): AUC media 0,042, rango 0,000 a 0,077.


**Tabla R6a. Transferencia entre sesiones, normalización de la tesis: entrenar en una sesión y probar en otra (AUC)**

| Características | Modelo | Diagonal (por bloque, intra) | Fuera de la diagonal | Rango | Entre sesiones de 0 kg | Mismo día |
|---|---|---|---|---|---|---|
| sEMG, amplitud (8) | Refuerzo por gradiente | 0,469 | 0,534 | 0,483 a 0,616 | 0,545 | 0,527 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | 0,473 | 0,542 | 0,457 a 0,670 | 0,557 | 0,540 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | 0,493 | 0,545 | 0,480 a 0,661 | 0,585 | 0,541 |
| Mecánicas (7) | Refuerzo por gradiente | 0,463 | 0,603 | 0,515 a 0,728 | 0,645 | 0,653 |
| Fusión (15) | Refuerzo por gradiente | 0,461 | 0,607 | 0,497 a 0,742 | 0,648 | 0,647 |

**Tabla R6b. Matriz completa, sEMG, amplitud y espectro (14), refuerzo por gradiente, normalización de la tesis (filas: entrenamiento; columnas: prueba)**

|  | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| S1 | 0,515 | 0,568 | 0,574 | 0,543 | 0,585 |
| S2 | 0,549 | 0,418 | 0,525 | 0,551 | 0,555 |
| S3 | 0,670 | 0,457 | 0,532 | 0,518 | 0,508 |
| S4 | 0,543 | 0,487 | 0,526 | 0,417 | 0,497 |
| S5 | 0,587 | 0,556 | 0,513 | 0,532 | 0,481 |

**Tabla R6c. Transferencia entre sesiones, normalización acumulada causal: entrenar en una sesión y probar en otra (AUC)**

| Características | Modelo | Diagonal (por bloque, intra) | Fuera de la diagonal | Rango | Entre sesiones de 0 kg | Mismo día |
|---|---|---|---|---|---|---|
| sEMG, amplitud (8) | Refuerzo por gradiente | 0,443 | 0,551 | 0,475 a 0,646 | 0,554 | 0,553 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | 0,431 | 0,548 | 0,467 a 0,617 | 0,561 | 0,554 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | 0,494 | 0,558 | 0,488 a 0,629 | 0,559 | 0,548 |
| Mecánicas (7) | Refuerzo por gradiente | 0,463 | 0,603 | 0,515 a 0,728 | 0,645 | 0,653 |
| Fusión (15) | Refuerzo por gradiente | 0,449 | 0,609 | 0,512 a 0,755 | 0,644 | 0,658 |

**Tabla R6d. Matriz completa, sEMG, amplitud y espectro (14), refuerzo por gradiente, normalización acumulada causal (filas: entrenamiento; columnas: prueba)**

|  | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| S1 | 0,375 | 0,507 | 0,582 | 0,467 | 0,530 |
| S2 | 0,555 | 0,383 | 0,617 | 0,546 | 0,537 |
| S3 | 0,526 | 0,579 | 0,518 | 0,553 | 0,577 |
| S4 | 0,531 | 0,516 | 0,600 | 0,410 | 0,509 |
| S5 | 0,587 | 0,516 | 0,610 | 0,520 | 0,468 |

**Tabla R7. Puerta de intención (61 episodios, 3660 ventanas, bosque aleatorio). Trayectoria: línea base B, de −1000 a −500 ms**

| Validación | AUC por pliegue | AUC agregada | IC 95 % (episodios) | AUC por sesión S1 / S2 / S3 / S4 / S5 | Inicio de divergencia (ms) | Masa | p |
|---|---|---|---|---|---|---|---|
| Agrupada por episodio | 0,632 ± 0,051 | 0,634 | 0,565 a 0,705 | 0,668 / 0,545 / 0,693 / 0,566 / 0,676 | -110 | 307,1 | 0,0005 |
| Sesión excluida | 0,598 ± 0,103 | 0,587 | 0,522 a 0,655 | 0,671 / 0,661 / 0,424 / 0,540 / 0,696 | +30 | 184,1 | 0,0010 |
| Por episodio, tipificada por sesión | 0,578 ± 0,064 | 0,576 | 0,507 a 0,644 | 0,556 / 0,532 / 0,617 / 0,557 / 0,581 | -180 | 259,1 | 0,0005 |
| Sesión excluida, tipificada por sesión | 0,527 ± 0,074 | 0,512 | 0,443 a 0,583 | 0,605 / 0,607 / 0,409 / 0,507 / 0,508 | +210 | 63,6 | 0,0075 |

**Tabla R8. Sensibilidad de la divergencia a la línea base (A: −2000 a −1000 ms; B: −1000 a −500 ms; C: −900 a −400 ms)**

| Validación | Línea base | Inicio (ms) | Masa | p | Nulo p95 | Nulo máx. |
|---|---|---|---|---|---|---|
| Agrupada por episodio | A | -370 | 459,8 | 0,0005 | 33,6 | 222,3 |
| Agrupada por episodio | B | -110 | 307,1 | 0,0005 | 36,2 | 176,3 |
| Agrupada por episodio | C | -110 | 305,6 | 0,0005 | 35,8 | 190,0 |
| Sesión excluida | A | -250 | 317,4 | 0,0005 | 36,0 | 250,2 |
| Sesión excluida | B | +30 | 184,1 | 0,0010 | 36,9 | 197,7 |
| Sesión excluida | C | +10 | 192,3 | 0,0005 | 35,7 | 164,4 |

**Tabla R9. Curva de calibración de sesión: AUC sobre los episodios de prueba de la sesión excluida al devolverle k episodios etiquetados (20 sorteos; media ± DE entre sorteos)**

| Sesión | Episodios | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|---|
| S1 | 8 | 0,684 ± 0,110 | 0,696 ± 0,117 | 0,691 ± 0,111 | 0,699 ± 0,114 | 0,707 ± 0,112 |
| S2 | 9 | 0,673 ± 0,060 | 0,668 ± 0,054 | 0,646 ± 0,063 | 0,624 ± 0,056 | 0,602 ± 0,060 |
| S3 | 15 | 0,428 ± 0,039 | 0,470 ± 0,067 | 0,492 ± 0,075 | 0,515 ± 0,078 | 0,540 ± 0,103 |
| S4 | 15 | 0,539 ± 0,046 | 0,555 ± 0,065 | 0,568 ± 0,054 | 0,573 ± 0,064 | 0,573 ± 0,066 |
| S5 | 14 | 0,708 ± 0,026 | 0,708 ± 0,039 | 0,714 ± 0,045 | 0,721 ± 0,040 | 0,719 ± 0,040 |
| Media |  | 0,606 | 0,619 | 0,622 | 0,626 | 0,628 |
| Solo k episodios (media) |  | — | 0,520 | 0,530 | 0,531 | 0,539 |

**Tabla R8b. Incertidumbre del instante de divergencia por remuestreo pareado de episodios (2000 réplicas). Inicio en ms [IC 95 %]; proporción de réplicas con inicio anterior al movimiento; diferencia sesión − episodio (mediana [IC 95 %]) y proporción de réplicas en que es positiva**

| Línea base | Por episodio | Antes de 0 | Sesión excluida | Antes de 0 | Diferencia (ms) | Diferencia > 0 |
|---|---|---|---|---|---|---|
| A | -370 [-580; +0] | 96 % | -250 [-460; +240] | 74 % | +40 [-270; +520] | 62 % |
| B | -110 [-370; +270] | 77 % | +30 [-270; +390] | 31 % | +110 [-290; +450] | 77 % |
| C | -110 [-370; +270] | 78 % | +10 [-340; +360] | 35 % | +90 [-270; +440] | 73 % |

**Tabla R8c. Diferencia de AUC de la puerta entre validación por episodio y sesión excluida, con remuestreo pareado de episodios**

| Episodios evaluados | AUC por episodio | AUC sesión excluida | Diferencia | IC 95 % | Réplicas con diferencia > 0 |
|---|---|---|---|---|---|
| Los 61 episodios | 0,634 | 0,587 | +0,048 | -0,018 a +0,120 | 92 % |
| Solo S1 | 0,668 | 0,671 | -0,003 | -0,057 a +0,066 | 46 % |
| Solo S2 | 0,545 | 0,661 | -0,116 | -0,225 a -0,026 | 0 % |
| Solo S3 | 0,693 | 0,424 | +0,269 | +0,128 a +0,438 | 100 % |
| Solo S4 | 0,566 | 0,540 | +0,026 | -0,145 a +0,189 | 62 % |
| Solo S5 | 0,676 | 0,696 | -0,020 | -0,125 a +0,090 | 35 % |
| Sin S1 | 0,626 | 0,573 | +0,054 | -0,024 a +0,138 | 91 % |
| Sin S2 | 0,650 | 0,571 | +0,078 | +0,008 a +0,159 | 99 % |
| Sin S3 | 0,617 | 0,639 | -0,022 | -0,088 a +0,046 | 25 % |
| Sin S4 | 0,656 | 0,598 | +0,058 | -0,012 a +0,130 | 94 % |
| Sin S5 | 0,622 | 0,556 | +0,067 | -0,011 a +0,155 | 95 % |

**Tabla R11. Sesión excluida con y sin la sesión de IMU congelada (S1) en el entrenamiento. AUC con S1 / sin S1, refuerzo por gradiente**

| Características | S2 | S3 | S4 | S5 | Media |
|---|---|---|---|---|---|
| Mecánicas (7) | 0,763 / 0,733 | 0,678 / 0,707 | 0,657 / 0,663 | 0,628 / 0,607 | 0,681 / 0,678 |
| Fusión (15) | 0,730 / 0,703 | 0,690 / 0,657 | 0,657 / 0,645 | 0,642 / 0,584 | 0,680 / 0,647 |

**Tabla R10. Efecto de la normalización de la sEMG (14 características) sobre cada peldaño (AUC)**

| Normalización | Modelo | A. Aleatoria | B. Por bloque | C. Sesión excluida [IC 95 %] | C, media por sesión | D. Carga excluida | Intra, aleatoria | Intra, por bloque |
|---|---|---|---|---|---|---|---|---|
| Calibración de sesión, recorte ±10 (tesis) | Refuerzo por gradiente | 0,804 | 0,501 | 0,568 [0,512; 0,626] | 0,577 | 0,542 | 0,917 | 0,469 |
| Calibración de sesión, recorte ±10 (tesis) | Discriminante lineal | 0,624 | 0,565 | 0,569 [0,512; 0,628] | 0,571 | 0,553 | 0,656 | 0,498 |
| Solo log1p | Refuerzo por gradiente | 0,786 | 0,514 | 0,568 [0,522; 0,619] | 0,564 | 0,578 | 0,912 | 0,468 |
| Solo log1p | Discriminante lineal | 0,618 | 0,548 | 0,573 [0,521; 0,631] | 0,584 | 0,556 | 0,657 | 0,490 |
| Acumulada causal | Refuerzo por gradiente | 0,805 | 0,494 | 0,570 [0,525; 0,615] | 0,567 | 0,540 | 0,935 | 0,430 |
| Acumulada causal | Discriminante lineal | 0,617 | 0,569 | 0,583 [0,537; 0,623] | 0,582 | 0,578 | 0,654 | 0,499 |
| Exponencial causal, 60 s | Refuerzo por gradiente | 0,810 | 0,505 | 0,577 [0,538; 0,618] | 0,579 | 0,556 | 0,936 | 0,433 |
| Exponencial causal, 60 s | Discriminante lineal | 0,600 | 0,535 | 0,547 [0,498; 0,592] | 0,549 | 0,554 | 0,652 | 0,477 |
| Mediana/IQR de sesión (no causal) | Refuerzo por gradiente | 0,798 | 0,506 | 0,530 [0,482; 0,573] | 0,533 | 0,524 | 0,927 | 0,461 |
| Mediana/IQR de sesión (no causal) | Discriminante lineal | 0,594 | 0,533 | 0,542 [0,489; 0,592] | 0,551 | 0,554 | 0,657 | 0,504 |

**Tabla R12. Solapamiento con el número de ventanas igualado (E6). Entrenamiento fijo, una ventana cada 600 ms por bloque; prueba desplazada. AUC media de las cinco sesiones y de seis rejillas de entrenamiento, según la fracción de muestras compartidas**

| Características | Modelo | 96,7 % | 90,0 % | 83,3 % | 66,7 % | 50,0 % | 0,0 % |
|---|---|---|---|---|---|---|---|
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | 0,955 | 0,858 | 0,794 | 0,711 | 0,680 | 0,650 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | 0,660 | 0,656 | 0,655 | 0,649 | 0,647 | 0,637 |
| Mecánicas (7) | Refuerzo por gradiente | 0,997 | 0,988 | 0,980 | 0,961 | 0,956 | 0,946 |
| Fusión (15) | Refuerzo por gradiente | 0,997 | 0,985 | 0,972 | 0,944 | 0,938 | 0,924 |

**Tabla R13. Simulación (E7): características sin información de clase, AUC verdadera 0,5. Media ± DE de 10 sesiones sintéticas**

| Etiquetas | Constante de la deriva | Modelo | Aleatoria | Por bloque | Purga ±1 | Purga ±2 |
|---|---|---|---|---|---|---|
| Alternadas | 2 s | Refuerzo por gradiente | 0,984 ± 0,001 | 0,343 ± 0,039 | 0,514 ± 0,035 | 0,484 ± 0,026 |
| Alternadas | 2 s | Discriminante lineal | 0,612 ± 0,018 | 0,295 ± 0,059 | 0,513 ± 0,047 | 0,475 ± 0,033 |
| Alternadas | 20 s | Refuerzo por gradiente | 0,976 ± 0,002 | 0,275 ± 0,050 | 0,670 ± 0,032 | 0,378 ± 0,046 |
| Alternadas | 20 s | Discriminante lineal | 0,585 ± 0,011 | 0,249 ± 0,047 | 0,686 ± 0,041 | 0,359 ± 0,044 |
| Alternadas | 200 s | Refuerzo por gradiente | 0,957 ± 0,002 | 0,442 ± 0,034 | 0,558 ± 0,033 | 0,461 ± 0,035 |
| Alternadas | 200 s | Discriminante lineal | 0,587 ± 0,012 | 0,399 ± 0,033 | 0,582 ± 0,045 | 0,435 ± 0,028 |
| Al azar por bloque | 2 s | Refuerzo por gradiente | 0,989 ± 0,001 | 0,496 ± 0,047 | 0,492 ± 0,033 | 0,485 ± 0,034 |
| Al azar por bloque | 2 s | Discriminante lineal | 0,692 ± 0,028 | 0,502 ± 0,055 | 0,499 ± 0,044 | 0,493 ± 0,056 |
| Al azar por bloque | 20 s | Refuerzo por gradiente | 0,984 ± 0,003 | 0,497 ± 0,052 | 0,475 ± 0,055 | 0,460 ± 0,039 |
| Al azar por bloque | 20 s | Discriminante lineal | 0,676 ± 0,031 | 0,478 ± 0,065 | 0,454 ± 0,057 | 0,445 ± 0,040 |
| Al azar por bloque | 200 s | Refuerzo por gradiente | 0,960 ± 0,004 | 0,486 ± 0,021 | 0,477 ± 0,027 | 0,476 ± 0,033 |
| Al azar por bloque | 200 s | Discriminante lineal | 0,619 ± 0,027 | 0,468 ± 0,054 | 0,463 ± 0,039 | 0,470 ± 0,040 |

**Tabla R14. Escalera en seis clases (E8), cuatro sesiones completas, 130927 ventanas y 171 bloques. Línea base de la clase mayoritaria: exactitud 0,561, F1 macro 0,120, condicionada al gesto 0,200. IC 95 % por remuestreo de bloques**

| Características | Modelo | Esquema | Exactitud | F1 macro | Condicionada al gesto |
|---|---|---|---|---|---|
| sEMG, amplitud (8) | Refuerzo por gradiente | A. Aleatoria | 0,470 [0,424; 0,520] | 0,479 [0,420; 0,525] | 0,678 |
| sEMG, amplitud (8) | Refuerzo por gradiente | B. Por bloque | 0,225 [0,198; 0,255] | 0,200 [0,165; 0,233] | 0,321 |
| sEMG, amplitud (8) | Refuerzo por gradiente | C. Sesión excluida | 0,298 [0,248; 0,348] | 0,167 [0,140; 0,195] | 0,204 |
| sEMG, amplitud (8) | Refuerzo por gradiente | D. Carga excluida | 0,279 [0,228; 0,327] | 0,152 [0,128; 0,174] | 0,196 |
| sEMG, amplitud (8) | Discriminante lineal | A. Aleatoria | 0,240 [0,196; 0,288] | 0,235 [0,189; 0,282] | 0,373 |
| sEMG, amplitud (8) | Discriminante lineal | B. Por bloque | 0,196 [0,157; 0,238] | 0,184 [0,139; 0,227] | 0,297 |
| sEMG, amplitud (8) | Discriminante lineal | C. Sesión excluida | 0,197 [0,155; 0,245] | 0,134 [0,104; 0,163] | 0,221 |
| sEMG, amplitud (8) | Discriminante lineal | D. Carga excluida | 0,200 [0,157; 0,246] | 0,142 [0,111; 0,174] | 0,236 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | A. Aleatoria | 0,612 [0,573; 0,649] | 0,611 [0,553; 0,648] | 0,834 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | B. Por bloque | 0,283 [0,254; 0,314] | 0,239 [0,204; 0,267] | 0,367 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | C. Sesión excluida | 0,339 [0,287; 0,390] | 0,184 [0,157; 0,210] | 0,210 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | D. Carga excluida | 0,413 [0,358; 0,467] | 0,173 [0,149; 0,197] | 0,218 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | A. Aleatoria | 0,313 [0,270; 0,359] | 0,281 [0,236; 0,326] | 0,411 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | B. Por bloque | 0,240 [0,204; 0,282] | 0,205 [0,165; 0,240] | 0,294 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | C. Sesión excluida | 0,237 [0,195; 0,282] | 0,172 [0,139; 0,203] | 0,234 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | D. Carga excluida | 0,198 [0,163; 0,238] | 0,144 [0,114; 0,175] | 0,197 |
| Mecánicas (7) | Refuerzo por gradiente | A. Aleatoria | 0,988 [0,985; 0,991] | 0,987 [0,980; 0,990] | 1,000 |
| Mecánicas (7) | Refuerzo por gradiente | B. Por bloque | 0,411 [0,364; 0,459] | 0,392 [0,342; 0,438] | 0,644 |
| Mecánicas (7) | Refuerzo por gradiente | C. Sesión excluida | 0,486 [0,424; 0,541] | 0,333 [0,280; 0,381] | 0,494 |
| Mecánicas (7) | Refuerzo por gradiente | D. Carga excluida | 0,487 [0,421; 0,544] | 0,322 [0,268; 0,371] | 0,450 |
| Mecánicas (7) | Discriminante lineal | A. Aleatoria | 0,476 [0,423; 0,534] | 0,475 [0,420; 0,521] | 0,704 |
| Mecánicas (7) | Discriminante lineal | B. Por bloque | 0,455 [0,408; 0,510] | 0,449 [0,396; 0,494] | 0,675 |
| Mecánicas (7) | Discriminante lineal | C. Sesión excluida | 0,387 [0,328; 0,450] | 0,368 [0,306; 0,422] | 0,557 |
| Mecánicas (7) | Discriminante lineal | D. Carga excluida | 0,358 [0,305; 0,418] | 0,349 [0,291; 0,401] | 0,534 |
| Fusión (15) | Refuerzo por gradiente | A. Aleatoria | 0,993 [0,991; 0,995] | 0,993 [0,989; 0,995] | 1,000 |
| Fusión (15) | Refuerzo por gradiente | B. Por bloque | 0,418 [0,369; 0,466] | 0,403 [0,348; 0,453] | 0,682 |
| Fusión (15) | Refuerzo por gradiente | C. Sesión excluida | 0,509 [0,445; 0,563] | 0,348 [0,291; 0,395] | 0,469 |
| Fusión (15) | Refuerzo por gradiente | D. Carga excluida | 0,500 [0,439; 0,559] | 0,327 [0,274; 0,376] | 0,472 |
| Fusión (15) | Discriminante lineal | A. Aleatoria | 0,483 [0,429; 0,534] | 0,495 [0,439; 0,540] | 0,747 |
| Fusión (15) | Discriminante lineal | B. Por bloque | 0,450 [0,401; 0,499] | 0,450 [0,400; 0,492] | 0,690 |
| Fusión (15) | Discriminante lineal | C. Sesión excluida | 0,385 [0,326; 0,447] | 0,352 [0,293; 0,402] | 0,517 |
| Fusión (15) | Discriminante lineal | D. Carga excluida | 0,375 [0,316; 0,435] | 0,359 [0,296; 0,409] | 0,523 |

**Tabla R15. CNN-LSTM de referencia de la tesis (E9) bajo cuatro particiones. Media ± DE entre pliegues (sesión y carga) o entre semillas de inicialización y de división (aleatoria y por bloque)**

| Entradas | Partición | Repeticiones | Exactitud | F1 macro | Condicionada al gesto | Base mayoritaria | F1 de la base | Épocas |
|---|---|---|---|---|---|---|---|---|
| Fusión (15) | Aleatoria por secuencias | 1 división × 5 semillas | 0,800 ± 0,023 | 0,805 ± 0,020 | 0,975 ± 0,007 | 0,561 | 0,120 | 69 / 94 / 91 / 63 / 53 |
| Fusión (15) | Agrupada por bloque | 1 división × 5 semillas | 0,398 ± 0,045 | 0,419 ± 0,039 | 0,661 ± 0,072 | 0,560 | 0,120 | 18 / 16 / 16 / 16 / 16 |
| Fusión (15) | Sesión excluida | 4 pliegues | 0,374 ± 0,111 | 0,281 ± 0,079 | 0,447 ± 0,087 | 0,558 | 0,119 | 17 / 18 / 17 / 16 |
| Fusión (15) | Carga excluida | 3 pliegues | 0,337 ± 0,122 | 0,271 ± 0,071 | 0,441 ± 0,019 | 0,563 | 0,120 | 17 / 16 / 17 |
| sEMG, amplitud (8) | Aleatoria por secuencias | 1 división × 5 semillas | 0,477 ± 0,016 | 0,504 ± 0,015 | 0,761 ± 0,014 | 0,561 | 0,120 | 85 / 100 / 100 / 97 / 100 |
| sEMG, amplitud (8) | Agrupada por bloque | 1 división × 5 semillas | 0,176 ± 0,038 | 0,175 ± 0,041 | 0,340 ± 0,061 | 0,560 | 0,120 | 16 / 17 / 16 / 17 / 17 |
| sEMG, amplitud (8) | Sesión excluida | 4 pliegues | 0,205 ± 0,102 | 0,148 ± 0,052 | 0,240 ± 0,098 | 0,558 | 0,119 | 16 / 16 / 16 / 16 |
| sEMG, amplitud (8) | Carga excluida | 3 pliegues | 0,205 ± 0,096 | 0,136 ± 0,025 | 0,232 ± 0,002 | 0,563 | 0,120 | 16 / 16 / 16 |
| Mecánicas (7) | Aleatoria por secuencias | 1 división × 1 semilla | 0,747 | 0,759 | 0,966 | 0,561 | 0,120 | 55 |
| Mecánicas (7) | Agrupada por bloque | 1 división × 1 semilla | 0,474 | 0,484 | 0,720 | 0,570 | 0,121 | 16 |
| Mecánicas (7) | Sesión excluida | 4 pliegues | 0,380 ± 0,057 | 0,323 ± 0,090 | 0,510 ± 0,154 | 0,558 | 0,119 | 16 / 17 / 19 / 16 |
| Mecánicas (7) | Carga excluida | 3 pliegues | 0,372 ± 0,062 | 0,313 ± 0,059 | 0,476 ± 0,075 | 0,563 | 0,120 | 16 / 16 / 19 |

**Tabla R16. Réplica en el conjunto público de codo (E10), fs = 1024,0 Hz: clasificación de la carga, AUC macro por sujeto. Aleatoria por ventanas frente a agrupada por registro; etiquetas reasignadas entre registros; solapamiento con N igualado**

| Sujeto | Ventanas | RG aleatoria | RG por registro | RG aleatoria, etiquetas reasignadas | DL aleatoria | DL por registro | RG, 96,7 % compartido | RG, 0 % compartido |
|---|---|---|---|---|---|---|---|---|
| 1 | 31572 | 0,995 | 0,926 | 0,969 | 0,918 | 0,905 | 0,996 | 0,923 |
| 2 | 12492 | 0,994 | 0,826 | 0,981 | 0,830 | 0,804 | 0,978 | 0,784 |
| 3 | 13582 | 0,992 | 0,844 | 0,979 | 0,823 | 0,787 | 0,985 | 0,784 |
| 4 | 12953 | 0,987 | 0,764 | 0,971 | 0,701 | 0,655 | 0,976 | 0,735 |
| 5 | 12495 | 0,994 | 0,786 | 0,987 | 0,788 | 0,731 | 0,977 | 0,794 |
| 6 | 15729 | 0,986 | 0,712 | 0,980 | 0,809 | 0,748 | 0,977 | 0,757 |
| 7 | 16988 | 0,985 | 0,769 | 0,972 | 0,802 | 0,760 | 0,979 | 0,785 |
| 8 | 11901 | 0,991 | 0,812 | 0,978 | 0,793 | 0,761 | 0,978 | 0,774 |
| 9 | 14163 | 0,984 | 0,715 | 0,973 | 0,793 | 0,707 | 0,969 | 0,761 |
| 10 | 14308 | 0,998 | 0,912 | 0,985 | 0,904 | 0,889 | 0,989 | 0,901 |
| Media |  | 0,991 | 0,807 | 0,978 | 0,816 | 0,775 | 0,980 | 0,800 |

Refuerzo por gradiente: aleatoria 0,991 ± 0,005; por registro 0,807 ± 0,073; diferencia mediana 0,194; aleatoria mayor en 10 de 10 sujetos; Wilcoxon unilateral p = 0,0010.

Discriminante lineal: aleatoria 0,816 ± 0,061; por registro 0,775 ± 0,077; diferencia mediana 0,039; aleatoria mayor en 10 de 10 sujetos; Wilcoxon unilateral p = 0,0010.

RG: refuerzo por gradiente. DL: discriminante lineal.


**Tabla R17. Sensibilidad por clase (diagonal de la matriz de confusión normalizada por fila), seis clases, refuerzo por gradiente (E8). Azar: 0,167**

| Configuración | Reposo | Flex. codo | Ext. codo | Flex. hombro | Ext. hombro | Levantamiento |
|---|---|---|---|---|---|---|
| sEMG (14), aleatoria | 0,460 | 0,843 | 0,869 | 0,781 | 0,731 | 0,809 |
| sEMG (14), sesión excluida | 0,512 | 0,154 | 0,092 | 0,097 | 0,145 | 0,098 |
| Fusión (15), aleatoria | 0,988 | 1,000 | 0,999 | 1,000 | 1,000 | 0,999 |
| Fusión (15), sesión excluida | 0,675 | 0,065 | 0,209 | 0,638 | 0,309 | 0,262 |

**Tabla R17a. Matriz de confusión normalizada por fila, sEMG (14), sesión excluida. Filas: clase real; columnas: predicha**

| Real \ Predicha | Reposo | Flex. codo | Ext. codo | Flex. hombro | Ext. hombro | Levantamiento |
|---|---|---|---|---|---|---|
| Reposo | 0,512 | 0,087 | 0,087 | 0,054 | 0,107 | 0,153 |
| Flex. codo | 0,460 | 0,154 | 0,078 | 0,012 | 0,116 | 0,180 |
| Ext. codo | 0,437 | 0,119 | 0,092 | 0,024 | 0,112 | 0,216 |
| Flex. hombro | 0,375 | 0,045 | 0,068 | 0,097 | 0,148 | 0,268 |
| Ext. hombro | 0,399 | 0,062 | 0,083 | 0,095 | 0,145 | 0,216 |
| Levantamiento | 0,465 | 0,131 | 0,096 | 0,091 | 0,118 | 0,098 |

**Tabla R17b. Matriz de confusión normalizada por fila, Fusión (15), sesión excluida. Filas: clase real; columnas: predicha**

| Real \ Predicha | Reposo | Flex. codo | Ext. codo | Flex. hombro | Ext. hombro | Levantamiento |
|---|---|---|---|---|---|---|
| Reposo | 0,675 | 0,016 | 0,018 | 0,181 | 0,064 | 0,047 |
| Flex. codo | 0,614 | 0,065 | 0,000 | 0,064 | 0,033 | 0,225 |
| Ext. codo | 0,664 | 0,001 | 0,209 | 0,026 | 0,058 | 0,043 |
| Flex. hombro | 0,324 | 0,002 | 0,000 | 0,638 | 0,017 | 0,019 |
| Ext. hombro | 0,544 | 0,001 | 0,002 | 0,117 | 0,309 | 0,027 |
| Levantamiento | 0,493 | 0,055 | 0,011 | 0,174 | 0,005 | 0,262 |

**Tabla R18. Error de Monte Carlo de los IC por remuestreo de bloques (sEMG 14, discriminante lineal, sesión excluida; 4 semillas del remuestreo)**

| Réplicas | DE del límite inferior | DE del límite superior |
|---|---|---|
| 500 | 0,004 | 0,003 |
| 2000 | 0,002 | 0,002 |

**Tabla R19. Sensibilidad de E10 a la frecuencia de muestreo. AUC macro, media de 10 sujetos**

| fs | Ventana (muestras) | Paso (muestras) | HGB aleatoria | HGB por registro | Sujetos con aleatoria > registro | LDA aleatoria | LDA por registro | HGB etiquetas reasignadas | HGB 96,7 % compartido | HGB 0 % compartido |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 Hz (declarada en el artículo, principal) | 307 | 10 | 0,991 | 0,807 | 10 | 0,816 | 0,775 | 0,978 | 0,980 | 0,800 |
| 780,2 Hz (sensibilidad) | 234 | 8 | 0,985 | 0,788 | 10 | 0,800 | 0,762 | 0,968 | 0,982 | 0,789 |

**Tabla R20. Frecuencia de muestreo del conjunto público: lo que declara el artículo asociado frente a lo que muestran los datos (A3 y A3b)**

| Evidencia | Resultado | Lectura |
|---|---|---|
| Artículo asociado (Toro-Ossaba et al., 2024, sección 2.5) | Shimmer3 configurado a 1024 Hz para sEMG e IMU | Fuente primaria |
| Punto de −3 dB del pasaaltos, medido entre columna cruda y filtrada | 0,00975 ciclos/muestra → 1026 Hz si 10 Hz es el punto de media potencia | Compatible con 1024 Hz |
| Borde de rizado del ajuste Chebyshev (convención de scipy) | 0,01320 ciclos/muestra → 758 Hz | Depende de la convención de corte; descartado |
| Atenuación a 500 Hz si fs = 1024 Hz | Ganancia 1,001: el pasabajos de 500 Hz no tiene efecto visible | Sin consecuencia: E10 filtra la señal cruda |
| Fase a 0,02 ciclos/muestra | 85° | Filtro causal |
| Repeticiones por archivo (ángulo) | Mediana 10 picos en 60 archivos | Coincide con series de diez repeticiones |
| Periodo del ciclo (mediana de muestras) | 2300 muestras: 2,25 s a 1024 Hz; 2,95 s a 780,2 Hz | El artículo indica unos 2 s/ciclo: más cercano a 1024 Hz |

**Tabla R21. Bloques vecinos (A4): IC 95 % del AUC con bloques independientes (método de E1) frente a tramos móviles de L bloques consecutivos; 1000 réplicas**

| Características | Modelo | Partición | AUC | Independientes | L = 2 | L = 4 | L = 8 | Autocorrelación de rezago 1 |
|---|---|---|---|---|---|---|---|---|
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | Sesión excluida | 0,568 | [0,511; 0,625] | [0,530; 0,607] | [0,531; 0,608] | [0,526; 0,612] | 0,339 |
| sEMG, amplitud y espectro (14) | Refuerzo por gradiente | Carga excluida | 0,542 | [0,485; 0,596] | [0,509; 0,575] | [0,510; 0,574] | [0,507; 0,578] | 0,460 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | Sesión excluida | 0,569 | [0,510; 0,627] | [0,530; 0,607] | [0,532; 0,605] | [0,530; 0,606] | 0,410 |
| sEMG, amplitud y espectro (14) | Discriminante lineal | Carga excluida | 0,553 | [0,484; 0,620] | [0,512; 0,595] | [0,514; 0,595] | [0,514; 0,594] | 0,482 |
| Mecánicas (7) | Refuerzo por gradiente | Sesión excluida | 0,612 | [0,551; 0,678] | [0,572; 0,649] | [0,578; 0,647] | [0,582; 0,645] | 0,114 |
| Mecánicas (7) | Refuerzo por gradiente | Carga excluida | 0,594 | [0,534; 0,659] | [0,553; 0,632] | [0,560; 0,627] | [0,564; 0,626] | 0,178 |
| Fusión (15) | Refuerzo por gradiente | Sesión excluida | 0,625 | [0,565; 0,687] | [0,582; 0,667] | [0,589; 0,664] | [0,594; 0,658] | 0,212 |
| Fusión (15) | Refuerzo por gradiente | Carga excluida | 0,591 | [0,527; 0,656] | [0,550; 0,633] | [0,556; 0,624] | [0,563; 0,623] | 0,179 |
