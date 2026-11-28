# Declaración del estándar de profundidad

Fecha: 2026-11-26
Tag de Git: `e3-estandar`, previo al congelamiento del protocolo.

El proyecto adopta el estándar único de profundidad teórica de la sección 2 del Entregable 3 y se compromete con sus cinco requisitos:

1. **Sustento teórico demostrado:** la invarianza exacta del modelo original a las ganancias de canal, f(Dx) = f(x) para toda ganancia diagonal positiva D, demostrada formalmente con supuestos explícitos y verificada numéricamente.
2. **Evaluación amplia:** tres conjuntos de datos adicionales de naturaleza distinta, elegidos y documentados en `experiments/protocolo.md` antes de ver los resultados.
3. **Familia completa de contrastes:** el modelo original frente a cada modelo del benchmark y cada baseline fuerte, con corrección de Holm sobre toda la familia.
4. **Potencia:** potencia por contraste de al menos 0,90 y potencia conjunta reportada, con al menos 10 semillas o las que exija el análisis de potencia.
5. **Pruebas de hipótesis:** al menos tres pruebas distintas por contraste.

El estándar no se reducirá después de ver los resultados. Si el análisis de potencia muestra que el cómputo disponible (una GPU NVIDIA GeForce RTX 3050 Laptop de 6 GB) no permite alcanzar la potencia exigida, se declarará antes de congelar el protocolo, se reportará la diferencia mínima detectable y el resultado se interpretará como no concluyente, no como equivalencia.

**Declaración previa al congelamiento (2026-11-26).** El análisis del cómputo disponible muestra que no se alcanzan los requisitos 2 y 4 tal como están formulados: los conjuntos adicionales se usan solo para evaluación externa, sin reentrenar, y la ejecución definitiva tiene 5 corridas con una semilla. Por lo tanto, el resultado de superioridad (R3) se interpretará como no concluyente si no se alcanza la potencia, no como equivalencia. Las desviaciones D1 a D5 se detallan en `experiments/protocolo.md`.
