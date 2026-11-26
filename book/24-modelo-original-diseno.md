---
title: "Modelo original: diseño"
---

El modelo original añade, antes del backbone, una capa de constancia de color aprendible e invariante a las ganancias de canal (en adelante, la capa CCI). Este capítulo presenta la limitación que motiva la capa, la hipótesis del mecanismo, su formulación, el pseudocódigo, la demostración de su propiedad y su complejidad.

## Limitación que se aborda

El paper guía constata que el desempeño cae 25 puntos de exactitud balanceada al pasar de HAM10000 a BCN20000 y que el orden de los modelos se invierte [@pinar2026], pero no analiza si, dentro de HAM10000, el desempeño ya depende de la fuente de adquisición. El análisis exploratorio muestra que esa dependencia es plausible y que el color es su vehículo principal. La saturación depende de la fuente con un tamaño de efecto de ε² = 0,669, frente a 0,115 de la clase; el canal rojo, con 0,421 frente a 0,125. Los embeddings preentrenados predicen la fuente con una exactitud balanceada de 0,829, frente a 0,479 de la clase, y el fondo de la imagen, sin la lesión, predice la fuente con 0,701.

La limitación se cuantifica como la diferencia absoluta de exactitud balanceada en prueba entre las imágenes de vidir_molemax y las del resto de fuentes, con su intervalo de confianza del 95 % por bootstrap sobre lesiones, en los modelos del benchmark: [COMPLETAR con los resultados del Entregable 2].

## Fuente del mecanismo y transferencia

El mecanismo procede de la estimación del iluminante en fotografía computacional. Shades of Gray estima el iluminante de cada canal con la media de Minkowski de orden $p$ de sus intensidades [@finlayson2004], y FC4 lo estima con una red que pondera cada región de la imagen según su confianza, supervisada con iluminantes medidos [@hu2017]. El mecanismo es transferible porque el problema comparte la estructura que lo motiva: imágenes del mismo tipo de escena, la piel, capturadas por equipos distintos, cuya diferencia de color se aproxima en primer orden por una ganancia multiplicativa por canal, el modelo diagonal de von Kries.

La adaptación modifica el mecanismo en tres aspectos. El orden $p$, fijo en Shades of Gray, se aprende por gradiente. Los pesos espaciales, que en FC4 se supervisan con iluminantes reales, se aprenden sin ellos, solo a través de la pérdida de clasificación, porque en dermatoscopia no hay iluminantes etiquetados. Y los pesos se calculan sobre una representación de la imagen que no cambia con las ganancias de canal, lo que convierte la invarianza en una propiedad exacta de la arquitectura, que ni Shades of Gray con pesos ni FC4 garantizan.

## Hipótesis del mecanismo

La hipótesis, formulada antes de los experimentos definitivos, es la siguiente: la capa CCI reduce la dependencia del desempeño respecto de la fuente de adquisición, porque elimina por construcción las ganancias de canal propias de cada equipo, y por tanto reduce en al menos $\delta = 0{,}02$ la diferencia de exactitud balanceada en prueba entre vidir_molemax y el resto de fuentes, respecto del mismo backbone sin la capa.

Si el mecanismo funciona por esa razón, deben cumplirse también cuatro predicciones secundarias, declaradas a priori:

1. Bajo perturbaciones de ganancia por canal con factores entre 0,5 y 2, la exactitud balanceada del modelo con la capa no cambia, salvo por la saturación de píxeles, mientras que la del backbone sin la capa cae con la severidad.
2. La ventaja desaparece al retirar la invarianza, es decir, al calcular los pesos sobre la imagen sin normalizar.
3. La capa no aporta ventaja frente al desenfoque, el ruido gaussiano o la compresión JPEG, que no son ganancias de canal.
4. Un clasificador lineal predice la fuente con menor exactitud balanceada a partir de las representaciones del modelo con la capa que a partir de las del backbone sin ella.

El diseño se considera exitoso si la diferencia entre fuentes se reduce en al menos $\delta$ con el límite inferior del intervalo de confianza mayor que cero, si se cumple la predicción 1 y si la exactitud balanceada global no es peor que la del backbone sin la capa (TOST con margen $\pm\delta$). Se considera fallido si la diferencia entre fuentes no se reduce o si la exactitud global empeora en más de $\delta$. La superioridad frente a toda la familia de contrastes (R3) es una hipótesis separada, evaluada con el protocolo estadístico.

## Formulación

Sea $x \in \mathbb{R}_{>0}^{N \times 3}$ una imagen con $N$ píxeles e intensidades positivas, y sea $D = \operatorname{diag}(d_1, d_2, d_3)$, con $d_c > 0$, una ganancia por canal, de modo que $(Dx)_{i,c} = d_c \, x_{i,c}$. La capa trabaja sobre una versión reducida $\bar{x} = \operatorname{pool}(x)$ de $\bar{N} = 56 \times 56$ píxeles, obtenida por promedio de áreas, y calcula en cuatro pasos:

$$
z_{i,c}(\bar{x}) = \log \bar{x}_{i,c} - \frac{1}{\bar{N}} \sum_{j=1}^{\bar{N}} \log \bar{x}_{j,c},
$$ (cci-guia)

$$
w_i(\bar{x}) = \frac{\exp\big(g_\phi(z(\bar{x}))_i\big)}{\sum_{j=1}^{\bar{N}} \exp\big(g_\phi(z(\bar{x}))_j\big)}, \qquad p = 1 + \operatorname{softplus}(\vartheta),
$$ (cci-pesos)

$$
e_c(\bar{x}) = \left( \sum_{i=1}^{\bar{N}} w_i(\bar{x}) \, \bar{x}_{i,c}^{\,p} \right)^{1/p},
$$ (cci-iluminante)

$$
h(x)_{i,c} = \kappa \, \frac{x_{i,c}}{e_c(\bar{x})}.
$$ (cci-correccion)

La ecuación {eq}`cci-guia` centra el logaritmo de cada canal en su media espacial; $g_\phi$ es una red convolucional pequeña (dos convoluciones de 3 × 3 con 16 filtros y ReLU, y una de 1 × 1 con una salida) que asigna a cada píxel un puntaje de confianza, convertido en pesos por la softmax espacial de {eq}`cci-pesos`. La ecuación {eq}`cci-iluminante` es la media de Minkowski ponderada de orden $p \ge 1$, que se reduce a Shades of Gray cuando los pesos son uniformes y $p$ es fijo, y {eq}`cci-correccion` divide cada canal por su iluminante estimado, con una constante $\kappa$ que devuelve las intensidades al rango que espera el backbone. El modelo completo es $f = \operatorname{softmax} \circ b_\psi \circ h$, donde $b_\psi$ es el backbone preentrenado con su cabeza de clasificación, y se entrena con la misma entropía cruzada ponderada que los modelos del benchmark.

Por estabilidad numérica, la ecuación {eq}`cci-iluminante` se calcula en el dominio logarítmico, $\log e_c = \frac{1}{p} \operatorname{logsumexp}_i\big(\log w_i + p \log \bar{x}_{i,c}\big)$, y las intensidades se acotan inferiormente por $\varepsilon = 10^{-3}$ antes de tomar logaritmos.

### Gradientes

Con $S_c = \sum_i w_i \bar{x}_{i,c}^{\,p}$, de modo que $e_c = S_c^{1/p}$, los gradientes propios de la capa son

$$
\frac{\partial e_c}{\partial w_i} = \frac{e_c^{\,1-p}}{p} \, \bar{x}_{i,c}^{\,p}, \qquad
\frac{\partial e_c}{\partial p} = e_c \left( \frac{\sum_i w_i \bar{x}_{i,c}^{\,p} \log \bar{x}_{i,c}}{p \, S_c} - \frac{\log S_c}{p^2} \right), \qquad
\frac{\partial p}{\partial \vartheta} = \sigma(\vartheta),
$$

$$
\frac{\partial w_i}{\partial s_k} = w_i (\mathbb{1}_{ik} - w_k), \qquad
\frac{\partial h_{i,c}}{\partial e_c} = -\kappa \, \frac{x_{i,c}}{e_c^{2}},
$$

donde $s = g_\phi(z)$ y $\sigma$ es la función logística. Los gradientes respecto de $\phi$ se obtienen por la regla de la cadena a través de $s$; la diferenciación automática del framework los calcula, y una prueba unitaria los compara con diferencias finitas.

## Pseudocódigo

```text
Algoritmo: capa CCI (paso hacia adelante)
Entrada: imagen x de H × W × 3 con intensidades en [0, 255]; parámetros φ, ϑ; constantes κ, ε
Salida: imagen corregida h(x) de H × W × 3

1. x ← max(x, ε)                                  ▹ intensidades positivas
2. x̄ ← promedio de áreas de x a 56 × 56           ▹ resolución reducida
3. Para cada canal c: z_c ← log x̄_c − media(log x̄_c)   ▹ ecuación (cci-guia)
4. s ← g_φ(z);  w ← softmax espacial(s)            ▹ ecuación (cci-pesos)
5. p ← 1 + softplus(ϑ)
6. Para cada canal c: log e_c ← (1/p) · logsumexp(log w + p · log x̄_c)   ▹ (cci-iluminante)
7. Para cada canal c: h_c ← κ · x_c / exp(log e_c)                       ▹ (cci-correccion)
8. Devolver h

Entrenamiento: la capa se antepone al backbone y el modelo completo se entrena con el mismo
pipeline, la misma pérdida, el mismo aumento de datos y la misma parada temprana sobre la
exactitud balanceada de validación que los modelos del benchmark; ϑ se inicializa de modo que
p = 6 y φ de modo que los pesos sean uniformes, lo que hace que la capa empiece como Shades of Gray.
```

## Propiedad demostrada: invarianza a las ganancias de canal

**Teorema.** Para toda imagen $x \in \mathbb{R}_{>0}^{N \times 3}$ y toda ganancia diagonal $D$ con $d_c > 0$, la capa cumple $h(Dx) = h(x)$. En consecuencia, el modelo completo cumple $f(Dx) = f(x)$ para cualquier backbone $b_\psi$.

*Demostración.* El promedio de áreas es lineal y actúa por separado en cada canal, de modo que $\overline{Dx} = D\bar{x}$. Para la representación de la ecuación {eq}`cci-guia`,

$$
z_{i,c}(D\bar{x}) = \log d_c + \log \bar{x}_{i,c} - \frac{1}{\bar{N}} \sum_j \big(\log d_c + \log \bar{x}_{j,c}\big) = z_{i,c}(\bar{x}),
$$

por lo que los puntajes $g_\phi(z)$ y los pesos $w_i$ no cambian. Como los pesos son invariantes y $p$ no depende de la imagen, la media de Minkowski es homogénea de grado uno en cada canal:

$$
e_c(D\bar{x}) = \left( \sum_i w_i \, d_c^{\,p} \, \bar{x}_{i,c}^{\,p} \right)^{1/p} = d_c \, e_c(\bar{x}).
$$

Por último,

$$
h(Dx)_{i,c} = \kappa \, \frac{d_c \, x_{i,c}}{d_c \, e_c(\bar{x})} = h(x)_{i,c},
$$

y, como $f$ depende de $x$ solo a través de $h(x)$, $f(Dx) = f(x)$. $\blacksquare$

**Corolario.** Con $D = \alpha I$, el modelo es invariante a cualquier cambio global de brillo multiplicativo.

**Supuestos y su plausibilidad.** La demostración supone intensidades estrictamente positivas, sin saturación, y que la diferencia de color entre equipos es una ganancia diagonal. Los datos reales violan parcialmente los tres supuestos. La cota $\varepsilon$ rompe la invarianza exacta en los píxeles negros, frecuentes en las esquinas viñeteadas. La saturación a 255 rompe la homogeneidad en los píxeles recortados, presentes en más del 1 % de los píxeles de 5,28 % de las imágenes de entrenamiento. Y las fuentes difieren también en saturación, gamma y viñeteado, que no son ganancias diagonales. Por ello, la invarianza se presenta como garantía exacta frente a las ganancias de canal y como motivación, no como garantía, frente al cambio de fuente real.

**Verificación numérica.** Se generan ganancias $d_c$ con distribución log-uniforme entre 0,5 y 2, se aplican a imágenes de prueba sin píxeles saturados ni nulos y se mide $\max \lvert f(Dx) - f(x) \rvert$, que debe ser del orden de la precisión de punto flotante, frente a la discrepancia del backbone sin la capa. La misma medición sobre imágenes reales con saturación cuantifica cuánto se aparta la invarianza práctica de la teórica.

## Complejidad

Con $\bar{N} = 3136$ píxeles reducidos y 16 filtros, la red $g_\phi$ tiene $3 \cdot 16 \cdot 9 + 16 + 16 \cdot 16 \cdot 9 + 16 + 16 + 1 = 2785$ parámetros, más el parámetro $\vartheta$, y requiere unos $8{,}7 \times 10^{6}$ multiplicaciones-sumas por imagen. El costo de la capa es $O(\bar{N} k^2 m)$ para la red de pesos, con $k$ el tamaño del núcleo y $m$ el número de filtros, más $O(N)$ para el promedio de áreas y la corrección, lineal en el número de píxeles tanto en entrenamiento como en inferencia. Es despreciable frente al backbone (por ejemplo, 4,5 GFLOPs en ConvNeXt-Tiny), de modo que el modelo original tiene prácticamente la misma capacidad que su backbone y no requiere una variante de capacidad igualada. Los FLOPs y el tiempo se miden empíricamente al variar la resolución de entrada.

## Backbone y ablación

El backbone se fija antes de los experimentos con una regla: es el modelo del benchmark con mayor exactitud balanceada media en validación en el Entregable 2, con los mismos pesos de ImageNet, de modo que el aporte original queda fuera del backbone y la comparación con él aísla el efecto de la capa.

La ablación aísla los tres componentes propios, con el mismo presupuesto de ajuste y una familia de contrastes separada con corrección de Holm:

| Variante | Orden $p$ | Pesos espaciales | Representación de los pesos |
|---|---|---|---|
| Modelo completo | Aprendido | Aprendidos | Invariante ($z$) |
| Sin $p$ aprendido | Fijo en 6 | Aprendidos | Invariante ($z$) |
| Sin pesos aprendidos | Aprendido | Uniformes | No aplica |
| Sin invarianza | Aprendido | Aprendidos | Imagen sin normalizar |
| Shades of Gray fijo [@barata2015] | Fijo en 6 | Uniformes | No aplica |
| Backbone sin la capa | No aplica | No aplica | No aplica |

La variante sin invarianza conserva el número de parámetros del modelo completo y pone a prueba la predicción secundaria 2; la de Shades of Gray fijo reproduce el trabajo cercano más directo.
