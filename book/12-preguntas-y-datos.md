---
title: Preguntas de investigación y conjunto de datos
---

Este capítulo amplía, para el benchmark del Entregable 2, la [definición del problema](02-definicion-del-problema.md). La métrica primaria (exactitud balanceada), el margen de relevancia práctica $\delta = 0{,}02$, la unidad de independencia (la lesión) y la partición se heredan sin cambios.

## Formulación

Sea $P$ la distribución conjunta de imágenes dermatoscópicas $x \in \mathcal{X}$ y diagnósticos $y \in \mathcal{Y}$, con $|\mathcal{Y}| = 7$. Cada modelo del benchmark define un espacio de hipótesis $\mathcal{H}_m = \{ f_\theta : \theta \in \Theta_m \}$, fijado por su arquitectura, y se entrena minimizando la entropía cruzada ponderada sobre el conjunto de entrenamiento. Su desempeño es el riesgo esperado

$$
R(f) = \mathbb{E}_{(x, y) \sim P}\left[ \mathcal{L}\big(f(x), y\big) \right],
$$

que se estima con el riesgo empírico sobre el conjunto de prueba sellado de 1431 imágenes. Como la métrica primaria es la exactitud balanceada, la comparación entre modelos se hace sobre $1 - \text{BA}$, el riesgo con pérdida 0-1 ponderada por la inversa de la frecuencia de cada clase.

## Preguntas de investigación

Las preguntas se fijan antes de entrenar cualquier modelo nuevo. Todas usan $\alpha = 0{,}05$ y la exactitud balanceada en prueba de las corridas pareadas, definidas por la misma semilla y el mismo pliegue de validación.

**RQ1 (confirmatoria).** ¿Supera alguno de los cinco modelos del estado del arte al mejor modelo base del Entregable 1 en al menos $\delta = 0{,}02$? Para cada modelo $k$, la hipótesis nula es $H_0 : \mu_k - \mu_{\text{base}} \le 0$ frente a $H_1 : \mu_k - \mu_{\text{base}} > 0$. Forman una familia de cinco contrastes con corrección de Holm. La decisión es "superior y relevante" si el p ajustado es menor que 0,05 y la diferencia estimada es al menos $\delta$; "superior sin relevancia" si es significativa pero menor que $\delta$; "equivalente" si la prueba TOST con margen $\pm\delta$ es significativa; y "no concluyente" en otro caso.

**RQ2 (confirmatoria).** ¿Supera cada modelo profundo al baseline no profundo, la regresión logística sobre embeddings congelados, en al menos $\delta$? Para cada uno de los ocho modelos profundos, $H_0 : \mu_k - \mu_{\text{LR}} \le 0$. Forman una familia de ocho contrastes con corrección de Holm y la misma regla de decisión. Un modelo que no supere este baseline no aporta valor sobre una representación genérica de ImageNet.

**RQ3 (exploratoria).** ¿Depende el desempeño de los modelos de la fuente de adquisición de la imagen? Para cada modelo se compara la exactitud balanceada en prueba entre las imágenes de vidir_molemax y las del resto de fuentes, con un intervalo bootstrap por lesión, y se explora si la diferencia es distinta entre las redes convolucionales y el transformador de visión. Al ser exploratoria, se corrige por la tasa de falsos descubrimientos (Benjamini–Hochberg) y sus resultados generan hipótesis para el Entregable 3, sin sustentar conclusiones confirmatorias.

En las dos preguntas confirmatorias, cada contraste se evalúa con tres pruebas de naturaleza distinta, como exige el protocolo: la prueba t pareada con la corrección de @nadeau2003, la prueba de rangos con signo de Wilcoxon junto con un bootstrap pareado por lesión de 10000 remuestreos, y la prueba de Friedman con la corrección de Iman y Davenport seguida de comparaciones con Holm. Las pruebas de McNemar y DeLong se reportan como complementarias.

## Justificación del conjunto de datos

### Ética, privacidad y licencia

HAM10000 se distribuye bajo la licencia CC BY-NC 4.0, que permite su uso académico sin fines comerciales, y sus imágenes están anonimizadas: los metadatos no contienen nombres, fechas ni identificadores de paciente [@tschandl2018]. Los pesos preentrenados proceden de ImageNet a través de `keras.applications` y `keras_hub`, y su uso se limita a la investigación. El principal riesgo ético es el desempeño desigual entre poblaciones: un clasificador que falle más en ciertos grupos trasladaría ese error a decisiones de biopsia o vigilancia, por lo que el desempeño se reporta desagregado por sexo, edad, localización y fuente.

### Representatividad y limitaciones

Las imágenes proceden de dos centros, en Viena (Austria) y Queensland (Australia), recogidas a lo largo de unos veinte años [@tschandl2018], de modo que representan sobre todo a poblaciones de piel clara. El conjunto no registra el fototipo, por lo que esa limitación no puede cuantificarse, y los resultados no deben extrapolarse a pieles oscuras [@daneshjou2021]. Además, la clase nv concentra el 66,95 % de las imágenes, las fuentes difieren sistemáticamente en color y composición por clase, y el 46,7 % de las imágenes carece de confirmación histopatológica, como documenta el análisis exploratorio.

### Partición oficial de ISIC 2018

No se usa la partición oficial del reto ISIC 2018 por dos razones. La primera es que el protocolo exige validación cruzada repetida para estimar la variabilidad entre corridas, y la partición oficial ofrece un único conjunto de prueba sin relación documentada entre sus lesiones y las de entrenamiento. La segunda es que la partición propia agrupa por lesión y queda verificada sin solapamiento, mientras que con la oficial esa verificación no es posible. El conjunto de prueba oficial se conserva sellado y no se consulta, de modo que puede servir como evaluación externa en el Entregable 3.

## Ficha del conjunto de datos

| Campo | Valor |
|---|---|
| Nombre y versión | HAM10000, versión 4 en Harvard Dataverse (doi: 10.7910/DVN/DBW86T) [@tschandl2018data] |
| Tamaño | 10015 imágenes de 7470 lesiones, con un archivo de metadatos y una máscara de segmentación por imagen |
| Tipo de datos | Imágenes dermatoscópicas JPEG, RGB, 8 bits por canal, 600 × 450 píxeles |
| Variable objetivo | Diagnóstico en 7 clases: nv, mel, bkl, bcc, akiec, vasc y df |
| Etiquetado | Histopatología (53,3 %), seguimiento (37,0 %), consenso de expertos (9,0 %) y microscopía confocal (0,7 %); sin acuerdo entre anotadores disponible |
| Recolección | Dos centros (Viena y Queensland) y cuatro subconjuntos de adquisición, a lo largo de unos veinte años |
| Cobertura | Poblaciones mayoritariamente de piel clara; fototipo no registrado |
| Filtros aplicados | Ninguno: no se eliminó ninguna imagen; no se encontraron duplicados confirmados |
| Unidad de independencia | La lesión (`lesion_id`); no hay identificador de paciente |
| Trazabilidad | Hashes SHA-256 en `data/hashes.json` y de la partición en `data/splits_hash.json` |
