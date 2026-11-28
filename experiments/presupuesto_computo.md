# Presupuesto de cómputo

Estimación del tiempo de GPU por modelo para el ajuste de hiperparámetros y la ejecución definitiva, medida el 26 de noviembre de 2026 en la GPU del proyecto (NVIDIA GeForce RTX 3050 Laptop, 6 GB) con el pipeline de `configs/base.yaml`: caché en disco, una corrida por proceso, XLA, precisión mixta y asignador `cuda_malloc_async`. Las mediciones están en `outputs/speed/<modelo>.csv` y se obtienen con `python -m src.speed --model <modelo>`.

| Modelo | Lote (micro × pasos) | img/s A | img/s B | s/época A | s/época B | Pico GPU TF (GB) | Pico GPU nvidia-smi (GB) | Ajuste (h) | Definitiva, peor caso (h) | Definitiva, típica (h) | Total peor caso (h) | Total típico (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 32 × 1 | 1 252 | 403 | 6,8 | 18,4 | 0,87 | 1,00 | 0,69 | 0,98 | 0,59 | 1,67 | 1,28 |
| ResNet-50 | 32 × 1 | 724 | 200 | 11,8 | 36,6 | 1,52 | 1,67 | 1,28 | 1,86 | 1,10 | 3,14 | 2,37 |
| VGG-16 | 32 × 1 | 275 | 90 | 29,2 | 80,5 | 5,02 | 5,21 | 2,36 | 3,69 | 2,01 | 6,05 | 4,37 |
| DenseNet-201 | 32 × 1 | 454 | 126 | 18,7 | 58,1 | 2,23 | 2,45 | 3,43 | 4,17 | 2,96 | 7,60 | 6,39 |
| Inception-V3 | 32 × 1 | 983 | 265 | 8,7 | 27,7 | 1,08 | 1,25 | 1,15 | 1,58 | 1,00 | 2,73 | 2,15 |
| ConvNeXt-Tiny | 32 × 1 | 583 | 194 | 14,7 | 38,3 | 2,56 | 2,87 | 1,40 | 2,01 | 1,21 | 3,40 | 2,61 |
| ViT-B/16 | 32 × 1 | 239 | 77 | 36,0 | 96,9 | 3,41 | 3,67 | 2,97 | 4,56 | 2,55 | 7,54 | 5,52 |
| EfficientNet-B0 | 32 × 1 | 1 053 | 327 | 8,1 | 22,6 | 1,05 | 1,25 | 0,99 | 1,33 | 0,86 | 2,31 | 1,84 |
| CCI + MobileNetV2 (backbone provisional) | 32 × 1 | 651 | 371 | 11,9 | 19,9 | 0,91 | 1,08 | 0,80 | 1,09 | 0,68 | 1,89 | 1,48 |
| **Total** | | | | | | | | **15,1** | **21,3** | **13,0** | **36,3** | **28,0** |

**Total general: entre 28,0 h (caso típico) y 36,3 h (peor caso) de GPU para los nueve modelos; con el margen del 15 % que sugiere la validación (abajo), entre 32 y 42 h.** Con el mismo método, el protocolo original (12 configuraciones de ajuste y 25 corridas definitivas por modelo, sin contar el piloto) requeriría 94,9 h en el caso típico y 136,4 h en el peor caso. Las variantes de ablación del modelo original (D4) no están incluidas: cada una costaría aproximadamente lo mismo que la fila CCI en la ejecución definitiva, sin ajuste.

## Escenarios

- **Ajuste:** 6 configuraciones (desviación D1) en el pliegue 0, cada una con 5 épocas de la fase A y 15 de la fase B, sin descontar la parada temprana; es una cota superior. MobileNetV2 ya tiene evaluadas t00 a t06, así que con D1 su ajuste no necesita más tiempo de GPU; la fila muestra lo que costarían 6 configuraciones con el pipeline actual.
- **Ejecución definitiva:** 5 corridas (semilla 0 en los pliegues 0 a 4, desviación D3), con 5 épocas de la fase A, más la evaluación final de validación, prueba y train sin aumento. Peor caso: las 30 épocas de la fase B. Caso típico: parada temprana hacia la época 15 de la fase B.
- **Total:** ajuste más ejecución definitiva en cada escenario.

## Método de estimación

- Imágenes por segundo: media de 50 pasos de entrenamiento tras 10 de calentamiento, con el pipeline real (caché en disco, aumento, pesos de clase) sobre el train del pliegue 0, en las fases A (backbone congelado) y B (todo descongelado). Ningún modelo usa acumulación de gradientes: todos entrenan con lotes de 32.
- Segundos por época = imágenes de train / (img/s) + predicción del pliegue de validación (`val_predict`, medida con el modelo de la fase B). Se usan las medias de los pliegues 0 a 4: 6867 imágenes de train, 1717 de validación y 1431 de prueba.
- Sobrecoste por corrida: arranque del proceso hijo y carga de datos (15 s, medido en corridas reales de MobileNetV2: la verificación SHA-256 de la caché tarda unos 5 s), dos compilaciones de XLA por fase (lote completo y último lote incompleto, cada una estimada como el primer paso menos un paso normal), dos compilaciones de la predicción y la predicción final de validación. Las compilaciones se midieron con la caché de autotuning vacía, así que son una cota superior. En DenseNet-201, la compilación de la fase B tarda 469 s en frío y 433 s con la caché llena, unos 18 min por corrida, y explica que su ajuste cueste más que el de VGG-16.
- Las mediciones suponen la GPU y la RAM libres de otros procesos pesados. El 14 de noviembre, con la caché en memoria y el sistema usando swap, MobileNetV2 tardaba unas 6 veces más por época (120 s frente a 18,4 s en la fase B).
- VGG-16 se mide sin acumulación de gradientes (`gradient_accumulation_steps: 1`), el valor configurado. Con 2 micro-lotes de 16 procesaba 75 img/s en la fase B (`outputs/speed/vgg16_accum2.csv`) y con 4 de 8, 55 img/s (`outputs/speed/vgg16_accum4.csv`).
- Validación: tres configuraciones completas del ajuste de MobileNetV2 (5 épocas de la fase A y 15 de la B, sin parada temprana), ejecutadas con `src.train.run_training_isolated` y salidas fuera de `outputs/`, tardaron 482, 449 y 430 s, frente a 414 s estimados (+4 % a +16 %). Las épocas estables coinciden con la estimación (7,2 s en la fase A y 18,9 s en la B); la diferencia viene de una compilación algo mayor (primera época de la fase B de 76 a 99 s) y de una a tres épocas por corrida de unos 28 s, sin causa identificada. Conviene sumar un 15 % a las horas de la tabla.
