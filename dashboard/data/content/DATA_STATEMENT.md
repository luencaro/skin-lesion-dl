# Declaración de datos

Declaración del conjunto de datos usado en el proyecto, en el espíritu de los *data statements*
(Bender y Friedman, 2018) y las *datasheets for datasets* (Gebru et al., 2021). Las cifras proceden
del análisis exploratorio (notebooks 05 y 06) y se recalculan en `python -m src.export_dashboard`.

## Procedencia

- **Conjunto:** HAM10000 (*Human Against Machine with 10000 training images*), versión 4 en Harvard
  Dataverse, doi: [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
- **Publicación:** Tschandl, P., Rosendahl, C. y Kittler, H. (2018). The HAM10000 dataset, a large
  collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*,
  5, 180161. doi: 10.1038/sdata.2018.161.
- **Recolección:** unos veinte años de imágenes de dos centros: el Departamento de Dermatología de la
  Universidad Médica de Viena (Austria) y la consulta de cáncer de piel de Cliff Rosendahl en
  Queensland (Australia), con distintos equipos y protocolos. Los metadatos las agrupan en cuatro
  fuentes de adquisición: rosendahl, vidir_modern, vidir_molemax y vienna_dias.
- **Contenido:** 10015 imágenes dermatoscópicas JPEG RGB de 600 × 450 píxeles, de 7470 lesiones, con
  diagnóstico en 7 clases, método de confirmación, edad, sexo y localización anatómica, y una máscara de
  segmentación por imagen.
- **Integridad:** SHA-256 de los archivos en `data/hashes.json` y de la partición en
  `data/splits_hash.json`; contrato de los metadatos en `configs/data_contract.yaml`.
- **No se usa:** el conjunto de prueba de ISIC 2018 que se distribuye en la misma versión.

## Licencia

HAM10000 se distribuye bajo la licencia **Creative Commons Atribución-NoComercial 4.0 Internacional
(CC BY-NC 4.0)**: se permite el uso y la adaptación con atribución y sin fines comerciales. Por eso:

- el repositorio no incluye los datos crudos (`data/raw/` está en `.gitignore`); cada usuario los
  descarga de la fuente original;
- las miniaturas y los ejemplos de `dashboard/data/` (unas 180 imágenes reducidas) se muestran con
  fines académicos, con atribución, y siguen sujetos a CC BY-NC 4.0;
- la licencia MIT del repositorio cubre solo el código.

## Etiquetas

- Histopatología en el 53,3 % de las imágenes (41,3 % de las lesiones), seguimiento en el 37,0 %,
  consenso de expertos en el 9,0 % y microscopía confocal en el 0,7 %. No hay acuerdo entre
  anotadores disponible.
- El método se asocia fuertemente con la clase: las 3704 imágenes de seguimiento son nevus y las 69 de
  confocal, queratosis benignas.

## Sesgos de población

- **Fototipo:** las poblaciones de Austria y Queensland son mayoritariamente de piel clara y el
  fototipo no está registrado; el desempeño en pieles oscuras es desconocido.
- **Edad y sexo:** edad media de 51,9 ± 17,0 años, registrada en intervalos de cinco años; el sexo se
  asocia débilmente con la clase (V de Cramér = 0,095).
- **Clases:** fuerte desbalance (nv 66,95 % frente a df 1,15 %; razón de 58,3), que no refleja la
  prevalencia en ningún entorno clínico concreto.
- **Unidad de independencia:** la lesión; no hay identificador de paciente, así que un mismo paciente
  puede aparecer en train y en test con lesiones distintas.
- **Faltantes:** se concentran en vidir_modern (53 de los 57 faltantes de edad y los 57 de sexo).

## Sesgos de fuente

- La fuente se asocia fuertemente con la clase (V de Cramér = 0,341): rosendahl contiene 295 de las
  327 imágenes de akiec y vidir_molemax es nv en un 94,1 %.
- La fuente se asocia con el método diagnóstico (V = 0,580), la localización (V = 0,293) y la edad.
- El color depende mucho más de la fuente que de la clase: ε² de Kruskal–Wallis de 0,669 para la
  saturación según la fuente frente a 0,115 según la clase, y de 0,421 frente a 0,125 en el canal
  rojo. Los embeddings preentrenados predicen la fuente con una exactitud balanceada de 0,829.
- Consecuencia: un modelo puede aprender atajos de la fuente (color, viñeteado, reglas de medida) en
  lugar de rasgos de la lesión. El modelo original del proyecto ataca la parte de ese atajo que se
  debe a las ganancias de color de cada equipo.

## Partición

Partición agrupada por lesión con `StratifiedGroupKFold`: test de 1431 imágenes (14,29 %) y train de
8584 en cinco pliegues de 1716 o 1717 imágenes. Ninguna lesión aparece en más de una partición ni en
más de un pliegue; las proporciones de clase difieren en menos de 0,2 puntos porcentuales; una
validación adversarial con 48 características de color no distingue train de test (AUC = 0,483).
No se encontraron duplicados confirmados entre lesiones (33 candidatos por pHash revisados).
