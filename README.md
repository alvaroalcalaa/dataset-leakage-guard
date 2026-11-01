# Dataset Leakage Guard 🔎🧠

Revisa posibles solapamientos entre dos conjuntos de textos de IA antes de evaluar un modelo o un sistema RAG. Detecta **copias normalizadas, textos cercanos, grupos e identificadores compartidos** y genera informes JSON y HTML sin copiar los textos originales.

Creado el **1 de noviembre de 2026**. Código y ejemplos originales bajo licencia MIT. **Python 3.10+**, biblioteca estándar únicamente; sin modelos, APIs, claves ni instalación de paquetes.

## Instalación y demo

```bash
git clone https://github.com/alvaroalcalaa/dataset-leakage-guard.git
cd dataset-leakage-guard
python leakage_guard.py examples/train.json examples/evaluation.json --json report.json --html report.html
python -m unittest -v
```

El análisis de la demo termina con **1 intencionadamente**, porque detecta casos para revisar. Abre `report.html` en el navegador; el mismo JSON se imprime en la terminal.

Los datos representan fragmentos ficticios del prototipo de videojuego Bruma. Resultado esperado:

| Señal | Resultado |
|---|---|
| Copia normalizada | `train-lantern` ↔ `eval-lantern` |
| Texto cercano | `train-key` ↔ `eval-key`, Jaccard 0,875 |
| Grupo compartido | `chapter-port`, aunque los textos sean distintos |
| Registros de evaluación señalados | 3 de 4 |

Ningún caso se elimina ni se modifica automáticamente. Revisa los registros en los archivos de origen utilizando sus IDs.

## Formato de entrada

```json
{
  "records": [
    {"id": "row-1", "text": "Texto de ejemplo para revisar.", "group": "document-17"}
  ]
}
```

`id` y `text` son obligatorios. `group` es opcional y puede ser `null`: úsalo para identificar el mismo documento de origen, sesión, conversación u otra unidad que hayas decidido mantener en un solo conjunto. La herramienta no deduce grupos a partir del texto.

Los IDs deben ser únicos **dentro** de cada archivo. Si un ID aparece en ambos conjuntos se señala, incluso con textos distintos; esto puede ser simplemente una convención local de numeración y requiere revisión. Los grupos y los IDs se comparan literalmente, distinguiendo mayúsculas y espacios. Otros campos se ignoran.

Se admiten de 0 a 300 registros por archivo, hasta 2 MiB de JSON UTF-8 y textos no vacíos de hasta 20000 caracteres. IDs y grupos deben ser textos no vacíos de hasta 120 caracteres. Un conjunto vacío es válido: revisa los tamaños en `summary` para no interpretar una evaluación vacía como una evaluación limpia. Se rechazan claves JSON duplicadas e IDs repetidos dentro de un mismo archivo.

## Método de detección

1. **Copias:** normalización Unicode NFC, `casefold` y sustitución de cada secuencia de espacios por un único espacio, quitando extremos. No se eliminan acentos ni signos de puntuación.
2. **Textos cercanos:** palabras obtenidas con `\w+` de Python (incluye letras Unicode, números y guion bajo); conjuntos de tres palabras consecutivas. Se calcula Jaccard: tamaño de la intersección dividido por tamaño de la unión. Se señalan valores **mayores o iguales** al umbral, 0,8 por defecto.
3. **Grupos:** intersección de etiquetas `group` no nulas entre los conjuntos.
4. **IDs:** intersección de identificadores entre los conjuntos.

```bash
python leakage_guard.py examples/train.json examples/evaluation.json --threshold 0.9
```

El umbral admite valores finitos de 0,1 a 1. Un índice invertido limita las comparaciones cercanas a pares que comparten algún trigrama; por ser el umbral positivo, un par sin intersección no puede superarlo. Un par detectado como copia no se cuenta otra vez como cercano. Los trigramas son **conjuntos**: no conservan frecuencias. Dos textos con diferente orden pueden tener los mismos conjuntos; eso es una señal de similitud léxica, no equivalencia semántica.

Los textos con menos de tres palabras no reciben análisis cercano, pero sí análisis de copias, grupos e IDs. `coverage` muestra cuántos hay. Se conservan todos los pares coincidentes, aunque un texto esté repetido varias veces; `flagged_eval_records` cuenta cada registro de evaluación señalado una sola vez. El orden del informe es determinista e independiente del orden de entrada.

## Informes y códigos de salida

El JSON contiene `summary`, `method`, `coverage`, `exact_pairs`, `near_pairs`, `shared_groups` y `shared_ids`. El HTML presenta el mismo contenido, con los textos de IDs/grupos escapados y sin recursos externos. No se incluyen hashes ni fragmentos de los textos originales.

| Código | Significado |
|---:|---|
| 0 | No se encontraron señales con estos datos y parámetros |
| 1 | Hay al menos una señal para revisión; los informes se generan igualmente |
| 2 | Error de argumentos, datos o escritura |

`--json` y `--html` son opcionales. Sus rutas deben ser distintas entre sí y de las entradas. Los archivos de salida existentes se reemplazan; un error de escritura puede dejar una salida parcial.

## Límites reales

Es una auditoría léxica orientativa para conjuntos pequeños, **no una certificación de ausencia de contaminación**. No detecta de forma fiable paráfrasis, traducciones, similitud conceptual, imágenes, audio ni datos externos no suministrados. Una coincidencia puede ser una frase común o un dato legítimamente compartido; tampoco demuestra que un modelo se entrenara con ella.

Para RAG, compara fuentes que tu política exija mantener separadas o textos de respuestas que no deberían estar disponibles durante la evaluación. Que una fuente de conocimiento aparezca en el corpus de recuperación puede ser intencional; la decisión depende del diseño de tu evaluación.

Solo se comparan pares **entre** los dos archivos. No se buscan duplicados internos salvo IDs repetidos. No divide datasets, no entrena modelos, no corrige los datos ni calcula cuánto afecta un solapamiento a una métrica. Los IDs y las etiquetas de grupo sí aparecen en el informe: utiliza etiquetas adecuadas para compartirlo. Los datos se procesan localmente y no se envían por red.

Las pruebas utilizan datos sintéticos locales y cubren resultados conocidos, Unicode, Jaccard en el límite, textos cortos, recuentos sin duplicar, determinismo, escape HTML, omisión de textos y ejecución CLI. **No se ha probado integración con un proveedor de modelos, una base vectorial ni un framework externo.**

## Dependencias y licencia

Solo módulos estándar de Python (`json`, `unicodedata`, `re`, `collections`, `html`, etc.). Licencia MIT en `LICENSE`.
