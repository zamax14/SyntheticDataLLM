# Corpus diverso del IIEG (2023–2025)

Corpus local para una **futura** regeneración del conjunto de entrenamiento de la tesis. La recolección se amplió el 25 de septiembre de 2026 porque la muestra inicial de 20 documentos era insuficiente. Contiene **138 PDF con texto útil** y 138 archivos de texto, publicados en rutas del IIEG de 2023 a 2025: 75 de economía, 36 de gobierno y seguridad, 16 de sociedad, 7 de geografía y ambiente y 4 cuadernillos municipales. La muestra municipal usa Guadalajara, Puerto Vallarta, Mezquitic y Lagos de Moreno; no incorpora los 124 cuadernillos anteriores. La distribución temática sigue desbalanceada porque varias publicaciones ambientales son mapas sin texto extraíble.

## Archivos

- `manifest.csv`: página oficial de publicación, URL del PDF, título, mes indicado por la ruta de publicación, tema, serie, SHA-256, páginas y pasajes conservados. `source_group` identifica la página de catálogo; los cuatro cuadernillos comparten grupo propio.
- `md/*.md`: texto extraído con `pdftotext -layout` y dividido en bloques de al menos 200 caracteres y 30 palabras. Son los archivos listos para la entrada `data_path` de `synthetic.py`.
- `similarity.csv`: pasajes retirados por coincidencia normalizada exacta o similitud textual de al menos 0.94, con referencia al documento conservado. El SHA-256 corresponde al pasaje retirado.
- `skipped.csv`: publicaciones descubiertas que no llegaron al corpus por falta de texto, tamaño de descarga o porque todo su texto útil repetía otros documentos.
- `pdf/*.pdf`: copias locales de los originales y caché de descartes. Son datos ignorados por Git; solo los PDF del manifiesto pertenecen al corpus de entrenamiento.

Los 138 documentos suman 4,888 páginas. Se examinaron 6,446 pasajes de 141 PDF con texto candidato y se conservaron **4,640** en 138 documentos; se retiraron 474 coincidencias exactas (433 entre documentos y 41 internas) y 1,332 casi idénticas entre documentos. Quedaron 3,020 pasajes de economía, 783 de gobierno y seguridad, 483 de sociedad, 211 municipales y 143 de geografía y ambiente. El filtro de similitud es una heurística textual y no demuestra ausencia de paráfrasis o contenido equivalente. El PDF del manual de accesibilidad solo aportó tres pasajes.

## Reproducir y usar

Desde `llm-synthetic-data-develop/`:

```bash
python3 collect_iieg.py
python3 -m unittest -q test_collect_iieg.py
```

Se requieren `curl`, `pdfinfo` y `pdftotext`; el recolector no instala paquetes de Python. Reutiliza los PDF descargados y consulta las páginas oficiales solo si falta su copia local. Para generar pares más adelante, configurar `data_path: ./corpora/iieg_diverso_2023_2025/md` y una **salida nueva** en la configuración de `synthetic.py`, junto con un modelo disponible localmente. No se han generado pares ni entrenado modelos: la DGX no está disponible.

Antes de medir train/val, asociar cada `source_file` del CSV generado con `source_group` del manifiesto y separar por grupo; dividir solo por filas o por `source_file` puede repartir informes de la misma serie entre ambos conjuntos. Equilibrar las áreas al seleccionar pasajes para generación o entrenamiento, revisar manualmente una muestra de texto y de pares generados, y repetir la auditoría de similitud después de cualquier conversión o fragmentación distinta.
