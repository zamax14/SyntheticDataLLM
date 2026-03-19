# Copilot Instructions — SyntheticDataLLM

## Descripción del proyecto

Generador de datos sintéticos Q&A en español para entrenamiento de modelos LLM.
Convierte documentos PDF a Markdown y genera pares pregunta-respuesta usando modelos de lenguaje locales vía Ollama + LangChain.

Repositorio: `git@github.com:zamax14/SyntheticDataLLM.git`

## Stack tecnológico

- **Python 3.12+**
- **LangChain + langchain-ollama**: orquestación de prompts y modelos
- **Ollama**: servidor de modelos locales (llama3.1, phi4, qwen3)
- **Docling**: conversión de PDF/DOCX a Markdown
- **jsonargparse**: CLI con soporte YAML
- **Polars / Pandas**: manipulación de datos tabulares
- **tqdm**: barras de progreso

## Estructura del proyecto

```
synthetic.py        # CLI principal: create, create_reasoning, convert, summarize
pdf2md.py           # CLI para conversión PDF → Markdown
core/
  extractor.py      # Extractores: QandAExtractor, SummarizeExtractor, ReasoningExtractor
  llm.py            # Wrapper de OllamaLLM con generación por tipo
  paragraph.py      # Procesamiento y limpieza de párrafos
  pattern.py        # Patrones regex reutilizables
  taxonomy.py       # Modelos de datos para formato InstructLab
utils/
  logger.py         # Logger centralizado
  templates.py      # Prompts para LLM (solo español)
  utils.py          # Funciones de I/O y conversión
configs/            # Archivos YAML de configuración
```

## Reglas de código

### Tipado obligatorio
- Todos los parámetros de funciones deben tener type hints.
- Los valores de retorno deben tener type hints.

### Documentación (español)
- Toda función debe tener docstring en español con:
  - Descripción breve
  - `Args:` con nombre, tipo y descripción de cada parámetro
  - `Returns:` con tipo y descripción del valor de retorno

### Comentarios
- Solo agregar comentarios cuando el bloque de código no sea autoexplicativo.
- No saturar de comentarios; un comentario breve por bloque lógico si es necesario.

### Principio Unix
- Funciones pequeñas y enfocadas en una sola responsabilidad.
- Evitar funciones monolíticas; dividir en subfunciones cuando la lógica crezca.
- Los cambios deben ser fáciles de implementar y testear de forma aislada.

### Imports
- Todas las importaciones deben ir únicamente en la parte superior del archivo.
- No usar imports inline ni dentro de funciones.

### Idioma
- Todo el código de generación de datos sintéticos es en español.
- No hay soporte para generación en inglés.
- Los prompts, mensajes de log y documentación técnica van en español.

### Convenciones generales
- Usar `dataclass` para clases de datos.
- Usar `Enum` para valores fijos.
- Preferir `pathlib` o `os.path` para rutas.
- Formato de salida: YAML (InstructLab), CSV, JSONL.
