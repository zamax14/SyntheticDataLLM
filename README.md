# SyntheticDataLLM 🤖

![build succeeded](https://img.shields.io/badge/Application-LLM-blue.svg) ![build succeeded](https://img.shields.io/badge/Version-0.1-yellow.svg)  ![build succeeded](https://img.shields.io/badge/Python-3.12+-brightgreen.svg) ![build succeeded](https://img.shields.io/badge/License-MIT-purple.svg)

## 📖 Descripción

Aplicación CLI en Python para generar datos sintéticos Q&A en español para entrenamiento de modelos LLM. Genera archivos con formato **taxonomy** inspirado en [`InstructLab`](https://github.com/instructlab/taxonomy). Este formato facilita el entrenamiento usando tanto la librería InstructLab como [`Unsloth`](https://unsloth.ai/).

## 🔧 Herramientas

- [`pdf2md.py`](./pdf2md.py): Aplicación CLI para convertir archivos PDF a formato Markdown.

## 📦 Requisitos

- Python 3.12+
- Librerías requeridas:
  - `jsonargparse`
  - `langchain`
  - `langchain-ollama`
  - `docling`
  - `polars`
  - `pandas`
  - `tqdm`
- [`ollama`](https://ollama.com/)

> 💡 Una vez instalado ollama, descarga y ejecuta cualquiera de los [`modelos`](https://ollama.com/search) disponibles. Se recomienda `llama3.1` para Q&A, `phi4` para resúmenes y `qwen3` para razonamiento.

## 🛠️ Instalación

1. Clonar el repositorio:

```bash
git clone git@github.com:zamax14/SyntheticDataLLM.git
cd SyntheticDataLLM
```

2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## 🚀 Uso

> 💡 Puedes modificar los archivos `create.yaml`, `convert.yaml`, `pdf2md.yaml`, `summary.yaml` o `create_reasoning.yaml` en el directorio `configs/`.

#### Crear archivos Q&A

```bash
python synthetic.py create --config configs/create.yaml
```

#### Crear dataset de razonamiento

```bash
python synthetic.py create_reasoning --config configs/create_reasoning.yaml
```

#### Convertir Q&A a CSV y/o JSONL

```bash
python synthetic.py convert --config configs/convert.yaml
```

#### Resumir párrafos

> ⚠️ Los archivos de entrada deben estar en formato Markdown.

```bash
python synthetic.py summarize --config configs/summary.yaml
```

#### Convertir PDF a Markdown

```bash
python pdf2md.py convert --config configs/pdf2md.yaml
```

## 📚 Referencias

- **[LangChain](https://www.langchain.com/)**: Framework para desarrollo de aplicaciones con modelos de lenguaje.
- **[Unsloth](https://unsloth.ai/)**: Herramientas para entrenamiento eficiente de modelos.
- **[InstructLab](https://github.com/instructlab/instructlab)**: Formato de taxonomía para datos de instrucción.
- **[Ollama](https://ollama.com/)**: Despliegue y gestión de modelos de IA locales.
- **[Docling](https://github.com/DS4SD/docling)**: Conversión de documentos PDF, DOCX, XLSX, HTML a Markdown.