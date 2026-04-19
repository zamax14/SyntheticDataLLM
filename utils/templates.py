from langchain_core.prompts import ChatPromptTemplate


def create_q_and_a_prompt_es() -> ChatPromptTemplate:
    """
    Crea un template de prompt para generar pares de pregunta y respuesta
    a partir de un párrafo en español. La estructura de salida es impuesta
    por el modelo Pydantic QandAResponse.

    Returns:
        ChatPromptTemplate: Template para el prompt de generación Q&A.
    """
    system_prompt = (
        'A partir del siguiente párrafo, genera entre 1 y 3 pares de pregunta y respuesta.\n\n'
        'Reglas:\n'
        '- Varía los tipos de pregunta: factuales, comparativas, de definición, causales y de resumen.\n'
        '- Las respuestas deben ser detalladas y en español.\n'
        '- Usa SOLO información del párrafo, no inventes datos.\n\n'
        'Párrafo:\n{text}'
    )
    return ChatPromptTemplate.from_template(system_prompt)


def create_translate_prompt_es() -> ChatPromptTemplate:
    """
    Create a prompt template for converting tables and statistics into a human-readable format.
    Returns:
        ChatPromptTemplate: A template for the translation prompt.
    """
    system_prompt = (
        'Reescribe el siguiente contenido como texto corrido en español.\n'
        'Si contiene tablas, conviértelas a texto descriptivo.\n'
        'Conserva todos los datos y estadísticas.\n'
        'No agregues introducciones como "El resumen es...", "A continuación..." o similares.\n\n'
        'Contenido:\n{text}\n'
    )
    return ChatPromptTemplate.from_template(system_prompt)


def create_summarize_prompt_es() -> ChatPromptTemplate:
    """
    Crea un template de prompt para resumir contenido con tablas.

    Returns:
        ChatPromptTemplate: Template para el prompt de resumen.
    """
    system_prompt = (
        'Sigue las instrucciones para procesar el contenido:\n'
        '1. Si el contenido es una tabla, extrae una lista con la relación de todas las celdas. \
            No consideres las celdas que no contengan información.\n'
        '2. Proporciona el resultado en formato markdown en formato de lista.\n'
        'Contenido:\n{text}\n'
    )
    return ChatPromptTemplate.from_template(system_prompt)


def create_reasoning_prompt_es() -> ChatPromptTemplate:
    """
    Crea un template de prompt para generar respuestas de razonamiento en español.

    Returns:
        ChatPromptTemplate: Template para el prompt de razonamiento.
    """
    system_prompt = (
        'Eres un experto en resolución de problemas. Para cada problema:\n\n'
        'ESTRUCTURA TU RESPUESTA EXACTAMENTE ASÍ:\n\n'
        
        '<think>\n'
        'Déjame resolver esto paso a paso...\n\n'
        '[Tu proceso de razonamiento detallado aquí - desglosa el problema, '
        'identifica la información clave, trabaja la lógica sistemáticamente.]\n'
        '</think>\n\n'
        '[Proporciona una respuesta directa sin explicaciones adicionales. '
        'Simplemente indica la respuesta final de forma clara y concisa.]\n\n'
        'REQUISITOS:\n'
        '- Siempre comienza con la sección <think>\n'
        '- Proporciona una respuesta directa sin explicaciones\n'
        '- Indica la respuesta final de forma clara y concisa\n\n'
        'Problema: {question}\n'
        'Solución esperada: {answer}\n'
        'Fuente: {source}'
    )
    return ChatPromptTemplate.from_template(system_prompt)


def create_rag_questions_prompt_es(num_questions: int) -> ChatPromptTemplate:
    """
    Crea un template de prompt para generar preguntas de validación RAG
    a partir de un chunk de texto en español.

    Args:
        num_questions (int): Número de preguntas a generar.

    Returns:
        ChatPromptTemplate: Template para el prompt de generación de preguntas RAG.
    """
    system_prompt = (
        f'A partir del siguiente fragmento de un documento, genera exactamente {num_questions} '
        'preguntas que un usuario podría hacer a un sistema de búsqueda.\n\n'
        'Reglas:\n'
        '- Cada pregunta debe poder responderse EXCLUSIVAMENTE con la información del fragmento.\n'
        '- Varía los tipos de pregunta: factuales, comparativas, de definición, causales y de resumen.\n'
        '- Las preguntas deben ser naturales, como las haría un usuario real.\n'
        '- Las preguntas deben estar en español.\n'
        '- No repitas preguntas ni hagas variaciones de la misma pregunta.\n\n'
        'Fragmento:\n{{text}}'
    )
    return ChatPromptTemplate.from_template(system_prompt)


def create_rag_variants_prompt_es(num_variants: int) -> ChatPromptTemplate:
    """
    Crea un template de prompt para generar variantes (reescrituras semánticas)
    de una pregunta en español.

    Args:
        num_variants (int): Número de variantes a generar.

    Returns:
        ChatPromptTemplate: Template para el prompt de generación de variantes.
    """
    system_prompt = (
        f'Reescribe la siguiente pregunta de {num_variants} formas diferentes, '
        'manteniendo exactamente el mismo significado.\n\n'
        'Reglas:\n'
        '- Cada variante debe ser una forma distinta de hacer la misma pregunta.\n'
        '- Usa sinónimos, cambia la estructura gramatical o el orden de las palabras.\n'
        '- Mantén el mismo nivel de formalidad y detalle.\n'
        '- Las variantes deben estar en español.\n\n'
        'Pregunta original:\n{{question}}'
    )
    return ChatPromptTemplate.from_template(system_prompt)
