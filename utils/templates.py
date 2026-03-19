from langchain_core.prompts import ChatPromptTemplate


def create_q_and_a_prompt_es() -> ChatPromptTemplate:
    """
    Creates a prompt template for generating question and answer pairs from a given paragraph
    in Spanish.
    Returns:
        ChatPromptTemplate: A template for the Q&A generation prompt.
    """
    system_prompt = (
        'A partir del siguiente párrafo, genera entre 1 y 3 pares de pregunta y respuesta.\n\n'
        'Reglas:\n'
        '- Varía los tipos de pregunta: factuales, comparativas, de definición, causales y de resumen.\n'
        '- Las respuestas deben ser detalladas y en español.\n'
        '- Usa SOLO información del párrafo, no inventes datos.\n'
        '- Responde en formato JSON: una lista de objetos con claves "prompt" y "completion".\n\n'
        'Ejemplo de formato:\n'
        '[{{"prompt": "¿Cuál es la población de Acatic?", "completion": "La población de Acatic es de 22,986 habitantes según el censo 2020."}}, '
        '{{"prompt": "¿Cómo se compara la tasa de marginación de Acatic con el promedio estatal?", "completion": "Acatic tiene un grado de marginación medio, por encima del promedio estatal que es bajo."}}, '
        '{{"prompt": "¿Qué significa el nombre Acatic?", "completion": "El nombre Acatic proviene de la voz náhuatl laka-ti-k que significa Entre las cañas o Lugar de muchas cañas."}}]\n\n'
        'Párrafo:\n{text}\n\n'
        'JSON:\n'
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
