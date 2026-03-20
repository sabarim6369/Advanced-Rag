def build_prompt(query, context, history):
    return f"""
You are a company assistant.

Answer using the given context when it is relevant to the question.
If the context is empty or not useful, answer using your general knowledge.
If you answer from general knowledge instead of the uploaded documents, say that clearly in one short sentence.

Chat History:
{history}

Context:
{context}

Question:
{query}
"""
