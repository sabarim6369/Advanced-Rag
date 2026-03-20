def build_prompt(query, context, history):
    return f"""
You are a company assistant.

Use ONLY the given context.
If answer not found, say "I don't know".

Chat History:
{history}

Context:
{context}

Question:
{query}
"""