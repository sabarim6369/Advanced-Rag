from ingestion.loader import load_multiple_documents
from ingestion.chunker import split_documents
from ingestion.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from llm.prompt import build_prompt
from llm.generator import generate_response
from llm.guardrails import check_query
from memory.chat_memory import ChatMemory

memory = ChatMemory()
retriever = None


def build_knowledge_base(file_paths):
    global memory, retriever

    documents = load_multiple_documents(file_paths)
    chunks = split_documents(documents)

    store = VectorStore()
    store.build(chunks)

    retriever = HybridRetriever(chunks, store=store)
    memory = ChatMemory()

    return len(documents), len(chunks)


def chat(query):
    if retriever is None:
        return "Upload and process at least one PDF before asking a question."

    if not check_query(query):
        return "Blocked due to security policy"

    docs = retriever.search(query)
    docs = rerank(query, docs)

    if is_relevant(docs):
        context = "\n".join([doc.page_content for doc in docs])
        prompt = build_prompt(query, context, memory.get())
    else:
        prompt = f"""
You are a helpful AI assistant.

The uploaded documents do not contain relevant information for this question.
Answer using your general knowledge.
Say briefly that this answer is not based on the uploaded documents.

Chat History:
{memory.get()}

Question:
{query}
"""

    response = generate_response(prompt)

    memory.add(query, response)

    return response


def is_relevant(docs):
    return len(docs) > 0 and any(len(doc.page_content.strip()) > 20 for doc in docs)
