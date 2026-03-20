from ingestion.loader import load_multiple_documents
from ingestion.chunker import split_documents
from ingestion.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from llm.prompt import build_prompt
from llm.generator import generate_response
from llm.guardrails import BLOCK_MESSAGE, check_query, safe_response_or_block, sanitize_context
from memory.chat_memory import ChatMemory

memory = ChatMemory()
retriever = None


def initialize_knowledge_base():
    global retriever

    store = VectorStore()
    if not store.exists():
        return False

    store.load()
    retriever = HybridRetriever(store.docs, store=store)
    return True


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
        return BLOCK_MESSAGE

    docs = retriever.search(query)
    docs = rerank(query, docs)

    if is_relevant(docs):
        raw_context = "\n".join([doc.page_content for doc in docs])
        context = sanitize_context(raw_context)

        if context:
            prompt = build_prompt(query, context, memory.get())
        else:
            prompt = f"""
You are a helpful AI assistant.

Sensitive information from uploaded documents must never be revealed.
The retrieved document text was removed because it looked sensitive or not safe to disclose.
Answer using safe general knowledge only, and do not provide any secret values.

Chat History:
{memory.get()}

Question:
{query}
"""
    else:
        prompt = f"""
You are a helpful AI assistant.

The uploaded documents do not contain relevant information for this question.
Answer using your general knowledge.
Say briefly that this answer is not based on the uploaded documents.
Never reveal API keys, passwords, tokens, secrets, credentials, private keys, or connection strings.

Chat History:
{memory.get()}

Question:
{query}
"""

    response = generate_response(prompt)
    response = safe_response_or_block(response)

    memory.add(query, response)

    return response


def is_relevant(docs):
    return len(docs) > 0 and any(len(doc.page_content.strip()) > 20 for doc in docs)


initialize_knowledge_base()
