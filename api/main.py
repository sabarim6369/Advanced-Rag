from ingestion.loader import load_documents
from ingestion.chunker import split_documents
from ingestion.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from llm.prompt import build_prompt
from llm.generator import generate_response
from llm.guardrails import check_query
from memory.chat_memory import ChatMemory

memory = ChatMemory()

# load + build (run once)
docs = load_documents("company.pdf")
chunks = split_documents(docs)

store = VectorStore()
store.build(chunks)

retriever = HybridRetriever(chunks)

def chat(query):
    if not check_query(query):
        return "Blocked due to security policy"

    docs = retriever.search(query)
    docs = rerank(query, docs)

    context = "\n".join([d.page_content for d in docs])

    prompt = build_prompt(query, context, memory.get())

    response = generate_response(prompt)

    memory.add(query, response)

    return response