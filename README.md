# Advanced RAG Chatbot

This project is a Streamlit-based Hybrid RAG chatbot for querying uploaded PDF documents. It combines semantic search, keyword search, reranking, LLM generation, chat memory, and basic security guardrails.

## What This Project Does

- Upload one or more PDF files from the UI
- Split documents into chunks
- Generate embeddings using `sentence-transformers`
- Store vectors in `FAISS`
- Retrieve relevant chunks using:
  - vector search
  - keyword search with BM25
- Rerank the retrieved chunks with a cross-encoder
- Generate an answer using Groq LLM
- Fall back to general LLM knowledge when the question is unrelated to the uploaded documents
- Block or sanitize sensitive information such as API keys, passwords, tokens, and secrets

## Tech Stack

- UI: Streamlit
- LLM: Groq `llama-3.1-8b-instant`
- Embeddings: `sentence-transformers` with `all-MiniLM-L6-v2`
- Vector Store: FAISS
- Keyword Retrieval: BM25 via `rank_bm25`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- PDF Loader: LangChain `PyPDFLoader`

## Project Type

This is currently a `Hybrid RAG` system with `LLM fallback`.

Why:
- It uses both vector retrieval and keyword retrieval
- The pipeline is fixed in code
- It is not fully agentic yet because the system does not autonomously choose between multiple tools like web search, SQL, or clarification prompts

## End-to-End Flow

### 1. Upload and Indexing Flow

1. User uploads one or more PDF files from the Streamlit UI
2. Files are saved inside the `uploads/` folder
3. PDFs are loaded using `PyPDFLoader`
4. Documents are split into chunks using `RecursiveCharacterTextSplitter`
5. Each chunk is converted into embeddings using `SentenceTransformer`
6. Embeddings are stored in a FAISS index
7. The FAISS index and chunked docs are saved to `faiss_store.pkl`
8. On server restart, if `faiss_store.pkl` already exists, it is auto-loaded

### 2. Question Answering Flow

1. User asks a question in the UI
2. Query is checked against security guardrails
3. Hybrid retrieval runs:
   - vector retrieval from FAISS
   - keyword retrieval with BM25
4. Results are merged and duplicates are removed
5. Retrieved chunks are reranked using a cross-encoder
6. If relevant document context exists:
   - context is sanitized for secrets
   - prompt is built using document context and recent chat history
   - LLM answers using RAG
7. If relevant document context does not exist:
   - the app falls back to general LLM knowledge
   - the response clearly states that the answer is not based on uploaded documents
8. Final response is checked again for sensitive content before being shown

## Architecture Flow

```text
User
  ->
Streamlit UI
  ->
PDF Upload / Question Input
  ->
API Layer
  ->
Guardrails
  ->
Hybrid Retrieval
  -> Vector Search (FAISS)
  -> Keyword Search (BM25)
  ->
Reranker
  ->
Prompt Builder
  ->
Groq LLM
  ->
Response Guardrails
  ->
UI Response
```

## Folder Structure

```text
Advanced-Rag/
|
|-- app.py                     # Streamlit UI
|-- requirements.txt          # Python dependencies
|-- faiss_store.pkl           # Saved FAISS index + documents
|-- .env                      # Environment variables
|
|-- api/
|   |-- main.py               # Main orchestration flow
|
|-- config/
|   |-- settings.py           # API key and model settings
|
|-- ingestion/
|   |-- loader.py             # PDF loading
|   |-- chunker.py            # Document chunking
|   |-- embedder.py           # Embedding generation
|   |-- vector_store.py       # FAISS storage and search
|
|-- retrieval/
|   |-- vector_retriever.py   # Vector retrieval wrapper
|   |-- keyword_retriever.py  # BM25 retrieval
|   |-- hybrid_retriever.py   # Combined retrieval
|   |-- reranker.py           # Cross-encoder reranking
|   |-- metadata_filter.py    # Placeholder for metadata filtering
|
|-- llm/
|   |-- prompt.py             # Prompt template
|   |-- generator.py          # Groq response generation
|   |-- guardrails.py         # Query/output security filtering
|
|-- memory/
|   |-- chat_memory.py        # Stores recent chat history
|
|-- uploads/                  # Uploaded PDF files
```

## Core Modules

### `app.py`
- Streamlit frontend
- Uploads multiple PDFs
- Triggers indexing
- Sends user questions to backend
- Displays chat history

### `api/main.py`
- Main orchestration file
- Builds the knowledge base
- Loads saved FAISS store on restart
- Handles chat flow
- Chooses between:
  - RAG answer
  - General LLM fallback

### `ingestion/embedder.py`
- Uses `SentenceTransformer("all-MiniLM-L6-v2")`
- Converts document chunks and queries into embeddings

### `ingestion/vector_store.py`
- Builds FAISS index with `IndexFlatL2`
- Saves and loads `faiss_store.pkl`
- Returns no vector matches if similarity is too weak

### `retrieval/hybrid_retriever.py`
- Combines:
  - FAISS vector retrieval
  - BM25 keyword retrieval
- Removes duplicate chunks

### `retrieval/reranker.py`
- Uses cross-encoder reranking
- Scores each `(query, chunk)` pair
- Returns the top most relevant chunks

### `llm/guardrails.py`
- Blocks risky user prompts like:
  - API key requests
  - password requests
  - token requests
  - secret extraction attempts
- Removes sensitive lines from retrieved context
- Blocks unsafe final outputs

## Security Guardrails

This project includes a raw regex-based security layer.

It currently protects against:
- API keys
- passwords
- tokens
- secrets
- credentials
- private keys
- connection strings
- `.env`-style secret requests

Protection happens in 3 stages:
- query blocking
- context sanitization
- response blocking

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run the App

```bash
streamlit run app.py
```

## How Persistence Works

- When files are processed, the FAISS index and chunk list are saved into `faiss_store.pkl`
- On restart, the app checks if that file already exists
- If it exists, the knowledge base is loaded automatically
- This avoids reprocessing the same PDFs every time the server restarts

## Current Limitations

- Only PDF upload is supported right now
- Metadata filtering is not fully implemented
- Guardrails are regex-based, not enterprise-grade DLP
- Retrieval scoring is simple and can be improved further
- The system is hybrid RAG, not fully agentic RAG

## Future Improvements

- Add citations with file name and page number
- Add `.docx`, `.txt`, and `.csv` support
- Add metadata-based filtering
- Improve retrieval score fusion
- Add ingestion-time redaction for sensitive data
- Add agentic routing:
  - document retrieval
  - direct LLM answer
  - web search
  - clarification questions
- Add evaluation dataset and metrics

## Summary

This project is a practical Hybrid RAG chatbot that supports:
- multi-PDF upload
- semantic plus keyword retrieval
- reranking
- LLM fallback for unrelated questions
- persistence with FAISS
- basic security guardrails

It is a strong base for extending into a more advanced or agentic RAG system.
