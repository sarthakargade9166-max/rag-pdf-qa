# 📄 PDF Q&A with RAG — Portfolio Project

A minimal, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload a PDF, ask natural-language questions about it, and get grounded answers backed by source citations.

Built as a placement portfolio project to demonstrate practical ML engineering skills.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────────────────────────────────────┐
│  Streamlit   │     │              rag_pipeline.py                    │
│   (app.py)   │────▶│                                                 │
│              │     │  PDF ─▶ Chunks ─▶ Embeddings ─▶ ChromaDB       │
│  Upload PDF  │     │                                                 │
│  Ask Question│────▶│  Question ─▶ Embed ─▶ Retrieve ─▶ LLM Answer  │
└──────────────┘     └─────────────────────────────────────────────────┘
```

**Pipeline stages:**

| # | Stage | Tool | Why |
|---|-------|------|-----|
| 1 | PDF text extraction | PyMuPDFLoader | Fast, preserves page metadata for citations |
| 2 | Text chunking | RecursiveCharacterTextSplitter | Splits on natural boundaries (¶ → sentence → word) |
| 3 | Embedding | all-MiniLM-L6-v2 | Free, local, no API key — 384-dim vectors |
| 4 | Vector storage | ChromaDB | Lightweight, persists to disk, no server needed |
| 5 | Retrieval | Cosine similarity (top-4) | Finds the most relevant chunks for the question |
| 6 | Generation | Groq (Llama 3.1 8B) | Fast inference, free tier available |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier is sufficient)

### 1. Clone / download the project

```bash
cd rag-pdf-qa
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

> **Note:** The first run will download the embedding model (~80 MB). This is a one-time download.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📂 File Structure

```
rag-pdf-qa/
├── app.py              # Streamlit UI — wires the pipeline together
├── rag_pipeline.py     # All RAG logic in one file
├── requirements.txt    # Dependencies
├── README.md           # This file
├── .env                # Your API key (not committed to git)
└── chroma_db/          # Auto-created — persisted vector store
```

**That's it — 4 files.** No config files, no test suite, no Docker.

---

## 🔧 Swapping the LLM

The LLM call is isolated behind one function: `get_answer(context, question)`. To switch from Groq to OpenAI:

1. `pip install langchain-openai`
2. In `rag_pipeline.py`, replace `ChatGroq` with `ChatOpenAI`
3. Change the env var from `GROQ_API_KEY` to `OPENAI_API_KEY`
4. Set `model="gpt-4o"` (or any OpenAI model)

See the inline comments in `rag_pipeline.py` → `_get_llm()` for details.

---

## 🎯 Skills Demonstrated

This project is designed to showcase the following in interviews:

| Skill | How it's used |
|-------|---------------|
| **LangChain** | Orchestrates the full RAG pipeline — document loading, text splitting, embeddings, vector store integration, and LLM calls |
| **Vector Databases (ChromaDB)** | Stores and retrieves document embeddings with cosine similarity search; persistent storage to disk |
| **Embeddings** | Uses sentence-transformers (all-MiniLM-L6-v2) for dense vector representations; understands the role of embedding dimension, normalization, and similarity metrics |
| **Retrieval-Augmented Generation** | Implements the complete RAG pattern: retrieve → augment prompt → generate; understands grounding and hallucination prevention |
| **Prompt Engineering** | Constructs structured system + user prompts that constrain the LLM to answer only from retrieved context |
| **Text Chunking Strategies** | Applies recursive character splitting with overlap to preserve semantic coherence across chunk boundaries |
| **API Design** | Clean separation of concerns — all RAG logic in one module, UI in another; single swappable function for the LLM provider |
| **Error Handling** | Graceful handling of empty PDFs, missing API keys, and no-results scenarios |
| **Streamlit** | Interactive UI with file upload, real-time status indicators, expandable debug views, and source citations |

---

## 🗣️ Interview Talking Points

**"Walk me through the RAG pipeline."**
> "The user uploads a PDF. I extract text with PyMuPDF, which preserves page numbers. I split the text into ~600-token chunks with 15% overlap so no information is lost at boundaries. Each chunk is embedded into a 384-dimensional vector using a local sentence-transformer model. These vectors are stored in ChromaDB on disk. When the user asks a question, I embed the question with the same model, retrieve the 4 most similar chunks via cosine similarity, inject them into a system prompt as context, and send it to an LLM. The prompt explicitly constrains the model to answer only from the provided context, which prevents hallucination."

**"Why these specific parameters?"**
> "600 tokens per chunk balances specificity (small enough to be about one topic) with enough context for the LLM to work with. 15% overlap means adjacent chunks share ~90 tokens, so a concept that spans a chunk boundary still appears in at least one chunk. Top-4 retrieval gives the LLM enough context without hitting token limits."

**"How would you improve this for production?"**
> "I'd add hybrid search (BM25 + dense retrieval), use a re-ranker like Cohere Rerank on the retrieved chunks, implement streaming responses, add conversation memory for follow-up questions, and swap ChromaDB for a managed vector DB like Pinecone or Weaviate for scalability."

---

## 📝 License

MIT — use freely for your own portfolio.
