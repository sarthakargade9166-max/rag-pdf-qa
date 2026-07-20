# PDF Q&A with RAG

A minimal, end-to-end Retrieval-Augmented Generation (RAG) pipeline that lets you upload a PDF, ask natural-language questions about it, and get grounded answers backed by source citations.

---

## Architecture

```
┌──────────────┐     ┌─────────────────────────────────────────────────┐
│  Streamlit   │     │              rag_pipeline.py                    │
│   (app.py)   │────▶│                                                 │
│              │     │  PDF ─▶ Chunks ─▶ Embeddings ─▶ ChromaDB        │
│  Upload PDF  │     │                                                 │
│  Ask Question│────▶│  Question ─▶ Embed ─▶ Retrieve ─▶ LLM Answer    │
└──────────────┘     └─────────────────────────────────────────────────┘
```

**Pipeline stages:**

| # | Stage | Tool | Description |
|---|-------|------|-------------|
| 1 | PDF text extraction | PyMuPDF | Fast extraction, preserves page metadata for citations |
| 2 | Text chunking | RecursiveCharacterTextSplitter | Splits on natural boundaries (paragraphs -> sentences -> words) |
| 3 | Embedding | all-MiniLM-L6-v2 | Local sentence-transformer model (384-dimensional vectors) |
| 4 | Vector storage | ChromaDB | Lightweight, persists to disk, no server needed |
| 5 | Retrieval | Cosine similarity (top-4) | Finds the most relevant chunks for the user's question |
| 6 | Generation | Groq (Llama 3.1 8B) | Fast inference for generating the final answer |

---

## Setup & Run

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

### 1. Clone / download the project

```bash
cd rag-pdf-qa
```

### 2. Create a virtual environment

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

## File Structure

```
rag-pdf-qa/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Core RAG logic 
├── requirements.txt    # Project dependencies
├── README.md           # This file
├── .env                # Environment variables (API key)
└── chroma_db/          # Auto-created persistent vector store
```

---

## Swapping the LLM

The LLM call is isolated behind one function: `get_answer(context, question)`. To switch from Groq to OpenAI:

1. `pip install langchain-openai`
2. In `rag_pipeline.py`, replace `ChatGroq` with `ChatOpenAI`
3. Change the environment variable from `GROQ_API_KEY` to `OPENAI_API_KEY`
4. Set `model="gpt-4o"` (or any other OpenAI model)
