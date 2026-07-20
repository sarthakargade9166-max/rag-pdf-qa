"""
rag_pipeline.py — Complete RAG Pipeline
========================================
All retrieval-augmented generation logic lives here:
  1. PDF text extraction (PyMuPDFLoader)
  2. Text chunking (RecursiveCharacterTextSplitter)
  3. Embedding generation (HuggingFace sentence-transformers)
  4. Vector storage & retrieval (ChromaDB)
  5. LLM answer generation (Groq — swappable)
"""

import os
import tempfile
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"                       # local persistence path
EMBEDDING_MODEL = "all-MiniLM-L6-v2"            # free, local, no API key
LLM_MODEL = "llama-3.1-8b-instant"              # Groq-hosted, fast & free-tier
CHUNK_SIZE = 2400                                # ~600 tokens (1 token ≈ 4 chars)
CHUNK_OVERLAP = 360                              # ~15% of chunk_size


# ===========================================================================
# Stage 1 — PDF Loading
# ===========================================================================

def load_pdf(uploaded_file) -> list:
    """
    Extract text + metadata from an uploaded PDF.

    Uses PyMuPDFLoader (built on the `fitz` library) which preserves
    page numbers in each Document's metadata — useful for citations.

    Args:
        uploaded_file: A Streamlit UploadedFile object (bytes-like).

    Returns:
        List of LangChain Document objects, one per page.

    Raises:
        ValueError: If the PDF yields zero text (scanned image, corrupt, etc.)
    """
    # PyMuPDFLoader needs a file path, so write the upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)  # clean up the temp file

    # Edge case: empty or image-only PDF
    if not documents or all(doc.page_content.strip() == "" for doc in documents):
        raise ValueError(
            "The PDF appears to be empty or contains only scanned images. "
            "This pipeline requires text-based PDFs."
        )

    return documents


# ===========================================================================
# Stage 2 — Text Chunking
# ===========================================================================

def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split along
    natural boundaries (paragraphs → sentences → words) before falling
    back to raw character splits. This keeps semantic units intact.

    Chunk size ~600 tokens with ~15% overlap ensures:
      - Enough context per chunk for meaningful retrieval
      - Overlap prevents information loss at chunk boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,              # character-based length
        separators=["\n\n", "\n", ". ", " ", ""],  # split hierarchy
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ===========================================================================
# Stage 3 — Embedding Model
# ===========================================================================

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformer embedding model.

    all-MiniLM-L6-v2 produces 384-dim vectors, runs locally on CPU,
    and requires no API key — ideal for portfolio projects.
    The model is downloaded & cached on first run (~80 MB).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )


# ===========================================================================
# Stage 4 — Vector Store (ChromaDB)
# ===========================================================================

def _collection_name(file_name: str) -> str:
    """Deterministic collection name from the PDF filename."""
    safe = hashlib.md5(file_name.encode()).hexdigest()[:16]
    return f"pdf_{safe}"


def create_vector_store(chunks: list, embedding_model, file_name: str) -> Chroma:
    """
    Embed chunks and persist them in ChromaDB.

    Each PDF gets its own Chroma collection (keyed by filename hash)
    so re-uploading the same file reuses the existing index.

    ChromaDB stores vectors + raw text + metadata on disk under ./chroma_db.
    """
    collection = _collection_name(file_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection,
        persist_directory=CHROMA_DIR,
    )
    return vector_store


def load_vector_store(embedding_model, file_name: str) -> Chroma:
    """Reload a previously persisted Chroma collection."""
    collection = _collection_name(file_name)
    return Chroma(
        collection_name=collection,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )


# ===========================================================================
# Stage 5 — Retrieval
# ===========================================================================

def retrieve_chunks(query: str, vector_store: Chroma, k: int = 4) -> list:
    """
    Embed the user's question and find the top-k most similar chunks.

    Uses cosine similarity (via normalized embeddings) to rank chunks.
    Returns LangChain Document objects with .page_content and .metadata.

    Args:
        query:        The user's natural-language question.
        vector_store: The Chroma vector store to search.
        k:            Number of chunks to retrieve (default 4).

    Returns:
        List of (Document, score) tuples, sorted by relevance.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


# ===========================================================================
# Stage 6 — LLM Answer Generation
# ===========================================================================

def _get_llm() -> ChatGroq:
    """
    Initialize the LLM client.

    Reads the API key from the GROQ_API_KEY environment variable.
    Uses Groq's hosted Llama 3.1 8B — fast inference, generous free tier.

    ---------------------------------------------------------------
    TO SWAP LLMs (e.g., to OpenAI GPT-4o):
      1. pip install langchain-openai
      2. Replace ChatGroq with ChatOpenAI
      3. Change the env var to OPENAI_API_KEY
      4. Set model_name="gpt-4o"
    ---------------------------------------------------------------
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Set it in your .env file or environment:\n"
            "  export GROQ_API_KEY=gsk_..."
        )

    return ChatGroq(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.2,        # low temp → more factual, less creative
        max_tokens=1024,
    )


def get_answer(context: str, question: str) -> str:
    """
    The single entrypoint for LLM generation.

    Takes pre-formatted context (retrieved chunks) and the user's question,
    builds a grounded prompt, and returns the LLM's answer.

    The system prompt instructs the model to:
      - Only use the provided context
      - Admit when the context doesn't contain the answer
      - Cite page numbers when available
    """
    system_prompt = (
        "You are a precise, helpful assistant that answers questions based ONLY "
        "on the provided context from a PDF document.\n\n"
        "Rules:\n"
        "1. Answer using ONLY information found in the context below.\n"
        "2. If the context does not contain enough information to answer, "
        "   say: 'I couldn't find enough information in the document to answer this.'\n"
        "3. When referencing information, mention the page number if available.\n"
        "4. Be concise but thorough."
    )

    user_prompt = (
        f"CONTEXT (retrieved from the PDF):\n"
        f"---\n{context}\n---\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return response.content


def format_context(results: list) -> str:
    """
    Format retrieved chunks into a single context string for the LLM.

    Includes page numbers from PyMuPDFLoader metadata so the LLM
    can cite sources in its answer.
    """
    if not results:
        return ""

    context_parts = []
    for doc, score in results:
        page = doc.metadata.get("page", "unknown")
        # PyMuPDFLoader uses 0-indexed pages; display as 1-indexed
        page_display = page + 1 if isinstance(page, int) else page
        context_parts.append(
            f"[Page {page_display}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)
