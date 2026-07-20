import os
import tempfile
import hashlib

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 360


def load_pdf(uploaded_file) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    if not documents or all(doc.page_content.strip() == "" for doc in documents):
        raise ValueError(
            "The PDF appears to be empty or contains only scanned images. "
            "This pipeline requires text-based PDFs."
        )

    return documents


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _collection_name(file_name: str) -> str:
    safe = hashlib.md5(file_name.encode()).hexdigest()[:16]
    return f"pdf_{safe}"


def create_vector_store(chunks: list, embedding_model, file_name: str) -> Chroma:
    collection = _collection_name(file_name)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection,
        persist_directory=CHROMA_DIR,
    )


def load_vector_store(embedding_model, file_name: str) -> Chroma:
    collection = _collection_name(file_name)
    return Chroma(
        collection_name=collection,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )


def retrieve_chunks(query: str, vector_store: Chroma, k: int = 4) -> list:
    return vector_store.similarity_search_with_score(query, k=k)


def _get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Set it in your .env file or environment:\n"
            "  export GROQ_API_KEY=gsk_..."
        )

    return ChatGroq(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=1024,
    )


def get_answer(context: str, question: str) -> str:
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
    if not results:
        return ""

    context_parts = []
    for doc, score in results:
        page = doc.metadata.get("page", "unknown")
        page_display = page + 1 if isinstance(page, int) else page
        context_parts.append(f"[Page {page_display}]\n{doc.page_content}")

    return "\n\n".join(context_parts)
