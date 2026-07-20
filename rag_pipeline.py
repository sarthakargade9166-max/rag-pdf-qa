import os
import hashlib
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# config
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 2400    # roughly 600 tokens
CHUNK_OVERLAP = 360  # ~15 percent overlap


def load_pdf(file):
    import tempfile

    pdf_bytes = file.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        pdf = fitz.open(tmp_path)
        docs = []
        for i in range(len(pdf)):
            text = pdf[i].get_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": i}))
        pdf.close()
    finally:
        os.unlink(tmp_path)

    if not docs:
        raise ValueError(
            "This PDF has no extractable text. "
            "Try a different PDF where you can select/copy text."
        )

    return docs


def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def get_collection_name(filename):
    # hash the filename so we get a safe collection name
    h = hashlib.md5(filename.encode()).hexdigest()[:16]
    return f"pdf_{h}"


def store_in_chroma(chunks, embed_model, filename):
    name = get_collection_name(filename)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        collection_name=name,
        persist_directory=CHROMA_PATH,
    )
    return db


def load_from_chroma(embed_model, filename):
    name = get_collection_name(filename)
    db = Chroma(
        collection_name=name,
        embedding_function=embed_model,
        persist_directory=CHROMA_PATH,
    )
    return db


def search_similar(query, db, k=4):
    results = db.similarity_search_with_score(query, k=k)
    return results


def build_context(results):
    if not results:
        return ""

    parts = []
    for doc, score in results:
        page = doc.metadata.get("page", "unknown")
        if isinstance(page, int):
            page = page + 1  # pymupdf pages are 0-indexed
        parts.append(f"[Page {page}]\n{doc.page_content}")

    return "\n\n".join(parts)


def get_llm():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Set it in your .env file or as an environment variable."
        )

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=key,
        temperature=0.2,
        max_tokens=1024,
    )
    return llm


def get_answer(context, question):
    sys_msg = (
        "You are a helpful assistant that answers questions based ONLY on the "
        "provided context from a PDF document.\n\n"
        "Rules:\n"
        "1. Only use information from the context below.\n"
        "2. If the context doesn't have enough info, say so.\n"
        "3. Mention page numbers when possible.\n"
        "4. Keep answers concise."
    )

    user_msg = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {question}\n\nANSWER:"

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_msg),
    ])

    return resp.content
