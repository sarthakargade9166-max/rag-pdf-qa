import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_pdf,
    chunk_documents,
    get_embedding_model,
    create_vector_store,
    load_vector_store,
    retrieve_chunks,
    format_context,
    get_answer,
)

# Load environment variables from .env file (for the LLM API key)
load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF Q&A — RAG Pipeline",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom styling for a clean, portfolio-worthy look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main container */
    .block-container { max-width: 760px; padding-top: 2rem; }

    /* Header styling */
    h1 { color: #1a1a2e; }

    /* Source citation badges */
    .source-badge {
        display: inline-block;
        background: #e8eaf6;
        color: #3949ab;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 2px 4px 2px 0;
        font-weight: 500;
    }

    /* Answer container */
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #3949ab;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        line-height: 1.6;
    }

    /* Pipeline stage indicators */
    .stage-label {
        color: #666;
        font-size: 0.85em;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.title("📄 PDF Q&A with RAG")
st.caption(
    "Upload a PDF, ask questions, get answers grounded in the document. "
    "Powered by LangChain · sentence-transformers · ChromaDB · Groq"
)

st.divider()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "page_count" not in st.session_state:
    st.session_state.page_count = 0

# ---------------------------------------------------------------------------
# Sidebar — PDF Upload & Processing
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a text-based PDF (scanned images won't work).",
    )

    if uploaded_file is not None:
        # Only re-process if a new file is uploaded
        if st.session_state.file_name != uploaded_file.name:
            with st.status("Processing PDF...", expanded=True) as status:

                # --- Stage 1: Load PDF ---
                st.write("📖 Extracting text from PDF...")
                try:
                    documents = load_pdf(uploaded_file)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                page_count = len(documents)
                st.write(f"✅ Extracted text from {page_count} pages")

                # --- Stage 2: Chunk text ---
                st.write("✂️ Splitting into chunks...")
                chunks = chunk_documents(documents)
                chunk_count = len(chunks)
                st.write(f"✅ Created {chunk_count} chunks")

                # --- Stage 3: Generate embeddings & store ---
                st.write("🧮 Generating embeddings (first run downloads the model)...")
                embedding_model = get_embedding_model()

                # --- Stage 4: Persist to ChromaDB ---
                st.write("💾 Storing in ChromaDB...")
                vector_store = create_vector_store(
                    chunks, embedding_model, uploaded_file.name
                )
                st.write("✅ Indexed and ready!")

                status.update(label="✅ PDF processed!", state="complete")

            # Save state
            st.session_state.processed = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.chunk_count = chunk_count
            st.session_state.page_count = page_count

    # Show current document info
    if st.session_state.processed:
        st.divider()
        st.subheader("📊 Document Info")
        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.page_count)
        col2.metric("Chunks", st.session_state.chunk_count)
        st.caption(f"**File:** {st.session_state.file_name}")

# ---------------------------------------------------------------------------
# Main area — Question & Answer
# ---------------------------------------------------------------------------

if not st.session_state.processed:
    # Empty state — prompt the user
    st.info("👈 Upload a PDF in the sidebar to get started.", icon="📤")
    st.stop()

# Question input
question = st.text_input(
    "Ask a question about the document:",
    placeholder="e.g., What are the main findings of this paper?",
)

if question:
    with st.spinner("🔍 Searching document and generating answer..."):
        try:
            # --- Stage 5: Retrieve relevant chunks ---
            embedding_model = get_embedding_model()
            vector_store = load_vector_store(
                embedding_model, st.session_state.file_name
            )
            results = retrieve_chunks(question, vector_store, k=4)

            # Edge case: no relevant chunks found
            if not results:
                st.warning(
                    "No relevant sections found in the document for this question. "
                    "Try rephrasing or asking something more specific.",
                    icon="🤷",
                )
                st.stop()

            # --- Stage 6: Format context and get LLM answer ---
            context = format_context(results)
            answer = get_answer(context, question)

        except EnvironmentError as e:
            # Missing API key
            st.error(
                f"⚠️ {str(e)}\n\n"
                "Create a `.env` file in the project root with:\n"
                "```\nGROQ_API_KEY=gsk_your_key_here\n```"
            )
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            st.stop()

    # --- Display the answer ---
    st.subheader("💡 Answer")
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True,
    )

    # --- Show source pages (from PyMuPDFLoader metadata) ---
    st.subheader("📑 Sources")
    source_pages = set()
    for doc, score in results:
        page = doc.metadata.get("page", None)
        if page is not None:
            source_pages.add(page + 1)  # 0-indexed → 1-indexed

    if source_pages:
        badges = " ".join(
            f'<span class="source-badge">Page {p}</span>'
            for p in sorted(source_pages)
        )
        st.markdown(badges, unsafe_allow_html=True)
    else:
        st.caption("Page numbers not available in metadata.")

    # --- Expandable: show the raw retrieved chunks ---
    with st.expander("🔎 View retrieved chunks (debug)"):
        for i, (doc, score) in enumerate(results, 1):
            page = doc.metadata.get("page", "?")
            page_display = page + 1 if isinstance(page, int) else page
            st.markdown(f"**Chunk {i}** — Page {page_display} — Similarity: `{score:.4f}`")
            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            st.divider()
