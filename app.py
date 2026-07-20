import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import (
    load_pdf, split_into_chunks, get_embeddings,
    store_in_chroma, load_from_chroma,
    search_similar, build_context, get_answer,
)

load_dotenv()

st.set_page_config(page_title="PDF Q&A - RAG", layout="centered")

# some custom css
st.markdown("""
<style>
    .block-container { max-width: 760px; padding-top: 2rem; }
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
    .answer-box {
        background: rgba(57, 73, 171, 0.1);
        border-left: 4px solid #3949ab;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        line-height: 1.6;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

st.title("PDF Q&A with RAG")
st.caption("Upload a PDF, ask questions, get grounded answers.")
st.divider()

# init session state
if "ready" not in st.session_state:
    st.session_state.ready = False
    st.session_state.fname = None
    st.session_state.num_chunks = 0
    st.session_state.num_pages = 0

# sidebar - file upload
with st.sidebar:
    st.header("Upload")
    pdf = st.file_uploader("Choose a PDF", type=["pdf"])

    if pdf is not None and st.session_state.fname != pdf.name:
        with st.spinner("Processing..."):
            try:
                docs = load_pdf(pdf)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            chunks = split_into_chunks(docs)
            embed_model = get_embeddings()
            store_in_chroma(chunks, embed_model, pdf.name)

        st.session_state.ready = True
        st.session_state.fname = pdf.name
        st.session_state.num_chunks = len(chunks)
        st.session_state.num_pages = len(docs)

    if st.session_state.ready:
        st.divider()
        st.subheader("Document Info")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.num_pages)
        c2.metric("Chunks", st.session_state.num_chunks)
        st.caption(f"**File:** {st.session_state.fname}")

# main area
if not st.session_state.ready:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

question = st.text_input(
    "Ask a question about the document:",
    placeholder="e.g. What are the main findings?",
)

if question:
    with st.spinner("Generating answer..."):
        try:
            embed_model = get_embeddings()
            db = load_from_chroma(embed_model, st.session_state.fname)
            results = search_similar(question, db, k=4)

            if not results:
                st.warning("Couldn't find relevant sections. Try rephrasing your question.")
                st.stop()

            context = build_context(results)
            answer = get_answer(context, question)

        except EnvironmentError as e:
            st.error(f"{e}\n\nAdd your API key to a .env file:\n```\nGROQ_API_KEY=gsk_...\n```")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.subheader("Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # show which pages the answer came from
    st.subheader("Sources")
    pages = set()
    for doc, score in results:
        p = doc.metadata.get("page", None)
        if p is not None:
            pages.add(p + 1)

    if pages:
        badges = " ".join(f'<span class="source-badge">Page {p}</span>' for p in sorted(pages))
        st.markdown(badges, unsafe_allow_html=True)
    else:
        st.caption("Page numbers not available.")
