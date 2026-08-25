"""Streamlit UI for the Students RAG application.

Architecture: ingestion (load -> split) -> embed -> Chroma vector store ->
retrieval (with confidence score) -> augmented generation via OpenAI LLM.

Teaching note: Streamlit re-runs this ENTIRE script top-to-bottom every time
the user interacts with a widget (types text, clicks a button, moves a
slider). Two tools keep that from being wasteful/losing data:
  - st.session_state: a dict that survives across re-runs, used here to keep
    the (potentially large, slow-to-build) vector store in memory.
  - @st.cache_resource: caches the *return value* of a function across
    re-runs (and across users), keyed by its arguments — so we don't reload
    the vector store from disk on every single click.
"""
import streamlit as st

from src import config
from src.ingestion import ingest
from src.rag_pipeline import answer_question
from src.vector_store import get_or_build_vector_store, vector_store_exists

st.set_page_config(page_title="Students RAG", page_icon="🎓", layout="wide")
st.title("🎓 Students RAG Assistant")
st.caption("Ask questions about student records using LangChain + Chroma + OpenAI.")

# --- Sidebar: API key + index management ---
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for embeddings and answers.")

    st.divider()
    st.header("Knowledge Base")
    index_ready = vector_store_exists()
    st.write("Status:", "✅ Indexed" if index_ready else "⚠️ Not built yet")

    # This button triggers the full offline pipeline: STAGE 1 ingestion
    # (ingest()) then STAGE 2 embedding/storage (get_or_build_vector_store()).
    # force_rebuild=True guarantees we always re-embed with the latest code
    # instead of silently reusing a stale on-disk index.
    if st.button("Build / Rebuild Index", use_container_width=True, disabled=not openai_api_key):
        with st.spinner("Loading PDFs and splitting into chunks..."):
            chunks = ingest()
        with st.spinner(f"Embedding {len(chunks)} chunks with text-embedding-3-large (this may take a while)..."):
            st.session_state.vector_store = get_or_build_vector_store(
                openai_api_key=openai_api_key, documents=chunks, force_rebuild=True
            )
        st.success(f"Index built from {len(chunks)} chunks.")
        st.rerun()  # re-run the script so the UI reflects the new "Indexed" status
    if not openai_api_key:
        st.caption("Enter your OpenAI API key to enable indexing.")

    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=config.RETRIEVAL_TOP_K)


@st.cache_resource(show_spinner="Loading vector store...")
def _load_vector_store(openai_api_key: str):
    """Cached loader so re-opening the existing Chroma index only happens once
    per (openai_api_key) value, instead of on every Streamlit re-run."""
    return get_or_build_vector_store(openai_api_key=openai_api_key)


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if st.session_state.vector_store is None and openai_api_key and vector_store_exists():
    st.session_state.vector_store = _load_vector_store(openai_api_key)

# --- Main: question answering ---
question = st.text_input("Ask a question about the student records")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if st.session_state.vector_store is None:
        st.error("Please build the index first using the sidebar button.")
    elif not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        # This one call runs the full online RAG flow: retrieve -> augment -> generate.
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            result = answer_question(
                vector_store=st.session_state.vector_store,
                query=question,
                openai_api_key=openai_api_key,
                top_k=top_k,
            )

        st.subheader("Answer")
        st.write(result.answer)

        st.metric("Confidence score", f"{result.confidence:.2f}")
        if result.confidence < config.CONFIDENCE_THRESHOLD:
            st.warning("Low confidence: the retrieved context may not fully answer this question.")

        # Showing the raw retrieved chunks is good practice for RAG apps: it lets
        # a student/teacher verify *why* the model answered the way it did.
        if result.chunks:
            with st.expander("Retrieved context (sources)"):
                for i, chunk in enumerate(result.chunks, start=1):
                    source = chunk.document.metadata.get("source", "unknown")
                    page = chunk.document.metadata.get("page_label", chunk.document.metadata.get("page", "?"))
                    st.markdown(f"**{i}. {source} (page {page}) — score {chunk.score:.2f}**")
                    st.text(chunk.document.page_content)
