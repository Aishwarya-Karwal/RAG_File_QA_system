import os
import streamlit as st

from rag.pipeline import build_index_from_file, query_index

UPLOAD_DIR = "data/uploads"
VECTOR_DIR = "data/vector_store"

st.set_page_config(
    page_title="ChatWithDocs AI📄",
    layout="wide"
)

st.title("ChatWithDocs AI📄")
st.caption("RAG-based • Multi-Document • LLM-powered 🤖")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a text or PDF document",
        type=["txt", "pdf"]
    )

    st.markdown("---")
    st.subheader("📂 Available Documents")

    os.makedirs(VECTOR_DIR, exist_ok=True)

    docs = [
        d for d in os.listdir(VECTOR_DIR)
        if os.path.isdir(os.path.join(VECTOR_DIR, d))
    ]

    if docs:
        selected_docs = st.multiselect(
            "Select one or more documents",
            docs,
            default=docs[:1]
        )
    else:
        selected_docs = []
        st.info("No indexed documents yet.")

    st.markdown("---")
    st.write("🔒 LLM: Enabled")
    st.write("📦 Vector Store: FAISS")


# -----------------------------
# FILE UPLOAD + INDEX BUILD
# -----------------------------
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if uploaded_file is not None:

    if st.session_state.processed_file != uploaded_file.name:

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing document and building index..."):
            doc_id = build_index_from_file(file_path)

        st.session_state.processed_file = uploaded_file.name

        st.success(f"Index ready for {doc_id}")

        st.rerun()

# -----------------------------
# STOP IF NO DOCUMENT SELECTED
# -----------------------------
if not selected_docs:
    st.warning("Please select at least one document.")
    st.stop()


# -----------------------------
# CHAT SECTION
# -----------------------------
st.markdown("---")
st.header("💬 Ask Questions")

st.write(f"📄 Selected Documents: {', '.join(selected_docs)}")

query = st.text_input("Enter your question:")
ask_button = st.button("Ask")

# Initialize safe defaults (prevents crash)
answer = ""
all_sources = []

if ask_button and query:

    with st.spinner("🔍 Searching + Generating answer..."):

        # 🔥 Pass FULL list of selected docs
        answer, all_sources = query_index(query, selected_docs)

# -----------------------------
# SHOW RESPONSE
# -----------------------------
if answer:
    st.markdown("### 📝 Response:")
    st.write(answer)

    st.markdown("### 📚 Sources:")

    for i, src in enumerate(all_sources, 1):
        st.markdown(f"""
**Source {i}**
- 📄 File: {src['metadata']['source']}
- 📍 Page: {src['metadata']['page_number']}

> {src['text'][:1000]}...
""")