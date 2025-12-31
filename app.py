import os
import streamlit as st

from rag.pipeline import build_index_from_file, query_index

INDEX_PATH = "data/vector_store/faiss.index"
# streamlit config
st.set_page_config(
    page_title = "RAG Document QA System",
    layout = "wide"
)

st.title("RAG Document📄 QA System")
st.caption("RAG-based • Citations enabled • LLM-powered 🤖")

# sidebar
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a text or PDF document from which you want LLM powered answers.",
        type = ["txt", "pdf"]
    )

    st.markdown("---")
    st.write("🔒 LLM: Disabled (Safe mode)")
    st.write("📦 Vector Store: FAISS")

# stop if no file is uploaded
if uploaded_file is None:
    st.info("Please upload a document to proceed.")
    st.stop()

# save uploaded file
os.makedirs("data/uploads", exist_ok = True)
file_path = os.path.join("data/uploads", uploaded_file.name)

with open(file_path, "wb") as f:
    f.write(uploaded_file.read())


st.success(f"Uploaded: **{uploaded_file.name}**")

# Caching - build index only once per document
if not os.path.exists(INDEX_PATH):
    with st.spinner("Processing document and building index..."):
        build_index_from_file(file_path)
    st.success("Index built successfully! You can now chat with the document.")
else:
    st.info("Index already exists. You can chat with the document.")


# Chat interface
st.markdown("---")
st.header("💬 Ask Questions")

query = st.text_input("Enter your question about the document:")

if query:
    with st.spinner("🔍 Searching relevant chunks..."):
        answer = query_index(query)

    st.markdown("### 📝Response: ")
    st.write(answer) 