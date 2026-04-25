import os
import re
from loaders.document_loader import load_document
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.vector_store import FaissStore
from rag.qa_chain import QAChain


# BASE DIR
INDEX_BASE_DIR = "data/vector_store"


# -----------------------------
# Helper: clean doc_id
# -----------------------------
def get_doc_id(file_path: str):
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]
    # clean special characters
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return name


# -----------------------------
# BUILD INDEX (PER DOCUMENT)
# -----------------------------
def build_index_from_file(file_path: str, chunk_size: int = 300, overlap: int = 50):

    # create doc_id
    doc_id = get_doc_id(file_path)

    # create folder for this document
    doc_dir = os.path.join(INDEX_BASE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    index_path = os.path.join(doc_dir, "faiss.index")
    metadata_path = os.path.join(doc_dir, "metadata.pkl")

    # 🚨 If already exists → skip or overwrite (your choice)
    if os.path.exists(index_path):
        print(f"[INFO] Index already exists for {doc_id}")
        return doc_id   # return so UI can use it

    # 1️⃣ Load + chunk
    text_lst = load_document(file_path=file_path)

    chunks = chunk_text(
        text_lst,
        file_name=os.path.basename(file_path),
        chunk_size=chunk_size,
        overlap=overlap
    )

    print(f"[index] Created {len(chunks)} chunks")

    # 2️⃣ Embeddings
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]

    embeddings = embedder.embed_texts(texts, batch_size=64)
    dim = embeddings.shape[1]

    print(f"[index] Embedding shape: {embeddings.shape}")

    # 3️⃣ FAISS store
    store = FaissStore(dim=dim)
    store.create_index(embeddings, metadata=chunks)

    store.save(index_path, metadata_path)

    print(f"[index] Saved at {doc_dir}")

    return doc_id


# -----------------------------
# LOAD STORE (PER DOCUMENT)
# -----------------------------
def load_store(doc_id: str):

    doc_dir = os.path.join(INDEX_BASE_DIR, doc_id)
    index_path = os.path.join(doc_dir, "faiss.index")
    metadata_path = os.path.join(doc_dir, "metadata.pkl")

    store = FaissStore(dim=0)
    store.load(index_path, metadata_path)

    return store


# -----------------------------
# QUERY (PER DOCUMENT)
# -----------------------------
def query_index(query: str, doc_ids: list, top_k: int = 5):
    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    all_results = []

    for doc_id in doc_ids:
        index_path = os.path.join("data/vector_store", doc_id, "faiss.index")
        meta_path = os.path.join("data/vector_store", doc_id, "metadata.pkl")

        store = FaissStore(dim=0)
        store.load(index_path, meta_path)

        q_emb = embedder.embed_query(query)
        results = store.search(q_emb, top_k=top_k)

        all_results.extend(results)

    # sort across docs
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]

    retrieved_chunks = []
    sources = []

    for ind, score, meta in all_results:
        chunk = {
            "text": meta["text"],
            "metadata": meta["metadata"],
            "score": score,
            "idx": ind
        }
        retrieved_chunks.append(chunk)
        sources.append(chunk)

    qa_chain = QAChain()
    final_answer = qa_chain.answer(query, retrieved_chunks)

    return final_answer, sources   