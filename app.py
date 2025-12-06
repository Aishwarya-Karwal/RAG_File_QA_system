import os
from loaders.document_loader import load_document
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.vector_store import FaissStore
from rag.qa_chain import QAChain
import numpy as np
import google.generativeai as genai

INDEX_DIR = "data/vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

def build_index_from_file(file_path:str, chunk_size:int=300, overlap:int=50):
    # 1- Load and chunk the document
    text = load_document(file_path=file_path)
    chunks= chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"[index] created {len(chunks)} chunks from document")

    #2- embeddings generation
    embedder = Embedder(model_name = "all-MiniLM-L6-v2")
    embeddings = embedder.embed_texts(chunks, batch_size = 64) #shape (N, dim)
    dim = embeddings.shape[1]
    print(f"[index] embeddings shape : {embeddings.shape}")

    #3- create index or vector store
    store = FaissStore(dim=dim)
    store.create_index(embeddings, metadata = chunks)
    os.makedirs(INDEX_DIR, exist_ok=True)
    store.save(INDEX_PATH, METADATA_PATH)
    print(f"[index] saved indexx -> {INDEX_PATH}")


def query_index(query: str, top_k: int = 5):
    #Load embedder + index
    embedder = Embedder(model_name = "all-MiniLM-L6-v2")
    #Load store
    # we need the dim - Faiss read_index sets it, but create an instanc with dim=0 first
    store = FaissStore(dim=0)
    store.load(INDEX_PATH, METADATA_PATH)

    q_emb = embedder.embed_query(query)
    results = store.search(q_emb, top_k=top_k)
    print(results)

    """[
  {"text": "coding interview patterns include sliding window...", "idx": 53},
  {"text": "two pointers is a common interview pattern...", "idx": 21}
    ] --> list of dict with text and indx"""
    
    retrieved_chunks = [] # above example shows what this will contain
    for ind, _, chunk in results:
        d = {"text": chunk, "idx": ind}
        retrieved_chunks.append(d)
    
    #print(retrieved_chunks)
    qa_chain = QAChain()
    final_answer = qa_chain.answer(query, retrieved_chunks)

    print("\n=================FINAL ANSWER=================\n")
    print(final_answer)
    print("\n=========================================\n")

    return final_answer
   

def main():
    sample_file = r"D:\Books DSA\coding-interview-patterns-nail-your-next-coding-interview.pdf"
    # run once to build the index
    if not os.path.exists(INDEX_PATH):
        build_index_from_file(sample_file)

    # query the index
    query = "What is Topological sort?"
    query_index(query, top_k=5)
    # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # print("Available Gemini Models:")
    # for m in genai.list_models():
    #     print(m.name, "=>", m.supported_generation_methods)

    

# so that running `python -m loaders.app` still works too
if __name__ == "__main__":
    main()
