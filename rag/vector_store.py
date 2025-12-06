from typing import List, Tuple
import numpy as np
import faiss
import os
import pickle

class FaissStore:
    """
    A wrapper around FAISS for storing and retrieving vector embeddings.

    Attributes:
        index: The FAISS index for storing embeddings.
        dim: Dimension of the embeddings.
        id_to_text: A mapping from vector IDs to original texts.

    Simple FAISS-backed vector store that keeps:
      - a FAISS index (on disk)
      - a metadata list (texts or identifiers) stored via pickle
    """

    def __init__(self, dim: int):
        self.dim = dim
        # index will br created in crete_index and loaded in load_index
        self.index = None
        self.metadata = []

    def create_index(self, embeddings:np.ndarray, metadata: List[str]) -> None:
        """
         Build an IndexFlatIP (inner product) on normalized embeddings.
        embeddings: shape (N, dim)
        metadata: list of strings (chunks), length N

        """
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.dim
        # initialize the index - (vector store here is the index itself and it will store the embeddings, its flat IP means it will be using inner product or cosine similarity for search and flat means no compression or clustering, so it stores all vectors as is)
        self.index = faiss.IndexFlatIP(self.dim)
        # FAISS expects data in float32 - so casting it to float32
        emb32 = embeddings.astype(np.float32)
        # add embeddings to the index
        self.index.add(emb32)
        # add metadata, such that metadata[i] corresponds to embeddings[i], and  i is the index in FAISS store
        self.metadata = metadata

    def save(self, index_path:str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.
        """
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok = True)
        faiss.write_index(self.index, index_path) # this stores the index in a file in binary format
        # this write the metadata list into a pickle file.
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path:str, metadata_path: str) -> None:
        """
        Load the FAISS index and metadata from disk.
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found.")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        #set dimension from index
        self.dim = self.index.d

    def search(self, query_emb: np.ndarray, top_k:int = 5) -> List[Tuple[int,float,str]]:
        """
        Search the FAISS index for the top_k most similar embeddings to the query embedding.
        query_emb: 1d numpy array of shape (dim, )
        Returns a list of tuples (index, score, metadata)
        score is inner-product (since we use IndexFlatIP on normalized vectors -> cosine)
        """
        assert query_emb.ndim == 1 and query_emb.shape[0] == self.dim
        # FAISS expects data in float32 and 2d array for search
        query_emb32 = query_emb.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_emb32, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue  # no more results
            meta = self.metadata[idx] if idx < len(self.metadata) else ""
            results.append((int(idx), float(score), meta))
        return results