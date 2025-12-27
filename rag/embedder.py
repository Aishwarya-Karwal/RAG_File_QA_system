from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Wraps a SentenceTransformer model for batched text embedding.

    """

    def __init__(self, model_name:str = "all-MiniLM-L6-v2", device: str = None):
        # device can be cpu or cuda or none (auto)
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of texts into 2d numpy array (num_textx * dim).
        Uses batching for efficiency (to avoid OOM - Out of Memory).
                      """
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i+batch_size]
            embedding = self.model.encode(batch,show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True) # normalize helps for cosine similarity by doing unit L2 norm, so that the dot product of 2 vectors is their cosine similarity - without giving much importancr tp magnitude - thus no bias to longer texts
            all_embeddings.append(embedding)
            # all embeddings looks like below, if 2 is the batch size and 5 is the dim of an embedding
            # all_embeddings =
            # [
            # array(shape=(2, 5)),
            # array(shape=(2, 5))
            # ]

            # after vstack it becomes
            # np.ndarray(shape=(4, 5)) which is what is required by FAISS index
        return np.vstack(all_embeddings)
    
    def embed_query(self, query : str) -> np.ndarray:
        """
        Embeds a single query string into 1d numpy array (dim, )., normalized.
        """
        embedding = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0]