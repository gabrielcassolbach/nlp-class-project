import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, vectors: np.ndarray, docs: list):
        self.docs = docs
        self.vectors = vectors
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(self.vectors)
        self.index.add(self.vectors)

    def retrieve(self, query_vec: np.ndarray, top_k=3):
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = np.asarray(query_vec, dtype=np.float32)
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, top_k)
        results = []
        for j, i in enumerate(I[0]):
            if i != -1:
                doc = self.docs[i].copy()
                doc["score"] = float(D[0][j])
                results.append(doc)
        return results
