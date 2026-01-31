from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os

class Vectorizer:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Vectorizer using device: {self.device}")

    def encode_texts(self, texts, batch_size=64):
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def compute_vectors_and_metadata(self, texts, docs, dataset_tag, batch_size=64):
        model_dim = self.model.get_sentence_embedding_dimension()
        model_name_clean = "bge_large" if model_dim == 1024 else "allMini"

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CACHE_DIR = os.path.join(BASE_DIR, "cache")

        vectors_file = f"{CACHE_DIR}/vectors_{dataset_tag}_{model_name_clean}.npy"
        
        try:
            vectors = np.load(vectors_file)
            print(f"Vectors loaded from {vectors_file}")
        except FileNotFoundError:
            print(f"Computing vectors for {len(texts)} documents...")
            vectors = self.model.encode(
                texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True
            ).astype(np.float32)
            np.save(vectors_file, vectors)
            print(f"Vectors computed and saved to {vectors_file}")
        
        return vectors, docs