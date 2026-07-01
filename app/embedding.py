import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def get_embedding(face_result: dict) -> np.ndarray:
    emb = face_result["embedding"]
    if emb.ndim == 1:
        return normalize_embedding(emb)
    return normalize_embedding(emb.flatten())
