import numpy as np
import faiss
from pathlib import Path
from typing import Optional, Tuple, List

from app.config import config
from app.logger import get_app_logger


class Indexer:
    def __init__(self):
        self.logger = get_app_logger()
        self.dim = config["index"]["embedding_dim"]
        self.index_path = Path(config["index"]["faiss_index_path"])
        self.index: Optional[faiss.Index] = None
        self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.logger.info(f"加载FAISS索引: {self.index_path} ({self.index.ntotal} 条)")
                return
            except Exception as e:
                self.logger.warning(f"FAISS索引加载失败，重新创建: {e}")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.index)
        self.logger.info("创建新FAISS索引")

    def add(self, embeddings: np.ndarray, ids: np.ndarray):
        if len(embeddings) == 0:
            return
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
        self.index.add_with_ids(emb, ids_arr)
        self.logger.info(f"FAISS 添加 {len(embeddings)} 条向量 (总数: {self.index.ntotal})")

    def remove(self, ids: List[int]) -> int:
        if not ids:
            return 0
        ids_arr = np.ascontiguousarray(np.array(ids, dtype=np.int64))
        removed = int(self.index.remove_ids(ids_arr))
        if removed > 0:
            self.logger.info(f"FAISS 删除 {removed} 条向量 (总数: {self.index.ntotal})")
        return removed

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(q, top_k)
        return distances[0], indices[0]

    def get_all_ids(self) -> List[int]:
        if self.index.ntotal == 0:
            return []
        if not hasattr(self.index, "id_map"):
            return []
        return [int(v) for v in faiss.vector_to_array(self.index.id_map)]

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.logger.info(f"FAISS索引已保存: {self.index_path}")

    def reset(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.index)
        if self.index_path.exists():
            self.index_path.unlink()
        self.logger.info("FAISS索引已重置")

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0


indexer = Indexer()
