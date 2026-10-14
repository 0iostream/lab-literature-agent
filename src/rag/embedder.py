"""本地 Embedding：BAAI/bge-small-zh-v1.5 + sentence-transformers。

- LocalEmbedder.encode: 文本列表 → L2 归一化向量 (n, dim)，便于点积=余弦
- get_local_embedder: 懒加载单例，避免每次 search 重新载入模型
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"


class LocalEmbedder:
    """双塔检索用的本地编码器；输出已单位化。"""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        # 延迟导入：没装 sentence-transformers 时，词法检索仍能用
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=2)
def get_local_embedder(model_name: str = DEFAULT_LOCAL_MODEL) -> LocalEmbedder:
    return LocalEmbedder(model_name=model_name)
