"""检索层：FAISS 向量语义检索 + bigram 关键词检索，RRF 融合（混合检索）。

向量模型用 fastembed(ONNX，无 torch 依赖) 的 BAAI/bge-small-zh-v1.5，512 维、中文语义强。
若 fastembed 未安装或模型下载失败，自动退回纯关键词模式，保证服务永远可用。
"""
import json
import os
import re
from typing import Dict, List, Tuple

# 国内网络直连 HuggingFace 常超时，默认走镜像；可用 HF_ENDPOINT 覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型缓存放到项目内，避免系统临时目录权限问题
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models")

# DeepSeek 无 embedding 接口，故用本地模型；可用环境变量覆盖
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
# bge 系列做检索时给 query 加指令前缀，召回更准
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
RRF_K = 60


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    out, i = [], 0
    while i < len(text):
        s = text[i:i + size].strip()
        if s:
            out.append(s)
        i += size - overlap
    return out


def tokenize(text: str) -> List[str]:
    """中英混合切词：英文单词 + 中文二元组(bigram)，解决中文无空格导致的检索失效"""
    text = text.lower()
    tokens = [w for w in re.findall(r"[a-z0-9_]+", text) if len(w) > 1]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        for i in range(len(run) - 1):
            tokens.append(run[i:i + 2])
    return tokens


def keyword_search(chunks: List[dict], query: str, k: int = 8) -> List[Tuple[dict, float]]:
    """词法检索：命中词频 + 命中词种类加权，避免单个高频词刷分"""
    terms = tokenize(query)
    if not terms:
        return []
    scored = []
    for ch in chunks:
        t = ch["text"].lower()
        score = sum(t.count(w) for w in terms)
        if score <= 0:
            continue
        diversity = sum(1 for w in set(terms) if t.count(w) > 0)
        scored.append((ch, float(score + diversity * 2)))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


class VectorStore:
    """混合检索器：同时保留语义(FAISS)与词法(bigram)两路召回，用 RRF 融合排名"""

    def __init__(self, index_path: str, chunks_path: str):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.chunks: List[dict] = []
        self.index = None
        self._model = None
        self._model_failed = False
        self._dim = 0
        try:
            import faiss  # noqa: F401
            self._faiss_ok = True
        except Exception:
            self._faiss_ok = False
        self.load()

    @property
    def mode(self) -> str:
        return "hybrid(faiss+keyword)" if self.index is not None else "keyword-only"

    # ---------- 向量模型 ----------
    def _ensure_model(self):
        if self._model is not None or self._model_failed:
            return self._model
        try:
            from fastembed import TextEmbedding
            os.makedirs(CACHE_DIR, exist_ok=True)
            self._model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=CACHE_DIR)
            print(f"[kb] 向量模型就绪: {EMBED_MODEL}")
        except Exception as e:
            self._model_failed = True
            print(f"[kb] 向量模型不可用，退回纯关键词检索: {e}")
        return self._model

    def _embed(self, texts: List[str], is_query: bool = False) -> List[list]:
        model = self._ensure_model()
        if model is None:
            return []
        payload = [QUERY_INSTRUCTION + t for t in texts] if is_query else texts
        try:
            fn = getattr(model, "query_embed" if is_query else "passage_embed", None)
            vecs = list(fn(payload) if fn else model.embed(payload))
        except Exception:
            vecs = list(model.embed(payload))
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    # ---------- 索引 ----------
    def add(self, new_chunks: List[dict]):
        self.chunks.extend(new_chunks)

    def build_index(self) -> bool:
        """向量化全部 chunk 并写入 FAISS（内积 = 余弦，向量已归一化）"""
        import numpy as np
        import faiss
        if not self.chunks:
            self.index = None
            return False
        vecs = self._embed([c["text"] for c in self.chunks])
        if not vecs:
            self.index = None
            return False
        arr = np.array(vecs, dtype="float32")
        faiss.normalize_L2(arr)
        self._dim = arr.shape[1]
        index = faiss.IndexFlatIP(self._dim)
        index.add(arr)
        self.index = index
        print(f"[kb] FAISS 索引已建：{index.ntotal} 条 / {self._dim} 维")
        return True

    # ---------- 持久化 ----------
    def save(self):
        try:
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)
        except Exception as e:
            print(f"[kb] chunks 保存失败: {e}")
        try:
            if self.index is not None:
                import faiss
                # faiss.write_index 的 C++ 层不认 Windows 中文路径，改用序列化字节 + Python 写文件
                blob = faiss.serialize_index(self.index)
                with open(self.index_path, "wb") as f:
                    f.write(blob.tobytes())
        except Exception as e:
            print(f"[kb] FAISS 索引保存失败: {e}")

    def load(self):
        try:
            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, encoding="utf-8") as f:
                    self.chunks = json.load(f)
        except Exception:
            self.chunks = []
        if not self.chunks:
            return
        if self._faiss_ok and os.path.exists(self.index_path):
            try:
                import faiss
                import numpy as np
                with open(self.index_path, "rb") as f:
                    blob = np.frombuffer(f.read(), dtype="uint8")
                self.index = faiss.deserialize_index(blob)
                self._dim = self.index.d
                print(f"[kb] 已加载 FAISS 索引：{self.index.ntotal} 条 / {self._dim} 维")
                if self.index.ntotal == len(self.chunks):
                    return
                print("[kb] 索引与 chunks 数量不一致，重建中…")
            except Exception as e:
                print(f"[kb] 索引加载失败，重建中: {e}")
        self.build_index()
        self.save()

    # ---------- 检索 ----------
    def _vector_search(self, query: str, k: int) -> List[Tuple[dict, float]]:
        if self.index is None:
            return []
        import numpy as np
        import faiss
        vecs = self._embed([query], is_query=True)
        if not vecs:
            return []
        arr = np.array(vecs, dtype="float32")
        faiss.normalize_L2(arr)
        scores, idxs = self.index.search(arr, min(k, len(self.chunks)))
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            out.append((self.chunks[int(i)], float(score)))
        return out

    def search(self, query: str, k: int = 4) -> List[dict]:
        """RRF 融合两路召回：score = Σ 1/(RRF_K + rank)"""
        if not self.chunks:
            return []
        vec_hits = self._vector_search(query, k=8)
        kw_hits = keyword_search(self.chunks, query, k=8)
        if not vec_hits and not kw_hits:
            return []
        fused: Dict[int, float] = {}
        by_id: Dict[int, dict] = {}
        for hits in (vec_hits, kw_hits):
            for rank, (ch, _score) in enumerate(hits):
                key = id(ch)
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
                by_id[key] = ch
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [{"source": by_id[k_]["source"], "chunk": by_id[k_]["chunk"],
                 "snippet": by_id[k_]["text"][:120], "text": by_id[k_]["text"]} for k_, _ in ranked]
