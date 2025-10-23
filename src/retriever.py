# src/retriever.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale
from nltk.tokenize import word_tokenize
import nltk
from typing import List, Optional, Dict

# Ensure required tokenizers are available (silent)
nltk.download("punkt", quiet=True)

class Retriever:
    """
    Retriever supports:
      - semantic_search (FAISS or vector dot)
      - lexical_search (tfidf or bm25)
      - hybrid_search (alpha blend of semantic + lexical)
    """

    def __init__(
        self,
        corpus_df,
        funds_df=None,
        tfidf_vectorizer=None,
        tfidf_matrix=None,
        bm25=None,
        faiss_index=None,
        embeddings: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
    ):
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.funds_df = funds_df
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.embeddings = embeddings.astype("float32") if embeddings is not None else None
        self.model_name = model_name or "all-MiniLM-L6-v2"
        # create quick lookup maps
        self.id_to_idx = {row["id"]: i for i, row in self.corpus_df.reset_index().to_dict("index").items()}
        # instantiate a local encoder for query embeddings
        try:
            self.encoder = SentenceTransformer(self.model_name)
        except Exception as e:
            # If model fails to load here, some methods will fall back to using precomputed embeddings only
            print("Warning: could not load SentenceTransformer model:", e)
            self.encoder = None

    def _safe_get_text_by_idx(self, idx):
        return self.corpus_df.iloc[idx]["text"]

    def semantic_search(self, query: str, topk: int = 5) -> List[Dict]:
        """
        Return list of dicts: {id, score, text, metadata}
        Uses faiss_index if available, otherwise computes dot product against embeddings.
        """
        if (self.encoder is None) and (self.embeddings is None):
            raise RuntimeError("No encoder or embeddings available for semantic search.")

        # encode query
        if self.encoder is not None:
            q_emb = self.encoder.encode([query], convert_to_numpy=True)
        else:
            # fallback: if no encoder, try to raise a helpful error
            raise RuntimeError("SentenceTransformer encoder not available for encoding query.")

        # normalize
        q_emb = q_emb.astype("float32")
        # if we have a faiss index:
        if self.faiss_index is not None:
            # ensure q_emb shape (n, dim)
            import faiss
            faiss.normalize_L2(q_emb)
            D, I = self.faiss_index.search(q_emb, topk)  # returns distances (inner product) and indices
            scores = D[0].tolist()
            indices = I[0].tolist()
        elif self.embeddings is not None:
            # compute cosine similarity via normalized dot product
            # normalize both
            emb_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)
            qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
            sims = (emb_norm @ qn.T).flatten()
            indices = list(np.argsort(sims)[::-1][:topk])
            scores = sims[indices].tolist()
        else:
            raise RuntimeError("No vector index or embeddings available for semantic search.")

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.corpus_df):
                continue
            row = self.corpus_df.iloc[idx]
            results.append({
                "id": row["id"],
                "score": float(score),
                "text": row["text"],
                "metadata": row["metadata"]
            })
        return results

    def lexical_search(self, query: str, topk: int = 5, mode: str = "tfidf") -> List[Dict]:
        """
        mode: 'tfidf' or 'bm25'
        returns same format as semantic_search.
        """
        if mode == "bm25":
            if self.bm25 is None:
                raise RuntimeError("BM25 index not initialized.")
            tokens = [t for t in word_tokenize(query.lower()) if t.strip()]
            scores = self.bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:topk]
            results = []
            for idx in top_idx:
                row = self.corpus_df.iloc[idx]
                results.append({
                    "id": row["id"],
                    "score": float(scores[idx]),
                    "text": row["text"],
                    "metadata": row["metadata"]
                })
            return results

        # default tfidf cosine-sim search
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF vectorizer/matrix not initialized.")
        q_vec = self.tfidf_vectorizer.transform([query])
        # compute cosine-like similarity (dot product with TF-IDF matrix)
        sims = (q_vec @ self.tfidf_matrix.T).toarray().flatten()
        top_idx = np.argsort(sims)[::-1][:topk]
        results = []
        for idx in top_idx:
            row = self.corpus_df.iloc[idx]
            results.append({
                "id": row["id"],
                "score": float(sims[idx]),
                "text": row["text"],
                "metadata": row["metadata"]
            })
        return results

    def hybrid_search(self, query: str, topk: int = 5, alpha: float = 0.6, lex_mode: str = "tfidf") -> List[Dict]:
        """
        Combine semantic + lexical scores.
        alpha: weight for semantic (0..1). (1 -> only semantic)
        lex_mode: 'tfidf' or 'bm25'
        """
        # get top candidates from both methods (use larger pool to allow better merging)
        sem = self.semantic_search(query, topk=topk * 3)
        lex = self.lexical_search(query, topk=topk * 3, mode=lex_mode)

        sem_map = {r["id"]: r["score"] for r in sem}
        lex_map = {r["id"]: r["score"] for r in lex}
        ids = set(sem_map.keys()) | set(lex_map.keys())

        # normalize scores to 0..1
        sem_scores = []
        lex_scores = []
        for _id in ids:
            sem_scores.append(sem_map.get(_id, 0.0))
            lex_scores.append(lex_map.get(_id, 0.0))
        # minmax scaling but handle constant arrays
        try:
            sem_norm = minmax_scale(sem_scores) if len(set(sem_scores)) > 1 else [1.0 if s>0 else 0.0 for s in sem_scores]
        except Exception:
            sem_norm = sem_scores
        try:
            lex_norm = minmax_scale(lex_scores) if len(set(lex_scores)) > 1 else [1.0 if s>0 else 0.0 for s in lex_scores]
        except Exception:
            lex_norm = lex_scores

        merged = []
        for i, _id in enumerate(ids):
            s = float(sem_norm[i])
            l = float(lex_norm[i])
            score = alpha * s + (1 - alpha) * l
            merged.append((_id, score))

        merged = sorted(merged, key=lambda x: x[1], reverse=True)[:topk]
        results = []
        for _id, score in merged:
            # find row in corpus_df (could optimize using id_to_idx map)
            df_row = self.corpus_df[self.corpus_df["id"] == _id]
            if df_row.shape[0] == 0:
                continue
            row = df_row.iloc[0]
            results.append({
                "id": _id,
                "score": float(score),
                "text": row["text"],
                "metadata": row["metadata"]
            })
        return results
