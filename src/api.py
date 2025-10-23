# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pathlib import Path
import numpy as np
import traceback
import os

from .preprocessing import load_and_prepare
from .embedder import create_or_load_embeddings
from .indexers import build_tfidf, build_bm25, build_faiss
from .retriever import Retriever
from .answerer import parse_metric_query, rank_funds, make_answer_template
from nltk.tokenize import word_tokenize

# App
app = FastAPI(title="Mini RAG for Financial Data", version="0.1")

# --- Initialization / wiring (runs on import)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FAQ_PATH = str(DATA_DIR / "faqs.csv")
FUNDS_PATH = str(DATA_DIR / "funds.csv")
EMBED_CACHE = str(ROOT / "cache" / "embeddings.pkl")

print("🔹 Loading data (api) ...")
corpus_df, funds_df = load_and_prepare(FAQ_PATH, FUNDS_PATH)

# load embeddings (uses cache if available)
corpus_df, embeddings, model_name = create_or_load_embeddings(corpus_df, cache_path=EMBED_CACHE)
emb_array = np.array(embeddings).astype("float32")

# build lexical indexes
tfidf_vectorizer, tfidf_matrix = build_tfidf(corpus_df["text"].fillna("").tolist())
tokenized = [word_tokenize(t.lower()) for t in corpus_df["text"].fillna("").tolist()]
bm25 = build_bm25(tokenized)

# build faiss (if faiss available)
faiss_index = None
try:
    faiss_index = build_faiss(emb_array, emb_array.shape[1])
except Exception as e:
    print("Warning: FAISS index build failed (falling back to numpy search).", e)

# instantiate retriever
retriever_obj = Retriever(
    corpus_df=corpus_df,
    funds_df=funds_df,
    tfidf_vectorizer=tfidf_vectorizer,
    tfidf_matrix=tfidf_matrix,
    bm25=bm25,
    faiss_index=faiss_index,
    embeddings=emb_array,
    model_name=model_name
)

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str
    retrieval: str = "semantic"   # "semantic" | "lexical" | "hybrid"
    topk: int = 5
    lex_mode: str = "tfidf"       # only used when retrieval == "lexical" or hybrid lex part (bm25|tfidf)
    alpha: float = 0.6            # only used for hybrid (weight of semantic)

from pathlib import Path
from fastapi import HTTPException

@app.post("/query")
def query_endpoint(req: QueryRequest):
    try:
        q = req.query
        retrieval = req.retrieval.lower()
        topk = max(1, int(req.topk))

        # 1) Metric-style queries: treat as coming from funds.csv only
        metric_info = parse_metric_query(q)
        if metric_info:
            metric = metric_info.get("metric")
            # if metric exists in funds dataframe, return ranked funds (derived from funds.csv)
            if metric and metric in funds_df.columns:
                ranked = rank_funds(funds_df, metric=metric, topk=topk, ascending=False)
                answer_text = make_answer_template(q, rank_results=ranked, retrieved=None)
                return {
                    "answer": answer_text,
                    "ranked_funds": ranked,
                    "retrieved": [],
                    "sources": [Path(FUNDS_PATH).name]
                }
            # else: fall through to do normal retrieval (FAQ may explain metric)

        # 2) Normal retrieval (lexical / semantic / hybrid)
        if retrieval == "semantic":
            retrieved = retriever_obj.semantic_search(q, topk=topk)
        elif retrieval == "lexical":
            retrieved = retriever_obj.lexical_search(q, topk=topk, mode=req.lex_mode)
        elif retrieval == "hybrid":
            retrieved = retriever_obj.hybrid_search(q, topk=topk, alpha=req.alpha, lex_mode=req.lex_mode)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown retrieval mode: {retrieval}")

        # 3) Build the answer text using answerer helper
        answer_text = make_answer_template(q, retrieved=retrieved)

        # 4) Aggregate unique source files strictly from retrieved metadata
        def _infer_source_from_metadata(md: dict) -> str:
            """
            Heuristic to infer which CSV a retrieved item came from based on metadata content.
            Returns 'funds.csv', 'faqs.csv', or None.
            """
            if not md:
                return None
            # prefer explicit source_file if present
            sf = md.get("source_file")
            if sf:
                return sf
            # inspect raw payload if present
            raw = md.get("raw") if isinstance(md.get("raw"), dict) else {}
            # heuristic key sets
            fund_keys = {"fund_id", "fund_name", "3yr_cagr", "sharpe_3yr", "volatility", "category"}
            faq_keys = {"question", "answer"}
            # if fund-like keys exist -> funds.csv
            if any(k in raw for k in fund_keys):
                return Path(FUNDS_PATH).name
            # if faq-like keys exist -> faqs.csv
            if any(k in raw for k in faq_keys):
                return Path(FAQ_PATH).name
            # also check top-level metadata keys
            if md.get("fund_name"):
                return Path(FUNDS_PATH).name
            # fallback: None
            return None

        sources_set = []
        for item in retrieved:
            md = item.get("metadata") or {}
            inferred = _infer_source_from_metadata(md)
            if inferred and inferred not in sources_set:
                sources_set.append(inferred)

        # return the answer, retrieved items, and the inferred sources
        return {
            "answer": answer_text,
            "retrieved": retrieved,
            "sources": sources_set
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Exception in /query:", str(e))
        print(tb)
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": tb.splitlines()[-20:]
        })

if __name__ == "__main__":
    # local debug runner (optional)
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
