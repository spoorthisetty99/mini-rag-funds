# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from .preprocessing import load_and_prepare
from .embedder import create_or_load_embeddings
from .indexers import build_tfidf, build_bm25, build_faiss
from .retriever import Retriever
from .answerer import parse_metric_query, rank_funds, make_answer_template
from fastapi.responses import JSONResponse
import traceback

app = FastAPI(title="Mini RAG for Financial Data", version="0.1")

print("🔹 Loading data ...")
corpus_df, funds_df = load_and_prepare("data/faqs.csv", "data/funds.csv")

print("🔹 Creating or loading embeddings ...")
corpus_df, embeddings, model_name = create_or_load_embeddings(corpus_df)

print("🔹 Building indexes ...")
vec, tfidf = build_tfidf(corpus_df["text"].tolist())
tokens = [t.split() for t in corpus_df["text"].fillna("").tolist()]
bm25 = build_bm25(tokens)
import numpy as np
faiss_index = build_faiss(np.array(embeddings, dtype="float32"), embeddings.shape[1])

retriever_obj = Retriever(
    corpus_df=corpus_df,
    embeddings=embeddings,
    tfidf_vec=vec,
    tfidf_matrix=tfidf,
    bm25=bm25,
    faiss_index=faiss_index,
    model_name=model_name,
)

class QueryIn(BaseModel):
    query: str
    retrieval: str = "semantic"
    topk: int = 5

# add these imports at top of file if not already present
from fastapi.responses import JSONResponse
import traceback

# ... existing code above (data loading, retriever_obj, etc.) ...

@app.post("/query")
def query_endpoint(payload: QueryIn):
    try:
        q = payload.query
        mode = payload.retrieval
        topk = payload.topk

        parsed = parse_metric_query(q)
        if parsed:
            metric = parsed["metric"]
            ranked = rank_funds(funds_df, metric, topk=topk)
            answer_text = make_answer_template(q, rank_results=ranked)
            return {"answer": answer_text, "ranked_funds": ranked, "retrieved": []}

        if mode == "semantic":
            retrieved = retriever_obj.semantic_search(q, topk=topk)
        elif mode == "lexical":
            retrieved = retriever_obj.lexical_search(q, topk=topk)
        else:
            retrieved = retriever_obj.hybrid_search(q, topk=topk)

        answer_text = make_answer_template(q, retrieved=retrieved)
        return {"answer": answer_text, "retrieved": retrieved}

    except Exception as e:
        tb = traceback.format_exc()
        # print to server logs too
        print("Exception in /query:", str(e))
        print(tb)
        # return a JSON error with traceback for debugging (DEV ONLY)
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": tb.splitlines()[-20:]   # send last 20 lines for readability
        })
