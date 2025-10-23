# Mini RAG for Financial Data

This project is a **Retrieval-Augmented Generation (RAG)** prototype that can answer both **financial FAQs** and **quantitative fund-performance queries** using a hybrid retrieval pipeline.

---

## Features
- Ingests **fund performance data** (`funds.csv`) and **financial FAQs** (`faqs.csv`)
- Creates **semantic embeddings** using `sentence-transformers`
- Supports both **lexical** (TF-IDF / BM25) and **semantic** (FAISS) retrieval
- Ranks mutual funds by **CAGR**, **volatility**, or **Sharpe ratio**
- Exposes a **FastAPI endpoint** for queries

---
## Overall Design

The system is built as a Mini Retrieval-Augmented Generation (RAG) pipeline with two primary data sources —
(a) textual FAQs and (b) numerical fund performance data.


---
# Architecture Overview

### Data Preparation

Load FAQ CSV and Fund Performance CSV.

Convert numerical data into text-like descriptions, e.g.
"Fund A has 3-year CAGR of 12%, volatility of 8%, and Sharpe ratio of 1.2."

Store all text data in a unified corpus.

### Embedding & Indexing

Semantic Search:
Uses Sentence Transformers to create vector embeddings.
FAISS is used for vector storage and retrieval.

Lexical Search:
Implemented using TF-IDF  for exact keyword-based retrieval.

A retrieval flag allows switching between lexical and semantic modes.

### Query Handling

Accepts user query via FastAPI endpoint:
{"query": "best Sharpe ratio funds", "mode": "semantic"}

Retrieves top relevant FAQs and fund entries.

Combines and formats results into a structured JSON


### Interface / Deployment

Exposed through a FastAPI service.

Caching implemented for embeddings to reduce recomputation time.

---
# Trade-off

-A lightweight model (all-MiniLM-L6-v2) was chosen for faster inference, trading off a bit of accuracy compared to larger models. <br>
-FAISS was used for simple in-memory indexing, which works well for small datasets but needs serialization for persistence.  <br>
-Converting numeric data into text simplified the embedding process but reduced the ability to perform precise numerical comparisons.<br>

# Assumptions

-Query intent is limited to factual retrieval and ranking — not deep reasoning.


# DEPLOYMENT INSTRUCTIONS

## Folder Structure
```
mini-rag-funds/
├── data/ 
├── src/ # Source code
├── cache/ # Cached embeddings 
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/spoorthisetty99/mini-rag-funds.git
cd mini-rag-funds
```
2. Create and Activate a Virtual Environment
macOS / Linux / windows :
```
python -m venv venv
source venv/bin/activate       # on macOS/Linux
venv\Scripts\activate          # on Windows
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Download NLTK Data
```
python -m nltk.downloader punkt
```
5. Run the API Server
```
CUDA_VISIBLE_DEVICES="" python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

```
Interactive API docs: http://127.0.0.1:8000/docs

<img width="750" height="750" alt="Screenshot from 2025-10-23 15-21-03" src="https://github.com/user-attachments/assets/4a2b774f-de11-4ad9-ab60-3b84685ac6a9" />

To use the API and enter your own queries in JSON format, you need to click the **`Try it out`** button in the Swagger UI panel. This action enables the fields in the request section, so you can enter your JSON query parameters, such as a financial question, the retrieval type, and the number of results to return. After filling in your details, you can then execute the request and view the live response from the RAG backend.<br>

<img width="750" height="750" alt="Screenshot from 2025-10-23 15-36-01" src="https://github.com/user-attachments/assets/45b998b5-ce44-40f9-8092-585e9c48f9b6" />
Once your query is ready, simply click the **`Execute`** button below to send your request and get a response from the API. This allows you to interactively test different queries and view live results immediately.<br>


<img width="750" height="750" alt="Screenshot from 2025-10-23 15-37-57" src="https://github.com/user-attachments/assets/ccf1db41-1c1b-49cd-9794-4b068b8d07f8" /><br>
Once you click **`Execute`**, your results will be displayed in the **`Response body`** section.<br>

Example Queries Lexical:
```
{
  "lexical_queries": [
    {"query": "What is a mutual fund?", "retrieval": "lexical", "topk": 4},
    {"query": "Define index fund", "retrieval": "lexical", "topk": 4},
    {"query": "Explain SIP", "retrieval": "lexical", "topk": 4},
    {"query": "What is CAGR?", "retrieval": "lexical", "topk": 4},
    {"query": "What is NAV in mutual funds?", "retrieval": "lexical", "topk": 4},
    {"query": "What is a balanced fund?", "retrieval": "lexical", "topk": 4},
    {"query": "What is an equity fund?", "retrieval": "lexical", "topk": 4},
    {"query": "What is diversification in mutual funds?", "retrieval": "lexical", "topk": 4},
    {"query": "What is an expense ratio?", "retrieval": "lexical", "topk": 4},
    {"query": "What are debt funds?", "retrieval": "lexical", "topk": 4}
  ]
}
```
Example Queries Semantic:
```
{
  "semantic_queries": [
    {"query": "Show me top performing equity funds", "retrieval": "semantic", "topk": 4},
    {"query": "Funds with high 5-year returns", "retrieval": "semantic", "topk": 4},
    {"query": "Low risk investment options", "retrieval": "semantic", "topk": 4},
    {"query": "Funds with balanced allocation between debt and equity", "retrieval": "semantic", "topk": 4},
    {"query": "Best index fund for long term investment", "retrieval": "semantic", "topk": 4},
    {"query": "Funds suitable for tax saving", "retrieval": "semantic", "topk": 4},
    {"query": "Schemes focused on small cap companies", "retrieval": "semantic", "topk": 4},
    {"query": "Funds investing in government securities", "retrieval": "semantic", "topk": 4},
    {"query": "Best performing hybrid funds", "retrieval": "semantic", "topk": 4},
    {"query": "Mutual funds ideal for short term goals", "retrieval": "semantic", "topk": 4}
  ]
}
```
Example Questies Hybrid
```
{
  "hybrid_queries": [
    {"query": "Top 5 funds with the highest 3-year Sharpe ratio", "retrieval": "hybrid", "topk": 6},
    {"query": "Funds with stable returns and low volatility", "retrieval": "hybrid", "topk": 6},
    {"query": "Which index funds track the Nifty 50 and perform well", "retrieval": "hybrid"  "topk": 6},
    {"query": "Show best performing debt-oriented funds", "retrieval": "hybrid", "topk": 6},
    {"query": "Which funds have the best 5-year CAGR with Sharpe ratio above 1", "retrieval": "hybrid",  "topk": 6},
    {"query": "Low expense ratio funds suitable for long term investment", "retrieval": "hybrid", "topk": 6},
    {"query": "Funds suitable for conservative investors seeking steady returns", "retrieval": "hybrid","topk": 6},
    {"query": "Funds offering best SIP performance over 3 years", "retrieval": "hybrid", "topk": 6},
    {"query": "Which funds invest primarily in government securities", "retrieval": "hybrid", "topk": 6},
    {"query": "Tax saving mutual funds with high long term returns", "retrieval": "hybrid","topk": 6},
    {"query": "Hybrid funds balancing equity and debt exposure", "retrieval": "hybrid","topk": 6},

  ]
}


```
