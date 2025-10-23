# 🧠 Mini RAG for Financial Data

This project is a **Retrieval-Augmented Generation (RAG)** prototype that can answer both **financial FAQs** and **quantitative fund-performance queries** using a hybrid retrieval pipeline.

---

## 🚀 Features
- Ingests **fund performance data** (`funds.csv`) and **financial FAQs** (`faqs.csv`)
- Creates **semantic embeddings** using `sentence-transformers`
- Supports both **lexical** (TF-IDF / BM25) and **semantic** (FAISS) retrieval
- Ranks mutual funds by **CAGR**, **volatility**, or **Sharpe ratio**
- Exposes a **FastAPI endpoint** for queries

---

```## 📂 Folder Structure
mini-rag-funds/
├── data/ 
├── src/ # Source code
├── cache/ # Cached embeddings 
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

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
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
Server URL: http://127.0.0.1:8000

Interactive API docs: http://127.0.0.1:8000/docs

Example Queries
```
| Query                                                         | Retrieval | Description          |
| ------------------------------------------------------------- | --------- | -------------------- |
| `Which funds have the best Sharpe ratio in the last 3 years?` | semantic  | Rank by `sharpe_3yr` |
| `Top funds by 3-year CAGR`                                    | semantic  | Rank by `3yr_cagr`   |
| `Which funds have the lowest volatility?`                     | semantic  | Rank by `volatility` |
| `What is an index fund?`                                      | lexical   | FAQ text retrieval   |
```
example Queries:
```
{
  "query": "Which funds have the best Sharpe ratio in the last 3 years?",
  "retrieval": "semantic",
  "topk": 5
}
```
