from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def build_tfidf(corpus_texts):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf = vec.fit_transform(corpus_texts)
    return vec, tfidf

def build_bm25(tokenized_texts):
    bm25 = BM25Okapi(tokenized_texts)
    return bm25

def build_faiss(embeddings, dim):
    index = faiss.IndexFlatIP(dim)   # inner product on normalized vectors -> cosine
    # normalize embeddings
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# tokens = [word_tokenize(t.lower()) for t in corpus_df['text'].fillna('').tolist()]
# bm25 = build_bm25(tokens)
