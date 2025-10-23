from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

MODEL_NAME = 'all-MiniLM-L6-v2'

def create_or_load_embeddings(corpus_df, cache_path='cache/embeddings.pkl'):
    if os.path.exists(cache_path):
        with open(cache_path,'rb') as f:
            data = pickle.load(f)
        print("Loaded cached embeddings")
        return data['corpus_df'], data['embeddings'], data['model_name']

    model = SentenceTransformer(MODEL_NAME)
    texts = corpus_df['text'].fillna('').tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    os.makedirs('cache', exist_ok=True)
    with open(cache_path,'wb') as f:
        pickle.dump({'corpus_df': corpus_df, 'embeddings': embeddings, 'model_name':MODEL_NAME}, f)
    return corpus_df, embeddings, MODEL_NAME
