import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale


from nltk.tokenize import word_tokenize
import nltk

# Ensure required tokenizers are available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)




class Retriever:
    def __init__(self, corpus_df, embeddings, tfidf_vec=None, tfidf_matrix=None, bm25=None, faiss_index=None, model_name='all-MiniLM-L6-v2'):
        self.corpus_df = corpus_df
        self.embeddings = embeddings
        self.model = SentenceTransformer(model_name)
        self.tfidf_vec = tfidf_vec
        self.tfidf_matrix = tfidf_matrix
        self.bm25 = bm25
        self.faiss = faiss_index

    def semantic_search(self, query, topk=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        import faiss
        faiss.normalize_L2(q_emb)
        D, I = self.faiss.search(q_emb, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({'id': self.corpus_df.iloc[idx]['id'],
                            'score': float(score),
                            'text': self.corpus_df.iloc[idx]['text'],
                            'metadata': self.corpus_df.iloc[idx]['metadata']})
        return results
    

    def lexical_search(self, query, topk=5):
        # use bm25 if available
        if self.bm25:
            from nltk.tokenize import word_tokenize
            q_tokens = word_tokenize(query.lower())
            scores = self.bm25.get_scores(q_tokens)
            top_idx = np.argsort(scores)[::-1][:topk]
            return [{'id': self.corpus_df.iloc[i]['id'], 'score': float(scores[i]), 'text': self.corpus_df.iloc[i]['text'], 'metadata': self.corpus_df.iloc[i]['metadata']} for i in top_idx]
        # fallback to tfidf cosine
        else:
            q_vec = self.tfidf_vec.transform([query])
            cosims = (self.tfidf_matrix @ q_vec.T).toarray().squeeze()
            top_idx = np.argsort(cosims)[::-1][:topk]
            return [{'id': self.corpus_df.iloc[i]['id'], 'score': float(cosims[i]), 'text': self.corpus_df.iloc[i]['text'], 'metadata': self.corpus_df.iloc[i]['metadata']} for i in top_idx]

    def hybrid_search(self, query, topk=5, alpha=0.5):
        sem = self.semantic_search(query, topk=50)
        lex = self.lexical_search(query, topk=50)
        # map id -> score
        sem_map = {r['id']: r['score'] for r in sem}
        lex_map = {r['id']: r['score'] for r in lex}
        ids = set(sem_map) | set(lex_map)
        merged = []
        for _id in ids:
            s = sem_map.get(_id, 0.0)
            l = lex_map.get(_id, 0.0)
            # normalize both lists before combining in production; here basic combine:
            merged.append((_id, alpha*s + (1-alpha)*l))
        merged = sorted(merged, key=lambda x: x[1], reverse=True)[:topk]
        results = []
        for _id, score in merged:
            row = self.corpus_df[self.corpus_df['id']==_id].iloc[0]
            results.append({'id': _id, 'score': float(score), 'text': row['text'], 'metadata': row['metadata']})
        return results
