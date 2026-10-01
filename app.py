"""FastAPI service for the two-tower book recommender.
GET /health | GET /recommend/{user_id}?k=10 | GET /similar/{book_id}?k=10"""
import numpy as np, pandas as pd, scipy.sparse as sp, faiss
from fastapi import FastAPI, HTTPException, Query

A = 'artifacts/'
item_emb = np.load(A + 'item_emb.npy').astype('float32')
user_emb = np.load(A + 'user_emb.npy').astype('float32')
seen = sp.load_npz(A + 'seen.npz').tocsr()
books = pd.read_csv(A + 'books.csv').set_index('book_id')
user_ids = np.load(A + 'user_ids.npy')
sample_users = set(np.load(A + 'users_in_sample.npy').tolist())      # users the model was trained on
uid_to_idx = {int(u): i for i, u in enumerate(user_ids) if i in sample_users}
index = faiss.IndexFlatIP(item_emb.shape[1]); index.add(item_emb)

app = FastAPI(title='Book recommender (two-tower + FAISS)')

def _fmt(ids, scores):
    out = []
    for i, s in zip(ids, scores):
        if i < 0: continue
        b = books.loc[i + 1]
        out.append({'book_id': int(i + 1), 'title': b.title, 'authors': b.authors, 'score': round(float(s), 4)})
    return out

@app.get('/health')
def health():
    return {'status': 'ok', 'items': int(index.ntotal), 'users': len(uid_to_idx)}

@app.get('/recommend/{user_id}')
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)):
    if user_id not in uid_to_idx:
        raise HTTPException(404, 'unknown user (cold-start users are not supported)')
    u = uid_to_idx[user_id]
    n_seen = len(seen[u].indices)
    D, I = index.search(user_emb[u:u + 1], k + n_seen)           # over-fetch, then drop already-read items
    keep = [(i, d) for i, d in zip(I[0], D[0]) if i not in set(seen[u].indices)][:k]
    return {'user_id': user_id, 'recommendations': _fmt([i for i, _ in keep], [d for _, d in keep])}

@app.get('/similar/{book_id}')
def similar(book_id: int, k: int = Query(10, ge=1, le=100)):
    if not 1 <= book_id <= index.ntotal:
        raise HTTPException(404, 'unknown book')
    D, I = index.search(item_emb[book_id - 1:book_id], k + 1)
    keep = [(i, d) for i, d in zip(I[0], D[0]) if i != book_id - 1][:k]
    return {'book_id': book_id, 'similar': _fmt([i for i, _ in keep], [d for _, d in keep])}
