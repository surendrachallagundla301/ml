# Book Recommender: baselines vs. two-tower retrieval (Goodbooks-10k)

Candidate-retrieval pipeline for book recommendations, built with PyTorch, scikit-learn and FAISS.

**Data:** Goodbooks-10k (public). Ratings >= 4 are treated as positives. 20,000 sampled users, 10,000 books,
1.23M train / 0.32M held-out interactions (per-user 80/20 random split). Item content = top-500 tag TF-IDF vectors.

**Models**
- Popularity baseline
- Item-item collaborative filtering (cosine similarity, top-100 neighbours per item)
- Two-tower model: user tower (ID embedding + MLP), item tower (ID embedding + tag-content projection + MLP),
  trained with in-batch sampled softmax and logQ popularity correction, temperature 0.07, 3 epochs on CPU

**Evaluation:** Recall@20 and NDCG@20 on 3,000 held-out users, already-seen items excluded.

| Model | Recall@20 | NDCG@20 |
|---|---|---|
| Popularity | 0.079 | 0.088 |
| Item-item CF | 0.213 | 0.238 |
| Two-tower (ID + tag content) | 0.202 | 0.214 |

**Serving (FAISS, 10k items, 64-d):** exact flat index 0.029 ms/query; HNSW 0.043 ms/query at 96.8% recall@100 vs. exact.
At this catalog size exact search is already faster; HNSW only pays off on much larger catalogs.

**Limitations (honest notes):** random (not temporal) split; 3 epochs only, so the two-tower is under-trained and
does not beat item-CF; small scale (1.2M interactions, one CPU machine); no online/A-B evaluation.

**Run:** `pip install torch faiss-cpu pandas scikit-learn scipy`, download `ratings.csv`, `books.csv`, `book_tags.csv`
from github.com/zygmuntz/goodbooks-10k, then `python recsys.py`.

## Serving API (FastAPI + FAISS)
`recsys.py` saves embeddings to `artifacts/`; `app.py` loads them and serves:
- `GET /recommend/{user_id}?k=10` – top-k unseen books for a user (cold-start users return 404)
- `GET /similar/{book_id}?k=10` – nearest books by item embedding
- `GET /health`

Run: `uvicorn app:app` | Tests: `pytest test_app.py` (4 pass) | Latency: `python bench.py`
(900 requests, in-process test client, single CPU machine: p50 10.0 ms, p95 17.5 ms, p99 23.8 ms; includes framework overhead, not network time).
Not done: containerization, auth, a real network load test, cold-start users, online evaluation.
