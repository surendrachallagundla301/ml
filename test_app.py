from fastapi.testclient import TestClient
import app as m
c = TestClient(m.app)
UID = next(iter(m.uid_to_idx))

def test_health():
    r = c.get('/health'); assert r.status_code == 200 and r.json()['items'] == 10000

def test_recommend_excludes_seen_and_respects_k():
    r = c.get(f'/recommend/{UID}?k=15'); assert r.status_code == 200
    recs = r.json()['recommendations']; assert len(recs) == 15
    seen_ids = set(m.seen[m.uid_to_idx[UID]].indices + 1)
    assert not seen_ids & {x['book_id'] for x in recs}
    scores = [x['score'] for x in recs]; assert scores == sorted(scores, reverse=True)

def test_similar_excludes_self():
    r = c.get('/similar/1?k=10'); assert r.status_code == 200
    assert 1 not in {x['book_id'] for x in r.json()['similar']} and len(r.json()['similar']) == 10

def test_errors():
    assert c.get('/recommend/999999999').status_code == 404
    assert c.get('/similar/0').status_code == 404
    assert c.get(f'/recommend/{UID}?k=0').status_code == 422
