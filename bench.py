import time, numpy as np, json
from fastapi.testclient import TestClient
import app as m
c = TestClient(m.app); uids = list(m.uid_to_idx)[:300]
lat = []
for u in uids * 3:
    t = time.perf_counter(); r = c.get(f'/recommend/{u}?k=10'); lat.append((time.perf_counter() - t) * 1e3); assert r.status_code == 200
res = {'requests': len(lat), 'p50_ms': float(np.percentile(lat, 50)), 'p95_ms': float(np.percentile(lat, 95)), 'p99_ms': float(np.percentile(lat, 99))}
print(res); json.dump(res, open('bench.json', 'w'))
