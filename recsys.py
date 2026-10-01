"""Book recommender on Goodbooks-10k: popularity / item-CF baselines vs. a two-tower
retrieval model (user+item embeddings + item tag content features), served with FAISS.
Metrics: Recall@K and NDCG@K on held-out positives (rating>=4), seen items excluded."""
import json, time, numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F, faiss
from sklearn.preprocessing import normalize
rng=np.random.default_rng(0); torch.manual_seed(0); torch.set_num_threads(2)
K=20; POS=4

# ---- data
r=pd.read_csv('ratings.csv'); r=r[r.rating>=POS][['user_id','book_id']].drop_duplicates()
r['u']=r.user_id.astype('category').cat.codes; nU=r.u.max()+1
books=pd.read_csv('books.csv'); nI=books.book_id.max()   # book_id 1..10000
r['i']=r.book_id-1
r=r.sample(frac=1,random_state=0)
# per-user 80/20 holdout (users with >=5 positives)
cnt=r.groupby('u').i.transform('size'); r=r[cnt>=5]
keep=rng.choice(r.u.unique(),20000,replace=False); r=r[r.u.isin(keep)]  # 20k-user sample for CPU speed
r['rk']=r.groupby('u').cumcount(); r['n']=r.groupby('u').i.transform('size')
test=r[r.rk>=(r.n*0.8).astype(int)]; train=r[r.rk<(r.n*0.8).astype(int)]
print(f'users={r.u.nunique()} items={nI} train={len(train)} test={len(test)}',flush=True)
Xtr=sp.csr_matrix((np.ones(len(train),dtype=np.float32),(train.u.values,train.i.values)),shape=(nU,nI))
Xte=sp.csr_matrix((np.ones(len(test),dtype=np.float32),(test.u.values,test.i.values)),shape=(nU,nI))
eval_users=rng.choice(np.unique(test.u.values),3000,replace=False)

# ---- content features: top-500 tags per book, tf-idf weighted
bt=pd.read_csv('book_tags.csv'); b2=books.set_index('goodreads_book_id').book_id
bt['i']=bt.goodreads_book_id.map(b2)-1; bt=bt.dropna(); bt['i']=bt.i.astype(int)
top=bt.groupby('tag_id')['count'].sum().nlargest(500).index; bt=bt[bt.tag_id.isin(top)]
tmap={t:j for j,t in enumerate(top)}
C=sp.csr_matrix((np.log1p(np.clip(bt['count'].values,0,None)),(bt.i.values,bt.tag_id.map(tmap).values)),shape=(nI,500))
df=(C>0).sum(0).A1+1; C=normalize(C.multiply(np.log(nI/df)).tocsr()).astype(np.float32)
Cd=torch.tensor(C.toarray())

# ---- evaluation
def evaluate(score_fn,name):
    rec=ndcg=0; disc=1/np.log2(np.arange(2,K+2))
    for s in range(0,len(eval_users),500):
        us=eval_users[s:s+500]; S=score_fn(us)
        S[Xtr[us].nonzero()]=-1e9
        top=np.argpartition(-S,K,axis=1)[:,:K]
        top=np.take_along_axis(top,np.argsort(-np.take_along_axis(S,top,1),1),1)
        for row,u in enumerate(us):
            t=set(Xte[u].indices); 
            if not t: continue
            hit=np.array([x in t for x in top[row]])
            rec+=hit.sum()/min(len(t),K); ndcg+=(hit*disc).sum()/disc[:min(len(t),K)].sum()
    n=len(eval_users); res=dict(model=name,recall=rec/n,ndcg=ndcg/n); print(res,flush=True); return res
results=[]
pop=np.asarray(Xtr.sum(0)).ravel()
results.append(evaluate(lambda us:np.tile(pop,(len(us),1)).astype(np.float64),'Popularity'))
# item-item CF (cosine)
Xn=normalize(Xtr.tocsc(),axis=0); sim=(Xn.T@Xn).toarray(); np.fill_diagonal(sim,0)
# keep top-100 neighbours per item
idx=np.argpartition(-sim,100,axis=1)[:,:100]; m=np.zeros_like(sim,dtype=bool); np.put_along_axis(m,idx,True,1); sim=np.where(m,sim,0)
results.append(evaluate(lambda us:(Xtr[us]@sim).astype(np.float64),'Item-item CF (cosine, top-100 nbrs)'))

# ---- two-tower
class Tower(nn.Module):
    def __init__(s,d=64,use_content=True):
        super().__init__(); s.ue=nn.Embedding(nU,d); s.ie=nn.Embedding(nI,d); s.use=use_content
        s.cp=nn.Linear(500,d,bias=False); s.u_mlp=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,d))
        s.i_mlp=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,d))
        nn.init.normal_(s.ue.weight,std=.05); nn.init.normal_(s.ie.weight,std=.05)
    def user(s,u): return F.normalize(s.u_mlp(s.ue(u))+s.ue(u),dim=-1)
    def item(s,i):
        e=s.ie(i)+(s.cp(Cd[i]) if s.use else 0); return F.normalize(s.i_mlp(e)+e,dim=-1)
    def all_items(s): return s.item(torch.arange(nI))
def train_tt(use_content,epochs=3,bs=4096,tau=0.07):
    m=Tower(use_content=use_content); opt=torch.optim.Adam(m.parameters(),1e-3)
    U=torch.tensor(train.u.values); I=torch.tensor(train.i.values)
    lp=torch.tensor(np.log(pop/pop.sum()+1e-9),dtype=torch.float32)   # logQ correction for in-batch negatives
    for ep in range(epochs):
        perm=torch.randperm(len(U)); tot=0; t0=time.time()
        for s in range(0,len(U)-bs,bs):
            b=perm[s:s+bs]; u,i=U[b],I[b]
            logits=(m.user(u)@m.item(i).T)/tau-lp[i][None,:]
            same=(i[:,None]==i[None,:])&~torch.eye(bs,dtype=torch.bool); logits=logits.masked_fill(same,-1e9)
            loss=F.cross_entropy(logits,torch.arange(bs)); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        print(f'  ep{ep} loss={tot/(len(U)//bs):.3f} {time.time()-t0:.0f}s',flush=True)
    return m
def tt_scores(m):
    with torch.no_grad(): It=m.all_items()
    def f(us):
        with torch.no_grad(): return (m.user(torch.tensor(us))@It.T).numpy().astype(np.float64)
    return f
print('training two-tower (+tag content)',flush=True); m1=train_tt(True); results.append(evaluate(tt_scores(m1),'Two-tower (ID + tag content)'))

# ---- FAISS serving benchmark
with torch.no_grad(): It=m1.all_items().numpy().astype('float32'); Uq=m1.user(torch.tensor(eval_users)).numpy().astype('float32')
flat=faiss.IndexFlatIP(It.shape[1]); flat.add(It)
hnsw=faiss.IndexHNSWFlat(It.shape[1],32,faiss.METRIC_INNER_PRODUCT); hnsw.add(It); hnsw.hnsw.efSearch=64
def bench(ix,q):
    t=time.time(); 
    for _ in range(3): D,Ix=ix.search(q,100)
    return (time.time()-t)/3/len(q)*1e3,Ix
lf,If=bench(flat,Uq); lh,Ih=bench(hnsw,Uq)
ov=np.mean([len(set(a)&set(b))/100 for a,b in zip(If,Ih)])
serving=dict(flat_ms_per_query=lf,hnsw_ms_per_query=lh,hnsw_recall_vs_exact_at100=ov,n_items=nI,dim=It.shape[1])
print(serving)
json.dump(dict(data=dict(users=int(r.u.nunique()),items=int(nI),train=len(train),test=len(test),eval_users=len(eval_users)),results=results,serving=serving),open('results.json','w'),indent=1,default=float)

# ---- save serving artifacts
import os, scipy.sparse as _sp
os.makedirs('artifacts',exist_ok=True)
with torch.no_grad():
    allU=m1.user(torch.arange(nU)).numpy().astype('float32')
np.save('artifacts/item_emb.npy',It); np.save('artifacts/user_emb.npy',allU)
_sp.save_npz('artifacts/seen.npz',Xtr)
books[['book_id','title','authors','average_rating']].to_csv('artifacts/books.csv',index=False)
np.save('artifacts/users_in_sample.npy',np.unique(train.u.values))
raw_ids=pd.read_csv('ratings.csv',usecols=['user_id']).user_id.astype('category').cat.categories.values
np.save('artifacts/user_ids.npy',raw_ids)
print('saved artifacts')
