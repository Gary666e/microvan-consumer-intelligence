import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,average_precision_score
P=Path(__file__).resolve().parent;d=pd.read_csv(P.parent/'data'/'microvan.csv');X=d.iloc[:,2:32];y=d.mvliking.to_numpy();cols=list(X);sizes=[5,10,15,20,30];pred={f'{m}:{k}':np.zeros(len(y)) for m in ['Factor loading','Forward selection'] for k in sizes};selection=[]
for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(X)):
 a,b=X.iloc[tr],X.iloc[te];yt=y[tr];inner=KFold(4,shuffle=True,random_state=101+fold)
 # factor loading heuristic: round robin by strongest absolute loadings; fit solely on outer train.
 sc=StandardScaler().fit(a);fa=FactorAnalysis(n_components=5,rotation='varimax',svd_method='lapack',random_state=42).fit(sc.transform(a));orders=np.argsort(-np.abs(fa.components_),axis=1);fl=[]
 for depth in range(30):
  for j in range(5):
   v=int(orders[j,depth]);
   if v not in fl:fl.append(v)
 # forward selection using precomputed inner fold sufficient statistics for fixed ridge alpha=10.
 cache=[]
 for it,iv in inner.split(a):
  scaler=StandardScaler().fit(a.iloc[it]);A=scaler.transform(a.iloc[it]);B=scaler.transform(a.iloc[iv]);m=yt[it].mean();cache.append((A.T@A,A.T@(yt[it]-m),B,yt[iv]-m))
 fs=[]
 for j in range(30):
  def error(c):
   ix=fs+[c];return np.mean([np.mean((yy-B[:,ix]@np.linalg.solve(G[np.ix_(ix,ix)]+10*np.eye(len(ix)),v[ix]))**2) for G,v,B,yy in cache])
  fs.append(min([c for c in range(30) if c not in fs],key=error))
 for method,order in [('Factor loading',fl),('Forward selection',fs)]:
  for k in sizes:
   ix=order[:k];# fixed alpha avoids reusing selection folds for another tuning step
   m=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(a.iloc[:,ix],yt);pred[f'{method}:{k}'][te]=m.predict(b.iloc[:,ix]);selection.extend([[method,k,fold,cols[c]] for c in ix])
 print('survey fold',fold,flush=True)
rows=[]
for key,p in pred.items():
 m,k=key.split(':');rows.append([m,int(k),mean_squared_error(y,p)**.5,mean_absolute_error(y,p),r2_score(y,p),roc_auc_score(y>=7,p),average_precision_score(y>=7,p)])
pd.DataFrame(rows,columns=['method','items','RMSE','MAE','R2','AUC','AP']).to_csv(P/'survey_extra.csv',index=False);pd.DataFrame(selection,columns=['method','items','fold','item']).to_csv(P/'survey_extra_selected.csv',index=False)
