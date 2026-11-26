"""Local prototype. No external data transmission. Load only a trusted model rebuilt by analysis_v4.py."""
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.optimize import milp,Bounds,LinearConstraint
P=Path(__file__).resolve().parent
M=joblib.load(P/'models.joblib');COLS=M['columns'];SC=pd.read_csv(P/'synthetic_cost_scenario.csv');
def score(values):
 a=np.asarray(values,float)
 if a.shape!=(30,) or not np.isfinite(a).all() or np.any((a<1)|(a>9)) or np.any(a%1):raise ValueError('Enter 30 integer responses between 1 and 9.')
 x=pd.DataFrame([a],columns=COLS);reg=M['reg'];p=float(reg.predict(x)[0]);ref=M['reference'];base=float(reg.predict(pd.DataFrame([ref],columns=COLS))[0]);w=reg.predict(pd.DataFrame(ref+np.eye(30),columns=COLS))-base;con=(a-ref)*w;f=reg[:-1].transform(x);km=M['kmeans'];dist=km.transform(f)[0];k=int(km.predict(f)[0]);ci=float(M['interval_model'].predict(x)[0]);r=M['interval_radius'];prob=M['gmm'].predict_proba(f)[0]
 assert abs(base+con.sum()-p)<1e-7
 return dict(liking=p,probability=float(M['class'].predict_proba(x)[0,1]),segment=M['segment_names'][k],centroid_distance=float(dist[k]),distance_margin=float(np.sort(dist)[1]-np.sort(dist)[0]),soft_component_probability=prob.tolist(),interval_model_center=ci,interval_90=[float(np.clip(ci-r,1,9)),float(np.clip(ci+r,1,9))],reference_prediction=base,contributions=dict(zip(COLS,con.tolist())))
def optimize(budget,capacity,floor):
 budget=float(budget);capacity=int(capacity);floor=int(floor)
 if budget<0 or not 0<=capacity<=400 or floor<0:raise ValueError('Invalid budget, capacity or minimum.')
 p=SC.OOF_probability.to_numpy();c=SC.synthetic_cost_usd.to_numpy();seg=SC.segment.to_numpy();groups=np.unique(seg);A=np.vstack([c,np.ones(len(p)),*[seg==g for g in groups]])
 r=milp(-p,integrality=np.ones(len(p)),bounds=Bounds(0,1),constraints=LinearConstraint(A,[-np.inf,-np.inf]+[floor]*len(groups),[budget,capacity]+[np.inf]*len(groups)),options={'time_limit':10,'mip_rel_gap':.0001})
 if r.x is None:return {'status':'No feasible solution found', 'solver_message':r.message}
 sel=r.x>.5; assert c[sel].sum()<=budget and sel.sum()<=capacity and all((sel&(seg==g)).sum()>=floor for g in groups)
 return dict(status='Optimal' if r.status==0 else 'Feasible incumbent',contacts=int(sel.sum()),cost=int(c[sel].sum()),expected_high_interest=float(p[sel].sum()),observed_high_interest=int(SC.observed_high_interest[sel].sum()),segment_counts={g:int((sel&(seg==g)).sum()) for g in groups},gap=float(r.mip_gap))
def monitor(batch):
 a=pd.DataFrame(batch,columns=COLS).apply(pd.to_numeric,errors='coerce');valid=a.notna()&a.ge(1)&a.le(9)&a.mod(1).eq(0);ok=valid.all(axis=1);x=a.loc[ok];sd=pd.Series(M['training_sd']);mean=pd.Series(M['training_mean']);shift=((x.mean()-mean)/sd).abs().sort_values(ascending=False) if len(x) else pd.Series(dtype=float)
 return {'rows':len(a),'valid_rows':int(ok.sum()),'invalid_cells':int((~valid).sum().sum()),'largest_standardized_mean_shifts':{str(k):float(v) for k,v in shift.head(5).items()},'note':'Descriptive checks only; no drift thresholds validated. Outcome error requires observed labels.'}
def summary():
 return {name:__import__('json').loads(pd.read_csv(P/(name+'.csv')).to_json(orient='records')) for name in ['classification','repeated_cv','rank_curve','subgroups','explainability','gmm','optimization']}
