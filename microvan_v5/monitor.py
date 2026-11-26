"""Offline monitoring prototype. Thresholds are proposed release gates, not production SLAs."""
from pathlib import Path
import pandas as pd, json, numpy as np
P=Path(__file__).resolve().parent

def monitor(ref,cur):
 alerts=[]
 for name,flag,action in [('score drift',cur.score_psi>=.25,'Investigate feature/segment mix and score drift.'),('recall degradation',ref.recall-cur.recall>=.10,'Inspect segment recall; compare fixed-capacity and coverage policies.'),('capacity utilization',cur.selection_rate<.8*ref.selection_rate,'Review decision threshold and contact capacity.'),('AUC degradation',ref.auc-cur.auc>=.05,'Evaluate challenger on independently labeled data.')]:
  if flag:alerts.append({'alert':name,'action':action})
 return {'status':'REVIEW' if alerts else 'PASS','alerts':alerts,'release_gate':'No automatic retraining or rollout; require labeled independent validation and owner approval.'}
if __name__=='__main__':
 d=pd.read_csv(P/'monitoring_stress.csv');base=monitor(d.iloc[0],d.iloc[0]);stress=monitor(d.iloc[0],d.iloc[1]);assert base['status']=='PASS' and stress['status']=='REVIEW' and len(stress['alerts'])>=2
 (P/'monitor_smoke_test.json').write_text(json.dumps({'baseline':base,'stress':stress,'test':'PASS'},indent=2));print(stress)
