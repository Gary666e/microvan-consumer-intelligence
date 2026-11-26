from pathlib import Path
import numpy as np,pandas as pd,json,os
from scipy.special import expit,logit
from scipy.stats import norm,chi2
from scipy.optimize import milp,LinearConstraint,Bounds
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.multitest import multipletests
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;BASE=Path(os.environ.get('MICROVAN_ROOT',P.parent));rng=np.random.default_rng(20261125)
D=pd.read_csv(BASE/'data'/'microvan.csv');O=pd.read_csv(BASE/'microvan_v4/oof_v4.csv');C=pd.read_csv(BASE/'microvan_v4/synthetic_cost_scenario.csv');A=O.merge(C,on='respondent_id');assert np.array_equal(D.subjnumb,O.respondent_id) and np.array_equal(D.mvliking,O.actual)
y=O.high_interest.to_numpy();p=O['cal_Factor logistic'].to_numpy();rank=O.factor_ols.to_numpy();seg=A.segment.to_numpy();cost=A.synthetic_cost_usd.to_numpy();n=len(y);X=D.iloc[:,2:32].to_numpy();names=sorted(set(seg));result={}
def save(name,rows,cols=None):
 d=pd.DataFrame(rows,columns=cols);d.to_csv(P/(name+'.csv'),index=False);return d
def ci(a):
 a=np.asarray(a);m=a.mean();se=a.std(ddof=1)/np.sqrt(len(a));return [float(m),float(m-1.96*se),float(m+1.96*se)]
def top(s,k):return np.argsort(-s,kind='stable')[:k]
def policy(s,k=100,floor=0):
 chosen=np.concatenate([np.where(seg==g)[0][top(s[seg==g],floor)] for g in names]) if floor else np.array([],int);left=np.setdiff1d(np.arange(n),chosen);return np.r_[chosen,left[top(s[left],k-len(chosen))]]
ix=policy(rank);selected=np.isin(np.arange(n),ix);rows=[]
for g in names:
 a=seg==g;tp=(a&selected&(y==1)).sum();pos=y[a].sum();rows.append([g,a.sum(),pos,(a&selected).sum(),tp,pos-tp,tp/pos])
rc=save('root_cause',rows,['segment','n','positive','selected','tp','fn','recall']);rows=[]
for f in [0,10,15,20,25,30]:
 i=policy(rank,floor=f);rows.append([f,*[(seg[i]==g).sum() for g in names],y[i].sum(),y[i].mean(),y[i].sum()/y.sum()])
pol=save('coverage_policy',rows,['min_per_segment',*names,'hits','precision','recall'])
# Nested cost-sensitive model and threshold selection (assumed utility +3/TP, -1/FP).
outer=StratifiedKFold(5,shuffle=True,random_state=76);pred=np.zeros(n);choice=np.zeros(n);thresholds=[];grid=np.linspace(.05,.95,91)
def model(w):return make_pipeline(StandardScaler(),FactorAnalysis(n_components=5,rotation='varimax',random_state=1),LogisticRegression(C=1,max_iter=1000,class_weight=w))
for j,(tr,te) in enumerate(outer.split(X,y)):
 inner=StratifiedKFold(4,shuffle=True,random_state=j+30);best=None
 for weight in [None,'balanced',{0:1,1:2}]:
  q=np.zeros(len(tr))
  for t,v in inner.split(X[tr],y[tr]):q[v]=model(weight).fit(X[tr[t]],y[tr[t]]).predict_proba(X[tr[v]])[:,1]
  for th in grid:
   u=((q>=th)*(4*y[tr]-1)).mean()
   if best is None or u>best[0]:best=(u,weight,th)
 _,w,th=best;pred[te]=model(w).fit(X[tr],y[tr]).predict_proba(X[te])[:,1];choice[te]=pred[te]>=th;thresholds.append([j,str(w),th])
save('threshold_folds',thresholds,['fold','class_weight','threshold']);base=np.zeros(n)
for tr,te in outer.split(X,y):base[te]=model(None).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
rows=[]
for name,z in [('fixed 0.5',base>=.5),('utility tuned',choice.astype(bool)),('fixed 0.25',base>=.25)]:
 tp=int((z*y).sum());fp=int((z*(1-y)).sum());rows.append([name,int(z.sum()),tp,fp,tp/max(z.sum(),1),tp/y.sum(),3*tp-fp])
thr=save('threshold_utility',rows,['policy','contacts','tp','fp','precision','recall','assumed_utility'])
# Simulated datasets, separate IDs, covariate profiles drawn from survey, NO actual outcomes copied.
cols=['lthrbetr','miniboxy','suvcmpct','kidtrans','perfimpt','shdcarpl'];pool=StandardScaler().fit_transform(D[cols])
def simulate(N,observational=False):
 x=pool[rng.integers(0,len(pool),N)];pre=.55*x[:,0]+.5*x[:,1]+rng.normal(0,.8,N);p0=expit(-1.1+.55*x[:,0]+.42*x[:,1]+.3*pre-.22*x[:,4]);tau=.02+.22*expit(-.25-1.05*x[:,0]+.85*x[:,2]-.35*x[:,5]);p1=np.minimum(p0+tau,.97);e=expit(-.2+.8*x[:,0]-.5*x[:,2]+.7*pre) if observational else np.full(N,.5);t=rng.binomial(1,e);yy=rng.binomial(1,np.where(t,p1,p0));df=pd.DataFrame(x,columns=cols);df['pre']=pre;df['treatment']=t;df['outcome']=yy;df['true_p0']=p0;df['true_p1']=p1;df['true_tau']=p1-p0;df['true_propensity']=e;df.insert(0,'synthetic_id',[('OBS-' if observational else 'RCT-')+str(i+1) for i in range(N)]);return df
R=simulate(6000);B=simulate(6000,True);R.to_csv(P/'synthetic_randomized_trial.csv',index=False);B.to_csv(P/'synthetic_observational_study.csv',index=False)
features=cols+['pre'];tx=R.treatment.to_numpy();yy=R.outcome.to_numpy();truth=R.true_tau.to_numpy();pre=R.pre.to_numpy();n1=tx.sum();n0=len(tx)-n1;diff=yy[tx==1].mean()-yy[tx==0].mean();se=np.sqrt(yy[tx==1].var(ddof=1)/n1+yy[tx==0].var(ddof=1)/n0);srm=chi2.sf((n1-n0)**2/len(tx),1)
theta=np.cov(pre,yy,ddof=1)[0,1]/np.var(pre,ddof=1);adj=yy-theta*(pre-pre.mean());d2=adj[tx==1].mean()-adj[tx==0].mean();se2=np.sqrt(adj[tx==1].var(ddof=1)/n1+adj[tx==0].var(ddof=1)/n0)
result['rct']={'n0':int(n0),'n1':int(n1),'control':float(yy[tx==0].mean()),'treatment':float(yy[tx==1].mean()),'true_ate':float(truth.mean()),'ate':float(diff),'lo':float(diff-1.96*se),'hi':float(diff+1.96*se),'p':float(2*norm.sf(abs(diff/se))),'srm_p':float(srm),'cuped':float(d2),'cuped_lo':float(d2-1.96*se2),'cuped_hi':float(d2+1.96*se2),'variance_reduction':float(1-se2**2/se**2),'theta':float(theta),'power_n_per_arm_5pp':int(np.ceil(NormalIndPower().solve_power(proportion_effectsize(.35,.30),power=.8,alpha=.05,ratio=1)))}
exper=[];event=[]
for label,N,mu0,mu1,sd,kind in [('Concept message',1600,5,5.5,2,'normal'),('Targeting policy',1600,.30,.45,0,'binary'),('Questionnaire time',1600,180,125,35,'normal'),('Allocation policy',1600,.30,.36,0,'binary')]:
 t=rng.binomial(1,.5,N);mu=np.where(t,mu1,mu0);v=rng.binomial(1,mu) if kind=='binary' else rng.normal(mu,sd,N);v=np.clip(v,1,9) if label=='Concept message' else v;v=np.maximum(v,1) if label=='Questionnaire time' else v;a=v[t==1];b=v[t==0];delta=a.mean()-b.mean();s=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b));sr=chi2.sf((len(a)-len(b))**2/N,1);exper.append([label,N,len(b),len(a),mu0,mu1,b.mean(),a.mean(),delta,delta-1.96*s,delta+1.96*s,2*norm.sf(abs(delta/s)),sr]);event.extend([[label,i,int(t[i]),v[i]] for i in range(N)])
ex=save('synthetic_experiment_results',exper,['experiment','n','control_n','treat_n','dgp_control_mean','dgp_treatment_mean','control_mean','treatment_mean','difference','ci_low','ci_high','p','srm_p']);ex['holm_p']=multipletests(ex.p,method='holm')[1];ex.to_csv(P/'synthetic_experiment_results.csv',index=False);save('synthetic_experiment_events',event,['experiment','synthetic_id','treatment','outcome'])
# Cross-fit observational nuisance models; matching ATT distinct from ATE.
x=B[features].to_numpy();t=B.treatment.to_numpy();v=B.outcome.to_numpy();e=np.zeros(len(t));m0=e.copy();m1=e.copy()
for tr,te in StratifiedKFold(5,shuffle=True,random_state=12).split(x,t):
 e[te]=LogisticRegression(max_iter=1000).fit(x[tr],t[tr]).predict_proba(x[te])[:,1]
 for a,m in [(0,m0),(1,m1)]:
  k=tr[t[tr]==a];m[te]=LogisticRegression(max_iter=1000).fit(x[k],v[k]).predict_proba(x[te])[:,1]
e0=e.copy();e=np.clip(e,.025,.975);ip=t*v/e-(1-t)*v/(1-e);dr=m1-m0+t*(v-m1)/e-(1-t)*(v-m0)/(1-e);na=v[t==1].mean()-v[t==0].mean();ns=np.sqrt(v[t==1].var(ddof=1)/sum(t)+v[t==0].var(ddof=1)/sum(1-t));l=logit(e);tr=np.where(t==1)[0];co=np.where(t==0)[0];dist,nn=NearestNeighbors(n_neighbors=1).fit(l[co,None]).kneighbors(l[tr,None]);keep=dist[:,0]<=.2*l.std();ti=tr[keep];ci0=co[nn[keep,0]];att=(v[ti]-v[ci0]).mean();result['matching']={'matched':len(ti),'treated':len(tr),'att':float(att),'true_matched_att':float(B.true_tau.iloc[ti].mean()),'unique_controls':int(len(set(ci0)))}
ca=save('causal_estimators',[['Naive difference',na,na-1.96*ns,na+1.96*ns],['IPW ATE',*ci(ip)],['AIPW ATE',*ci(dr)]],['method','estimate','lo','hi']);result['obs']={'true_ate':float(B.true_tau.mean()),'clipped':int(((e0<.025)|(e0>.975)).sum()),'propensity_min':float(e0.min()),'propensity_max':float(e0.max()),'ess':float((np.sum(t/e+(1-t)/(1-e))**2)/np.sum((t/e+(1-t)/(1-e))**2))}
# Holdout CATE learners. Honest transformed-outcome forest is NOT a generalized causal forest.
x=R[features].to_numpy();t=tx;v=yy;tr,te=train_test_split(np.arange(len(t)),test_size=.35,random_state=32,stratify=t);a=tr[t[tr]==0];b=tr[t[tr]==1]
def rf():return RandomForestRegressor(n_estimators=80,max_depth=5,min_samples_leaf=35,random_state=7,n_jobs=2)
f0=rf().fit(x[a],v[a]);f1=rf().fit(x[b],v[b]);pred0=f0.predict(x[te]);T=f1.predict(x[te])-pred0;sl=rf().fit(np.c_[x[tr],t[tr]],v[tr]);S=sl.predict(np.c_[x[te],np.ones(len(te))])-sl.predict(np.c_[x[te],np.zeros(len(te))]);x0=rf().fit(x[a],f1.predict(x[a])-v[a]);x1=rf().fit(x[b],v[b]-f0.predict(x[b]));XL=.5*(x0.predict(x[te])+x1.predict(x[te]));forest=[]
for j in range(100):
 s,h=np.array_split(rng.permutation(tr),2);pseudo=2*(2*t-1)*v;tree=DecisionTreeRegressor(max_depth=4,min_samples_leaf=60,max_features=.8,random_state=j).fit(x[s],pseudo[s]);leaf=tree.apply(x[h]);testleaf=tree.apply(x[te]);means={k:pseudo[h][leaf==k].mean() for k in np.unique(leaf) if (leaf==k).sum()>=10};forest.append([means.get(k,pseudo[h].mean()) for k in testleaf])
H=np.mean(forest,axis=0);testtruth=truth[te];pseudo=2*(2*t[te]-1)*v[te];k=len(te)//4;rows=[];curves=[];q=np.linspace(.01,1,100)
for name,score in [('Random',rng.random(len(te))),('Propensity',pred0),('T learner',T),('S learner',S),('X learner',XL),('Honest TO forest',H),('Oracle (simulation only)',testtruth)]:
 i=top(score,k);ordr=top(score,len(te));gain=np.array([pseudo[ordr[:max(1,int(z*len(te)))]].sum()/len(te) for z in q]);qini=np.trapezoid(gain-q*pseudo.mean(),q);rows.append([name,np.sqrt(np.mean((score-testtruth)**2)) if name not in ['Random','Propensity'] else np.nan,testtruth[i].mean(),pseudo[i].mean(),qini,testtruth[i].sum()]);curves.extend([[name,qq,gg] for qq,gg in zip(q,gain)])
upl=save('uplift_metrics',rows,['model','PEHE','true_top25_CATE','IPW_top25_CATE','Qini_area','true_incremental_top25']);save('uplift_curve',curves,['model','fraction','cumulative_incremental_per_eligible']);save('uplift_holdout_predictions',{'synthetic_id':R.synthetic_id.iloc[te].to_numpy(),'treatment':t[te],'outcome':v[te],'true_tau':testtruth,'propensity':pred0,'T':T,'S':S,'X':XL,'honest_TO_forest':H});result['uplift_n']=len(te)
rows=[]
for g,mask in [('Luxury score <= 0',R.lthrbetr<=0),('Luxury score > 0',R.lthrbetr>0)]:
 a=yy[mask&(tx==1)];b=yy[mask&(tx==0)];d=a.mean()-b.mean();s=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b));rows.append([g,int(mask.sum()),truth[mask].mean(),d,d-1.96*s,d+1.96*s])
save('cate_strata',rows,['group','n','true_cate','estimate','lo','hi'])
# Real records, hypothetical costs: budget frontier and robust maximin with stress scenarios.
G=np.vstack([seg==g for g in names]).astype(float)
def solve(scores,budget,cap=100,floor=10,cc=cost):
 mat=np.vstack([cc,np.ones(n),G]);r=milp(-scores,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(mat,[-np.inf,-np.inf,*([floor]*len(names))],[budget,cap,*([np.inf]*len(names))]),options={'time_limit':8,'mip_rel_gap':.001});return np.where(r.x>.5)[0],r
rows=[]
for budget in [400,600,800,1000,1200,1500,1800,2200]:
 i,r=solve(p,budget);rows.append([budget,len(i),cost[i].sum(),p[i].sum(),y[i].sum(),r.mip_gap,r.status])
front=save('budget_frontier',rows,['budget','contacts','cost','expected','observed_interest','gap','solver_status'])
sc=[p]+[np.clip(p-np.where(seg==g,.12,.02),0,1) for g in names];cc=cost*1.15;mat=np.vstack([np.r_[cc,0],np.r_[np.ones(n),0],np.c_[G,np.zeros(len(names))],np.c_[np.array(sc),-np.ones(len(sc))]]);lo=[-np.inf,-np.inf,*([10]*len(names)),*([0]*len(sc))];hi=[1500,100,*([np.inf]*len(names)),*([np.inf]*len(sc))];r=milp(np.r_[np.zeros(n),-1],integrality=np.r_[np.ones(n),0],bounds=Bounds(np.zeros(n+1),np.r_[np.ones(n),np.inf]),constraints=LinearConstraint(mat,lo,hi),options={'time_limit':12,'mip_rel_gap':.001});rob=np.where(r.x[:n]>.5)[0];nom,nr=solve(p,1500);result['robust_solver']={'status':int(r.status),'gap':float(r.mip_gap)}
nsim=1000;shock=rng.normal(-.02,.065,(nsim,len(names)));cp=np.clip(p[None,:]+shock[:,[names.index(g) for g in seg]],0,1);cm=cost[None,:]*rng.uniform(.9,1.2,(nsim,n));rows=[]
for name,i in [('Nominal',nom),('Maximin robust',rob)]:
 u=cp[:,i].sum(1);co=cm[:,i].sum(1);rows.append([name,len(i),cost[i].sum(),p[i].sum(),min(s[i].sum() for s in sc),y[i].sum(),u.mean(),np.quantile(u,.05),np.mean(co>1500)])
save('robust_policy',rows,['policy','contacts','nominal_cost','nominal_expected','worst_design_expected','observed_interest','MC_mean_expected','MC_p05_expected','MC_budget_violation'])
# Uplift allocation under assumed costs on synthetic holdout only.
cs=rng.integers(8,31,len(te));cap=k;budget=int(cap*16)
def alloc(score):
 r=milp(-score,integrality=np.ones(len(te)),bounds=Bounds(np.zeros(len(te)),np.ones(len(te))),constraints=LinearConstraint(np.vstack([cs,np.ones(len(te))]),[-np.inf,-np.inf],[budget,cap]),options={'time_limit':8,'mip_rel_gap':.001});return np.where(r.x>.5)[0],r
rows=[]
for name,score in [('Random priority',rng.random(len(te))),('Baseline propensity',pred0),('T learner uplift',T),('X learner uplift',XL),('Oracle upper bound',testtruth)]:
 i,r=alloc(score);rows.append([name,len(i),cs[i].sum(),testtruth[i].sum(),pseudo[i].sum(),r.mip_gap,r.status])
save('uplift_allocation',rows,['policy','contacts','assumed_cost','true_incremental_expected','IPW_incremental_estimate','gap','solver_status'])
# Mix-shift stress test, NOT longitudinal production data.
threshold=np.sort(rank)[-100];weights=np.where(seg=='Luxury pragmatists',.25,1.0);new=rng.choice(n,n,replace=True,p=weights/weights.sum());mon=[]
def psi(a,b):
 bins=np.r_[-np.inf,np.unique(np.quantile(a,np.arange(.1,1,.1))),np.inf];u=np.histogram(a,bins)[0]/len(a);v=np.histogram(b,bins)[0]/len(b);u=np.maximum(u,1e-6);v=np.maximum(v,1e-6);return ((v-u)*np.log(v/u)).sum()
for name,i in [('Reference survey',np.arange(n)),('Resampled mix-shift stress',new)]:
 z=rank[i]>=threshold;mon.append([name,len(i),np.mean(seg[i]=='Luxury pragmatists'),y[i].mean(),z.mean(),np.mean(y[i][z]),sum(y[i][z])/sum(y[i]),roc_auc_score(y[i],rank[i]),psi(rank,rank[i])])
save('monitoring_stress',mon,['period','n','luxury_share','positive_rate','selection_rate','precision','recall','auc','score_psi']);result['monitoring_threshold']=float(threshold)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
def chart(name,fn):
 fig,ax=plt.subplots(figsize=(7.1,2.55));fn(ax);fig.tight_layout();fig.savefig(P/(name+'.png'),bbox_inches='tight');plt.close(fig)
chart('root_chart',lambda ax:(ax.barh(rc.segment,rc.fn),ax.set_xlabel('Missed high-interest respondents'),ax.set_title('Missed cases by descriptive segment')))
chart('causal_chart',lambda ax:(ax.errorbar(ca.estimate,ca.method,xerr=[ca.estimate-ca.lo,ca.hi-ca.estimate],fmt='o',capsize=4),ax.axvline(result['obs']['true_ate'],ls='--',color='gray',label='Known simulated ATE'),ax.legend(loc='lower right'),ax.set_xlabel('Risk difference')))
curve=pd.read_csv(P/'uplift_curve.csv')
def gains(ax):
 for name in ['Random','Propensity','T learner','X learner','Honest TO forest','Oracle (simulation only)']:
  z=curve[curve.model==name];ax.plot(z.fraction,z.cumulative_incremental_per_eligible,label=name,linewidth=1.3)
 ax.set(xlabel='Fraction targeted',ylabel='IPW incremental / eligible');ax.legend(fontsize=7,ncol=2)
chart('uplift_chart',gains);chart('frontier_chart',lambda ax:(ax.plot(front.cost,front.expected,'o-'),ax.set(xlabel='Assumed spend ($)',ylabel='Expected high-interest contacts',title='Budget-constrained allocation frontier')))
(P/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print(rc.to_string(index=False));print(pol.to_string(index=False));print(thr.to_string(index=False));print(ca.to_string(index=False));print(upl.to_string(index=False))
