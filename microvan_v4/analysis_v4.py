import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint,linear_sum_assignment
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,StratifiedKFold,GridSearchCV
from sklearn.feature_selection import SelectKBest,f_regression,SelectFromModel,SequentialFeatureSelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import *
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from xgboost import XGBClassifier,XGBRegressor
import joblib
P=Path(__file__).resolve().parent;D=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else P.parent/'data'/'microvan.csv');X=D.iloc[:,2:32];y=D.mvliking.to_numpy();z=(y>=7).astype(int);n=len(y);cols=list(X);rng=np.random.default_rng(2026)
def factor():return FactorAnalysis(n_components=5,rotation='varimax',svd_method='lapack',random_state=42)
def fr():return make_pipeline(StandardScaler(),factor(),LinearRegression())
def fl():return make_pipeline(StandardScaler(),factor(),LogisticRegression(max_iter=2000))
def tune(est,grid,xx,yy,cv,score):return GridSearchCV(est,grid,cv=cv,scoring=score,n_jobs=1).fit(xx,yy)
def regmet(y,p):return [mean_squared_error(y,p)**.5,mean_absolute_error(y,p),r2_score(y,p)]
def clmet(y,p):return [roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p),log_loss(y,p,labels=[0,1])]
def rank(y,p,k):
 ix=np.argsort(-p,kind='stable')[:k];a=y[ix].sum();return [a/k,a/y.sum(),(a/k)/y.mean(),int(a)]
def save(name,rows,headers):pd.DataFrame(rows,columns=headers).to_csv(P/(name+'.csv'),index=False)
reg={'Factor OLS':(fr(),{}),'Ridge':(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[1,10,30,100]}),'OLS':(LinearRegression(),{})}
cl={'Prevalence':None,'Factor logistic':(fl(),{'logisticregression__C':[.03,.1,1,10]}),'Logistic L2':(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),{'logisticregression__C':[.01,.1,1,10]}),'Logistic L1':(make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',solver='liblinear',random_state=42)),{'logisticregression__C':[.03,.1,1,10]}),'Random forest':(RandomForestClassifier(n_estimators=70,random_state=42,n_jobs=1),{'max_depth':[3,6],'min_samples_leaf':[5,10]}),'XGBoost':(XGBClassifier(n_estimators=60,n_jobs=1,random_state=42,eval_metric='logloss',verbosity=0),{'max_depth':[1,2,3],'learning_rate':[.05,.1]})}
reps=[];rp={};cp={k:np.zeros(n) for k in cl};rawp={k:np.zeros(n) for k in cl};survey={f'{m}:{k}':np.zeros(n) for m in ['F-test','Lasso'] for k in [5,10,15,20,30]};selected=[];ordp=np.zeros(n);ordprob=np.zeros(n);ord_ll=[];xgbp=np.zeros(n);perm=[];contributions=np.zeros(X.shape);intercept=np.zeros(n);foldlog=[]
for rep,seed in enumerate([42,2026,31415]):
 pred={k:np.zeros(n) for k in reg}
 for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(X)):
  a,b=X.iloc[tr],X.iloc[te];yt=y[tr];ct=z[tr];inner=KFold(4,shuffle=True,random_state=101+fold);ci=StratifiedKFold(4,shuffle=True,random_state=101+fold)
  for name,(est,grid) in reg.items():
   g=tune(est,grid,a,yt,inner,'neg_root_mean_squared_error');pred[name][te]=g.predict(b);foldlog.append([rep,fold,name,str(g.best_params_)])
   if rep==0 and name=='Factor OLS':
    m=g.best_estimator_;perm.append(permutation_importance(m,b,y[te],scoring='neg_mean_squared_error',n_repeats=8,random_state=42+fold).importances_mean);ref=a.mean().to_numpy();base=m.predict(pd.DataFrame([ref],columns=cols))[0];w=m.predict(pd.DataFrame(ref+np.eye(len(cols)),columns=cols))-base;contributions[te]=(b.to_numpy()-ref)*w;intercept[te]=base
  if rep:continue
  for name,item in cl.items():
   if item is None:cp[name][te]=rawp[name][te]=ct.mean();continue
   g=tune(*item,a,ct,ci,'neg_log_loss');rawp[name][te]=g.predict_proba(b)[:,1];cal=CalibratedClassifierCV(g.best_estimator_,method='sigmoid',cv=ci).fit(a,ct);cp[name][te]=cal.predict_proba(b)[:,1];foldlog.append([rep,fold,'Class '+name,str(g.best_params_)])
  xg=tune(XGBRegressor(n_estimators=60,n_jobs=1,random_state=42,verbosity=0),{'max_depth':[1,2,3],'learning_rate':[.05,.1]},a,yt,inner,'neg_root_mean_squared_error');xgbp[te]=xg.predict(b);foldlog.append([rep,fold,'XGB reg',str(xg.best_params_)])
  for method in ['F-test','Lasso']:
   for k in [5,10,15,20,30]:
    sel=SelectKBest(f_regression,k=k) if method=='F-test' else SelectFromModel(Lasso(alpha=.05,max_iter=10000),threshold=-np.inf,max_features=k)
    g=tune(make_pipeline(StandardScaler(),sel,Ridge()),{'ridge__alpha':[1,10,30,100]},a,yt,inner,'neg_root_mean_squared_error');survey[f'{method}:{k}'][te]=g.predict(b);selected.extend([[method,k,fold,c] for c in np.array(cols)[g.best_estimator_[1].get_support()]])
  trans=make_pipeline(StandardScaler(),factor()).fit(a);o=OrderedModel(yt,trans.transform(a),distr='logit').fit(method='bfgs',disp=False,maxiter=400);prob=np.asarray(o.model.predict(o.params,exog=trans.transform(b)));levels=np.sort(np.unique(yt));ordp[te]=prob@levels;ordprob[te]=prob[:,levels>=7].sum(1);ord_ll.append([fold,bool(o.mle_retvals['converged']),float(-np.log(np.maximum(prob[np.arange(len(te)),np.searchsorted(levels,y[te])],1e-12)).mean())]);print('repeat',rep,'fold',fold,flush=True)
 rp[rep]=pred
 for name,p in pred.items():reps.append([rep,seed,name,*regmet(y,p)])
save('repeated_cv',reps,['repeat','seed','model','RMSE','MAE','R2']);save('tuning_log',foldlog,['repeat','fold','model','parameters']);save('ordinal_folds',ord_ll,['fold','converged','ordinal_log_loss'])
save('classification',[[k,*clmet(z,p),*rank(z,p,100),*clmet(z,rawp[k])[:3]] for k,p in cp.items()],['model','ROC_AUC','AP','Brier','logloss','precision25','recall25','lift25','hits25','raw_AUC','raw_AP','raw_Brier']);save('extra_regression',[['XGBoost',*regmet(y,xgbp)],['Ordered logit',*regmet(y,ordp)]],['model','RMSE','MAE','R2']);save('ordinal_ranking',[['Ordered logit',*clmet(z,ordprob)]],['model','ROC_AUC','AP','Brier','logloss']);save('survey',[[k.split(':')[0],int(k.split(':')[1]),*regmet(y,p),roc_auc_score(z,p),average_precision_score(z,p)] for k,p in survey.items()],['method','items','RMSE','MAE','R2','AUC','AP']);save('survey_selected',selected,['method','items','fold','item'])
pr=rp[0]['Factor OLS'];p=cp['Factor logistic'];save('rank_curve',[[k,k/n,*rank(z,pr,k),*rank(z,p,k)] for k in [20,40,80,100,120,160,200,300,400]],['k','share','reg_precision','reg_recall','reg_lift','reg_hits','class_precision','class_recall','class_lift','class_hits']);save('threshold_sensitivity',[[t,int((y>=t).sum()),roc_auc_score(y>=t,pr),average_precision_score(y>=t,pr),*rank((y>=t).astype(int),pr,100)] for t in [6,7,8]],['threshold','positive_n','AUC','AP','precision25','recall25','lift25','hits25'])
cal=[]
for name,pp,yy in [('Factor OLS',pr,y),('Factor logistic',p,z)]:
 for i,ix in enumerate(np.array_split(np.argsort(pp),5),1):cal.append([name,i,len(ix),pp[ix].mean(),yy[ix].mean()])
save('calibration',cal,['model','quintile','n','prediction','observed'])
boot=[]
for j in range(1000):
 ix=rng.integers(n,size=n);boot.append([np.sqrt(np.mean([(y[ix]-rp[r][m][ix])**2 for r in rp]))-np.sqrt(np.mean([(y[ix]-rp[r]['Factor OLS'][ix])**2 for r in rp])) for m in ['OLS','Ridge']])
save('paired_bootstrap',[[m,np.mean(np.array(boot)[:,i]),*np.quantile(np.array(boot)[:,i],[.025,.975])] for i,m in enumerate(['OLS','Ridge'])],['comparison_minus_factor','mean_delta_RMSE','lo','hi'])
conf=[]
for j in range(20):
 ix=np.random.default_rng(800+j).permutation(n);tr,ca,te=ix[:240],ix[240:320],ix[320:];m=fr().fit(X.iloc[tr],y[tr]);res=abs(y[ca]-m.predict(X.iloc[ca]));q=np.sort(res)[int(np.ceil((len(ca)+1)*.9))-1];pp=m.predict(X.iloc[te]);lo=np.clip(pp-q,1,9);hi=np.clip(pp+q,1,9);conf.append([j,q,np.mean((y[te]>=lo)&(y[te]<=hi)),np.mean(hi-lo)])
save('conformal',conf,['repeat','radius','coverage','mean_width'])
full=fr().fit(X,y);finalclass=tune(fl(),cl['Factor logistic'][1],X,z,StratifiedKFold(5,shuffle=True,random_state=42),'neg_log_loss');fullclass=CalibratedClassifierCV(finalclass.best_estimator_,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='sigmoid').fit(X,z);trans=full[:-1];F=trans.transform(X);km=KMeans(3,n_init=30,random_state=42).fit(F);L=trans[-1].components_.T;anchors=[cols.index(c) for c in ['lthrbetr','miniboxy','kidtrans','safeimpt','shdcarpl']];rr,cc=linear_sum_assignment(-abs(L[anchors,:]));order=cc[np.argsort(rr)];sign=np.sign(L[anchors,order]);cent=km.cluster_centers_[:,order]*sign;hi=(cent[:,0]+cent[:,1]).argmax();low=cent[:,0].argmin();mapping={int(hi):'Luxury pragmatists',int(low):'Disengaged',int(next(i for i in range(3) if i not in (hi,low))):'Status seekers'};seg=np.array([mapping[int(k)] for k in km.labels_]);gmm=[];gmms={}
for k in range(2,7):
 for cov in ['diag','full']:
  m=GaussianMixture(k,covariance_type=cov,n_init=5,random_state=42,reg_covar=1e-5).fit(F);lab=m.predict(F);gmm.append([k,cov,m.bic(F),silhouette_score(F,lab),adjusted_rand_score(km.labels_,lab),np.mean(m.predict_proba(F).max(1)<.7)]);gmms[k,cov]=m
save('gmm',gmm,['k','covariance','BIC','silhouette','ARI_vs_kmeans3','share_maxprob_lt70']);gm=gmms[3,'full'];grows=[]
for j in range(50):
 ix=np.random.default_rng(1200+j).integers(n,size=n);ft=make_pipeline(StandardScaler(),factor()).fit(X.iloc[ix]);ff=ft.transform(X);gg=GaussianMixture(3,covariance_type='full',n_init=3,random_state=42,reg_covar=1e-5).fit(ff[ix]);grows.append([j,adjusted_rand_score(gm.predict(F),gg.predict(ff))])
save('gmm_bootstrap',grows,['repeat','ARI'])
sub=[]
for field,labels in [('age',pd.cut(D.age,[-np.inf,34,44,np.inf],labels=['<=34','35–44','45+'])),('income',pd.cut(D.income,[-np.inf,49,99,np.inf],labels=['<50K','50–99K','100K+'])),('children',pd.Series(np.where(D.numkids==0,'0',np.where(D.numkids==1,'1','2+')))),('gender',D.female.map({0:'0',1:'1'}))]:
 for l in labels.unique():
  ix=np.asarray(labels==l);sub.append([field,str(l),int(ix.sum()),*regmet(y[ix],pr[ix])[:2],float(np.mean(pr[ix]-y[ix])),float(z[ix].mean())])
save('subgroups',sub,['dimension','group','n','RMSE','MAE','mean_error_pred_minus_actual','interest_rate']);base=sm.OLS(y,sm.add_constant(X)).fit();keep=base.get_influence().cooks_distance[0]<=4/n;sens=fr().fit(X[keep],y[keep]);save('influence_sensitivity',[[n,int(keep.sum()),full.score(X,y),sens.score(X[keep],y[keep]),np.corrcoef(full.predict(X),sens.predict(X))[0,1],np.mean(abs(full.predict(X)-sens.predict(X)))]],['n','retained','full_R2','reduced_sample_R2','prediction_correlation','mean_abs_score_change']);assert np.max(abs(intercept+contributions.sum(1)-pr))<1e-8
save('explainability',[[c,np.mean(np.array(perm)[:,i]),np.mean(abs(contributions[:,i]))] for i,c in enumerate(cols)],['item','permutation_delta_MSE','mean_abs_linear_SHAP_interventional']);pd.DataFrame(contributions,columns=cols).assign(base=intercept,prediction=pr,respondent_id=D.subjnumb).to_csv(P/'individual_contributions.csv',index=False)
cost=np.random.default_rng(2026).integers(8,31,n);scenario=pd.DataFrame({'respondent_id':D.subjnumb,'segment':seg,'synthetic_cost_usd':cost,'OOF_probability':p,'observed_high_interest':z});scenario.to_csv(P/'synthetic_cost_scenario.csv',index=False)
def optimize(p,cost,seg,budget,capacity,floor=0):
 groups=np.unique(seg);A=np.vstack([cost,np.ones(len(p)),*[seg==g for g in groups]]);res=milp(-p,integrality=np.ones(len(p)),bounds=Bounds(0,1),constraints=LinearConstraint(A,[-np.inf,-np.inf]+[floor]*len(groups),[budget,capacity]+[np.inf]*len(groups)),options={'time_limit':15,'mip_rel_gap':.0001})
 if res.x is None:raise RuntimeError(res.message)
 sel=res.x>.5;assert cost[sel].sum()<=budget and sel.sum()<=capacity and all(sum(sel&(seg==g))>=floor for g in groups);return sel,res

def greedy(order,budget,cap,floor):
 chosen=[];spend=0
 for g in np.unique(seg):
  for i in [i for i in order if seg[i]==g][:floor]:chosen.append(i);spend+=cost[i]
 for i in order:
  if i not in chosen and len(chosen)<cap and spend+cost[i]<=budget:chosen.append(i);spend+=cost[i]
 assert spend<=budget and len(chosen)<=cap;return np.isin(np.arange(n),chosen)
opt=[]
for budget,cap,floor in [(800,60,0),(1500,100,0),(1500,100,10),(2200,140,10)]:
 for method in ['MILP','Probability rank','Probability/cost','Random feasible']:
  if method=='MILP':sel,res=optimize(p,cost,seg,budget,cap,floor);gap=res.mip_gap;status=res.message;sels=[sel]
  else:
   orders=[np.argsort(-p)] if method=='Probability rank' else [np.argsort(-p/cost)] if method=='Probability/cost' else [np.random.default_rng(900+j).permutation(n) for j in range(100)];sels=[greedy(o,budget,cap,floor) for o in orders];gap=np.nan;status='heuristic'
  opt.append([budget,cap,floor,method,np.mean([a.sum() for a in sels]),np.mean([cost[a].sum() for a in sels]),np.mean([p[a].sum() for a in sels]),np.mean([z[a].sum() for a in sels]),gap,status])
  if method=='MILP' and (budget,cap,floor)==(1500,100,10):scenario.assign(selected=sels[0]).to_csv(P/'scenario_selected.csv',index=False)
save('optimization',opt,['budget','capacity','segment_min','method','contacts','cost','expected_high_interest','observed_high_interest','mip_gap','status'])
ix=np.random.default_rng(800).permutation(n);tr,ca=ix[:240],ix[240:320];im=fr().fit(X.iloc[tr],y[tr]);iq=np.sort(abs(y[ca]-im.predict(X.iloc[ca])))[72];joblib.dump({'reg':full,'class':fullclass,'kmeans':km,'segment_names':mapping,'gmm':gm,'interval_model':im,'interval_radius':float(iq),'columns':cols,'reference':X.mean().to_numpy(),'training_mean':X.mean().to_dict(),'training_sd':X.std().to_dict()},P/'models.joblib');pd.DataFrame({'respondent_id':D.subjnumb,'actual':y,'high_interest':z,'factor_ols':pr,'ordinal_mean':ordp,'ordinal_p':ordprob,'xgb_reg':xgbp,**{'cal_'+k:v for k,v in cp.items()},**{'raw_'+k:v for k,v in rawp.items()}}).to_csv(P/'oof_v4.csv',index=False);print('DONE',flush=True)
