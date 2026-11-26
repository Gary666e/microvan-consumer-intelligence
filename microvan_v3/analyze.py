import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,numpy as np,pandas as pd,scipy.stats as st,statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,silhouette_score,davies_bouldin_score,adjusted_rand_score
from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;d=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else P.parent/'data'/'microvan.csv'); X=d.iloc[:,2:32];y=d.mvliking;n=len(d);cols=list(X);dem=list(d.columns[32:]);names=['Luxury','Practicality','Family','Safety','Environment']
R=X.corr().values; inv=np.linalg.inv(R); Q=-inv/np.sqrt(np.outer(inv.diagonal(),inv.diagonal()));np.fill_diagonal(Q,0);Rc=R.copy();np.fill_diagonal(Rc,0);kmo=(Rc**2).sum()/((Rc**2).sum()+(Q**2).sum());bart=-(n-1-(2*30+5)/6)*np.linalg.slogdet(R)[1];eig=np.linalg.eigvalsh(R)[::-1];rng=np.random.default_rng(42);null=np.quantile([np.linalg.eigvalsh(np.corrcoef(rng.normal(size=X.shape),rowvar=False))[::-1] for i in range(200)],.95,axis=0)
base=sm.OLS(y,sm.add_constant(X)).fit(); remain=cols.copy(); step=base
while len(remain)>1:
  m,c=min([(sm.OLS(y,sm.add_constant(X[[v for v in remain if v!=c]])).fit(),c) for c in remain],key=lambda t:t[0].aic)
  if m.aic>=step.aic:break
  step=m;remain.remove(c)
# Maximum likelihood factor analysis (not original principal-axis extraction); varimax rotation
fa=make_pipeline(StandardScaler(),FactorAnalysis(n_components=5,rotation='varimax',random_state=42,svd_method='lapack'));fa.fit(X);L=fa[-1].components_.T; anchors=[cols.index(c) for c in ['lthrbetr','miniboxy','kidtrans','safeimpt','shdcarpl']];rr,cc=linear_sum_assignment(-abs(L[anchors,:]));order=cc[np.argsort(rr)];sign=np.sign(L[anchors,order]);L=L[:,order]*sign; F=pd.DataFrame(fa.transform(X)[:,order]*sign,columns=names); reduced=sm.OLS(y,sm.add_constant(F)).fit()
models={'Mean':None,'OLS':(LinearRegression(),{}),'Factor OLS':(make_pipeline(StandardScaler(),FactorAnalysis(n_components=5,rotation='varimax',random_state=42,svd_method='lapack'),LinearRegression()),{}),'Ridge':(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[1,10,30,100]}),'Lasso':(make_pipeline(StandardScaler(),Lasso(max_iter=10000)),{'lasso__alpha':[.01,.03,.1,.3]}),'Elastic net':(make_pipeline(StandardScaler(),ElasticNet(max_iter=10000)),{'elasticnet__alpha':[.01,.05,.1,.3],'elasticnet__l1_ratio':[.2,.5,.8]}),'Random forest':(RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=1),{'max_depth':[3,6],'min_samples_leaf':[3,8]}),'Gradient boosting':(GradientBoostingRegressor(random_state=42),{'n_estimators':[50,100],'max_depth':[1,2],'learning_rate':[.03,.1]})}
outer=KFold(5,shuffle=True,random_state=42);preds={k:np.zeros(n) for k in models};folds=[];params=[]
for fold,(tr,te) in enumerate(outer.split(X),1):
 for name,item in models.items():
  if item is None:p=np.full(len(te),y.iloc[tr].mean());best={}
  else:
   est,grid=item;gs=GridSearchCV(est,grid,cv=KFold(4,shuffle=True,random_state=100+fold),scoring='neg_root_mean_squared_error',n_jobs=1);gs.fit(X.iloc[tr],y.iloc[tr]);p=gs.predict(X.iloc[te]);best=gs.best_params_
  preds[name][te]=p;folds.append([name,fold,mean_squared_error(y.iloc[te],p)**.5,mean_absolute_error(y.iloc[te],p),r2_score(y.iloc[te],p)]);params.append([name,fold,str(best)])
 print('fold',fold,flush=True)
folds=pd.DataFrame(folds,columns=['model','fold','RMSE','MAE','R2']);cv=pd.DataFrame([[k,mean_squared_error(y,p)**.5,mean_absolute_error(y,p),r2_score(y,p),folds[folds.model==k].RMSE.std()] for k,p in preds.items()],columns=['model','RMSE','MAE','R2','fold_RMSE_SD']).sort_values('RMSE');print(cv)
Z=F.values;valid=[];ks={k:KMeans(k,n_init=30,random_state=42).fit(Z) for k in range(2,7)};aris={k:[] for k in ks}
for b in range(50):
 ix=np.random.default_rng(1000+b).integers(0,n,n); f=make_pipeline(StandardScaler(),FactorAnalysis(n_components=5,rotation='varimax',random_state=42,svd_method='lapack')).fit(X.iloc[ix]);z=f.transform(X)
 for k,km in ks.items():
  m=KMeans(k,n_init=10,random_state=42).fit(z[ix]);aris[k].append(adjusted_rand_score(km.labels_,m.predict(z)))
for k,km in ks.items():valid.append([k,silhouette_score(Z,km.labels_),davies_bouldin_score(Z,km.labels_),km.inertia_,np.median(aris[k]),np.quantile(aris[k],.1)])
valid=pd.DataFrame(valid,columns=['k','silhouette','DB','inertia','ARI_median','ARI_p10']);km=ks[3];cent=pd.DataFrame(km.cluster_centers_,columns=names);hi=(cent.Luxury+cent.Practicality).idxmax();lo=cent.Luxury.idxmin();other=next(i for i in range(3) if i not in (hi,lo));mapping={hi:'Luxury pragmatists',other:'Status seekers',lo:'Disengaged'};seg=pd.Series(km.labels_).map(mapping);ss=list(mapping.values());g=d.assign(segment=seg).groupby('segment');profile=g[['mvliking']+dem].mean().loc[ss];profile.insert(0,'n',g.size().loc[ss]);profile.insert(1,'share',profile.n/n);center=F.assign(segment=seg).groupby('segment').mean().loc[ss];anova=st.f_oneway(*[y[seg==s] for s in ss]);tukey=pairwise_tukeyhsd(y,seg);ssb=sum(len(y[seg==s])*(y[seg==s].mean()-y.mean())**2 for s in ss);sst=((y-y.mean())**2).sum();msw=(sst-ssb)/(n-3)
coef=pd.DataFrame({'coef':base.params,'SE':base.bse,'t':base.tvalues,'p':base.pvalues,'HC3_p':base.get_robustcov_results(cov_type='HC3').pvalues});fc=pd.DataFrame({'coef':reduced.params,'SE':reduced.bse,'t':reduced.tvalues,'p':reduced.pvalues,'lo':reduced.conf_int()[0],'hi':reduced.conf_int()[1]});load=pd.DataFrame(L,index=cols,columns=names)
for fn,t in [('baseline_coefficients',coef),('factor_coefficients',fc),('factor_loadings',load),('cv_metrics',cv),('cv_folds',folds),('cluster_validation',valid),('segment_profiles',profile),('segment_centers',center),('oof_predictions',pd.DataFrame({'actual':y,**preds})),('hyperparameters',pd.DataFrame(params,columns=['model','fold','parameters'])),('correlations',X.corr())]:t.to_csv(P/(fn+'.csv'),index=fn not in ['cv_metrics','cv_folds','cluster_validation','hyperparameters'])
(P/'model_outputs.txt').write_text(str(base.summary())+'\n\nBACKWARD AIC\n'+str(step.summary())+'\n\nFACTOR OLS\n'+str(reduced.summary())+'\n\n'+str(tukey));vif=[variance_inflation_factor(sm.add_constant(X).values,i) for i in range(1,31)];q=dict(n=n,missing=int(d.isna().sum().sum()),duplicate_id=int(d.subjnumb.duplicated().sum()),duplicate_nonid=int(d.drop(columns='subjnumb').duplicated().sum()),invalid_likert=int(((d.iloc[:,1:32]<1)|(d.iloc[:,1:32]>9)).sum().sum()),y_mean=y.mean(),y_sd=y.std(),kmo=kmo,bartlett=bart,eigen=eig.tolist(),parallel95=null.tolist(),factor_variance=float((L**2).sum()/30),base_r2=base.rsquared,base_adj=base.rsquared_adj,base_f=base.fvalue,step_r2=step.rsquared,step_adj=step.rsquared_adj,step_aic=step.aic,base_aic=base.aic,step_vars=remain,factor_r2=reduced.rsquared,factor_adj=reduced.rsquared_adj,vif_max=max(vif),vif_median=float(np.median(vif)),bp_p=het_breuschpagan(base.resid,base.model.exog)[1],cook_gt4n=int((base.get_influence().cooks_distance[0]>4/n).sum()),cook_max=float(base.get_influence().cooks_distance[0].max()),anova_F=anova.statistic,anova_p=anova.pvalue,eta=ssb/sst,omega=(ssb-2*msw)/(sst+msw),tukey=str(tukey),ward_ari=adjusted_rand_score(km.labels_,AgglomerativeClustering(n_clusters=3,linkage='ward').fit_predict(Z)),demo_tests={v:float(st.f_oneway(*[d.loc[seg==s,v] for s in ss]).pvalue) for v in dem if v!='female'},gender_p=float(st.chi2_contingency(pd.crosstab(seg,d.female))[1]));(P/'results.json').write_text(json.dumps(q,indent=2));print(q);print(profile);print(center);print(valid);print(fc)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160});blue='#28566E';teal='#15857D';gray='#BAC7CE'
def save(fig,name):fig.tight_layout();fig.savefig(P/(name+'.png'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,2.5));ax.bar(y.value_counts().sort_index().index,y.value_counts().sort_index(),color=blue);ax.set(xlabel='Concept liking (1–9)',ylabel='Respondents',xticks=range(1,10));save(fig,'distribution')
fig,ax=plt.subplots(figsize=(7,2.5));ax.plot(range(1,16),eig[:15],'o-',color=blue,label='Observed');ax.plot(range(1,16),null[:15],'--',color=teal,label='95th percentile, null');ax.axhline(1,color='gray',lw=.8);ax.set(xlabel='Component',ylabel='Eigenvalue',xticks=range(1,16));ax.legend(frameon=False);save(fig,'scree')
fig,ax=plt.subplots(figsize=(7,2.6));t=fc.iloc[1:];ax.errorbar(t.coef,range(5),xerr=[t.coef-t.lo,t.hi-t.coef],fmt='o',color=blue,capsize=3);ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(5),yticklabels=t.index,xlabel='OLS coefficient (95% CI)');ax.invert_yaxis();save(fig,'coefficients')
fig,ax=plt.subplots(figsize=(7,3));t=cv.sort_values('RMSE',ascending=False);ax.barh(t.model,t.RMSE,color=[teal if m==cv.iloc[0].model else gray for m in t.model]);ax.set(xlabel='Pooled out-of-fold RMSE (lower is better)',xlim=(0,max(t.RMSE)*1.15));[ax.text(v+.025,i,f'{v:.3f}',va='center',fontsize=9) for i,v in enumerate(t.RMSE)];save(fig,'cv')
fig,ax=plt.subplots(figsize=(7,2.5));ax.plot(valid.k,valid.silhouette,'o-',label='Silhouette',color=blue);ax.plot(valid.k,valid.ARI_median,'s-',label='Median bootstrap ARI',color=teal);ax.set(xticks=valid.k,xlabel='Number of clusters',ylim=(0,1));ax.legend(frameon=False);save(fig,'validation')
fig,ax=plt.subplots(figsize=(7,2.5));ax.boxplot([y[seg==s] for s in ss],labels=ss,patch_artist=True,boxprops={'facecolor':'#C2DEE0'},medianprops={'color':blue});ax.set(ylabel='Concept liking',ylim=(.5,9.5));save(fig,'appeal')
fig,ax=plt.subplots(figsize=(7,2.5));dendrogram(linkage(Z,method='ward'),truncate_mode='lastp',p=25,ax=ax,no_labels=True,color_threshold=0,above_threshold_color=blue);ax.set(ylabel='Ward distance',xlabel='Merged groups (truncated)');save(fig,'dendrogram')
fig,ax=plt.subplots(figsize=(7,2.5));ax.scatter(base.fittedvalues,base.resid,s=11,alpha=.5,color=blue);ax.axhline(0,color='gray',ls='--');ax.set(xlabel='Fitted liking',ylabel='Residual');save(fig,'residuals')
