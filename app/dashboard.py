"""Offline portfolio dashboard; run from repository root with streamlit run app/dashboard.py."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quickstart import load_survey, allocate, segment
st.set_page_config(page_title="Microvan | Consumer Intelligence",layout="wide")
st.title("Microvan | Consumer Intelligence")
st.caption("Academic decision-support prototype. Survey interest is not purchase probability; all costs and intervention outcomes are scenarios.")
@st.cache_data
def load():
    return load_survey()
@st.cache_resource
def models():
    d,x,y=load()
    def pipe(est):
        return make_pipeline(StandardScaler(),FactorAnalysis(n_components=5,rotation="varimax",svd_method="lapack",random_state=42),est)
    z=y>=7
    clf=pipe(LogisticRegression(max_iter=2000,C=.1))
    p=cross_val_predict(clf,x,z,cv=StratifiedKFold(5,shuffle=True,random_state=42),method="predict_proba")[:,1]
    labels,_=segment(x)
    return pipe(LinearRegression()).fit(x,y),clf.fit(x,z),p,labels
d,x,y=load()
reg,clf,p,labels=models()
tabs=st.tabs(["Overview","Consumer predictor","Segments","Budget simulator","Monitoring"])
def table(name):return pd.read_csv(ROOT/"results"/(name+".csv"))
with tabs[0]:
    a,b,c=st.columns(3)
    a.metric("Survey respondents",len(d));b.metric("Offline targeting lift","2.07×");c.metric("Short-survey item reduction","66.7%")
    bench=table("model_benchmark").set_index("model")
    st.bar_chart(bench.RMSE)
    st.dataframe(bench)
with tabs[1]:
    with st.form("survey"):
        cols=st.columns(3)
        values=[cols[i%3].slider(name,1,9,5) for i,name in enumerate(x.columns)]
        submitted=st.form_submit_button("Score")
    if submitted:
        row=pd.DataFrame([values],columns=x.columns)
        st.metric("Predicted concept liking",f"{reg.predict(row)[0]:.2f}/9")
        st.metric("Predicted high-interest probability",f"{clf.predict_proba(row)[0,1]:.1%}")
        st.caption("Quick-start fitted model; not the archived calibrated production prototype.")
with tabs[2]:
    st.dataframe(table("segment_profiles"))
    st.dataframe(table("root_cause"))
    st.dataframe(table("coverage_policy"))
with tabs[3]:
    st.caption("Generated per-contact costs; scores from quick-start cross-validation, not actual outreach.")
    budget=st.number_input("Scenario budget ($)",min_value=0,value=1500)
    cap=st.slider("Capacity",0,400,100)
    floor=st.slider("Minimum per cluster",0,40,10)
    if st.button("Optimize"):
        costs=np.random.default_rng(2026).integers(8,31,len(d))
        st.json(allocate(p,costs,budget,cap,labels,floor))
with tabs[4]:
    st.dataframe(table("monitoring_stress"))
    st.dataframe(table("threshold_utility"))
    st.caption("Offline resampling stress test; no live performance claims.")
