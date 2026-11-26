# Optional Streamlit front end for the archived model
import streamlit as st
import pandas as pd
from app_core import score,optimize,monitor,summary,COLS
st.set_page_config(page_title='Microvan Decision Science',layout='wide');st.title('Microvan | Decision Science Lab');st.caption('Academic prototype; synthetic contact costs only. Not a sales prediction.')
a,b,c,d,e=st.tabs(['Dashboard','Predictor','Segments','Optimizer','Diagnostics']);s=summary()
with a:st.dataframe(pd.DataFrame(s['classification']));st.dataframe(pd.DataFrame(s['repeated_cv']))
with b:
 with st.form('survey'):
  vals=[st.slider(col,1,9,5) for col in COLS];go=st.form_submit_button('Score')
 if go:st.json(score(vals))
with c:st.dataframe(pd.DataFrame(s['gmm']));st.caption('Mixture probabilities describe components, not K-means segment probabilities.')
with d:
 budget=st.number_input('Scenario budget',min_value=0,value=1500);cap=st.number_input('Capacity',0,400,100);floor=st.number_input('Minimum per segment',0,100,10)
 if st.button('Optimize'):st.json(optimize(budget,cap,floor))
with e:
 st.dataframe(pd.DataFrame(s['explainability']));st.dataframe(pd.DataFrame(s['subgroups']));upload=st.file_uploader('30-item CSV, original survey column order',type=['csv'])
 if upload:st.json(monitor(pd.read_csv(upload)[COLS].values.tolist()))
