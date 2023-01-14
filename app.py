# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 08:29:33 2022

@author: Tony
"""


import streamlit as st
import pandas as pd
import os

#st.write("Hello ")
# In terminal: streamlit run app.py

# Import profilling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# ML stuff
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image('https://images.unsplash.com/photo-1517404215738-15263e9f9178?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80')
    st.title('Auto Stream ML')
    choice = st.radio('Navigation', ['Upload', 'Profiling', 'ML', 'Download'])
    st.info('This applicaiton allows you to to build an automated ML pipeline using Streamlit, Pandas Profililng and PyCaret. And its magic.')


if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

if choice == 'Upload':
    st.title('Upload your data for Modelling!')
    if file := st.file_uploader('Upload your dataset here:'):
        df = pd.read_csv(file, index_col=None)
        df.to_csv('Sourcedata.csv', index=None)
        st.dataframe(df)


if choice == 'Profiling':
    st.title('Automated Exploratory Data Analysis')
    profile_report = df.profile_report()
    st_profile_report(profile_report)


if choice == 'ML':
    st.title("Machine Learning :-)")
    target = st.selectbox('Select Your Target', df.columns)
    if st.button('Train model'):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info('This is the ML Experiment settings')
        best_model = compare_models()
        compare_df = pull()
        st.info('This is the ML Model')
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice== 'Download':
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download the model', f, "trained_model.pkl")



