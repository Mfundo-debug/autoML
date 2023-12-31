import streamlit as st
import pandas as pd
import os
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model


if os.path.exists('data.csv'):
    df = pd.read_csv('data.csv', index_col=None)

with st.sidebar:
    st.image('image/logo.png', width=300)
    st.title('Auto ML')
    choice = st.radio("Navigation", ['Upload', 'Profiling', 'Modeling', 'Download'])
    st.info('This application allows you to build an automated machine learning model for your data. Please upload your data and select the target variable. The application will automatically perform data profiling, data cleaning, feature engineering, model selection, hyperparameter tuning, and model evaluation. The application will also provide you with the best model and the code to reproduce the model. The application is built using Streamlit and AutoML libraries.')

if choice == 'Upload':
    st.header('Upload Data')
    st.subheader('Upload your data here')
    st.write('Please upload your data in CSV format. The data should not have any categorical variables. The data should not have any duplicate columns.')

    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        st.write('Shape of the data:', df.shape)
        st.write('Columns of the data:', df.columns)
        st.write('Data types of the columns:', df.dtypes)
        st.write('Number of missing values in each column:', df.isnull().sum())
        st.write('Number of duplicate columns:', df.T.duplicated().sum())
        st.write('Number of duplicate rows:', df.duplicated().sum())
        st.write('Number of unique values in each column:', df.nunique())
        st.write('Remove duplicate columns and duplicate rows before proceeding to the next step.')

        if st.button('Remove Duplicates'):
            df_cleaned = df.drop_duplicates()
            st.write('Shape of the cleaned data:', df_cleaned.shape)
            st.write('Columns of the cleaned data:', df_cleaned.columns)
            st.write('Data types of the cleaned data:', df_cleaned.dtypes)
            st.write('Number of missing values in each column (cleaned data):', df_cleaned.isnull().sum())
            st.write('Number of duplicate columns (cleaned data):', df_cleaned.T.duplicated().sum())
            st.write('Number of duplicate rows (cleaned data):', df_cleaned.duplicated().sum())
            st.write('Number of unique values in each column (cleaned data):', df_cleaned.nunique())
            df = df_cleaned

    if st.button('Save'):
        df.to_csv('data.csv', index=False)
        st.success('Data saved successfully')

if choice == 'Profiling':
    st.header('Automated Exploratory Data Analysis')
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == 'Modeling':
    st.title('Machine Learning')
    target = st.selectbox('Select the target variable', df.columns)
    st.write('The target variable is:', target)

    if st.button('Train Model'):
        st.success('Model trained successfully')
        setup(df, target=target)
        setup_df = pull()
        st.info('This is the ML experiment settings')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_models_df = pull()
        st.info('This is the best model')
        st.dataframe(compare_models_df)
        st.info('This is the model performance')
        st.dataframe(best_model)
        save_model(best_model, 'best_model.pkl')

if choice == 'Download':
    st.header('Download the best model')
    st.write('Please click the button below to download the best model')

    if st.button('Download'):
        st.markdown(get_model('best_model.pkl'))
        st.success('Model downloaded successfully')
