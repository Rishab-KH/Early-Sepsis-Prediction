import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

streamlit_uri = os.getenv('streamlit_uri')


def main():
    st.set_page_config(page_title="EMR Sepsis Prediction System", layout="wide")
    # Title and information
    st.title("EMR Sepsis Prediction System")
    st.markdown("<br>", unsafe_allow_html=True)  # Add space between title and info
    st.info('This application predicts if a patient has sepsis or not from the patient record', icon="ℹ️")
    
    # Sidebar for file upload
    st.sidebar.header("Upload Patient Record")
    uploaded_file = st.sidebar.file_uploader("Upload a patient record (PSV format)", type="psv")

    if uploaded_file is not None:
        ## Reading the file in the form of dataframe and the delimiter 'pipe'
        df = pd.read_csv(uploaded_file, delimiter='|')
        
        
if __name__ == "__main__":
    main()