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
   
if __name__ == "__main__":
    main()