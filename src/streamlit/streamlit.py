import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()
# Retrieve the Streamlit URI from the environment variables
streamlit_uri = os.getenv('streamlit_uri')

def main():
    # Set the configuration for the Streamlit page
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
        y = df["SepsisLabel"]
        df =  df.drop(columns = "SepsisLabel")
        filename = uploaded_file.name
        pid = filename[2:-4]
        df['Patient_ID'] = pid
 
        #st.subheader("File Content:")
        #st.dataframe(df)  # Use st.dataframe for better visualization
        
        #df["SepsisLabel"] = y
        # Prepare the data for prediction and column names
        col_names = list(df.columns)
        features = df.replace([np.nan, np.inf, -np.inf], None).values.tolist()

        # Send the data to the /predict endpoint
        url = f"{streamlit_uri}/predict"
        if url is None:
            st.error("PREDICT_API_URL environment variable is not set.", icon="🔥")
            return

        try:
            response = requests.post(url, json={"data": features, "columns": col_names})
            response.raise_for_status()  # Raise an error for bad status codes
            predictions = response.json().get("predictions")
            
            st.subheader("Predictions:")
            st.table(predictions)  # Use st.table for better visualization remove later
            
            # Check if any prediction contains "1"
            flag = 0
            for pred in predictions:
                if pred == 1:
                    flag = 1
                    break
            if flag == 1:
                st.error("Patient has sepsis", icon="🚨")
            else:
                st.success("Patient doesn't have sepsis", icon="✅")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}", icon="🔥")

if __name__ == "__main__":
    main()
