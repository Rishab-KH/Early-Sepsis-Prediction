import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

streamlit_uri = os.getenv('streamlit_uri')

def main():
    st.title("PSV File Upload and Prediction")

    # Allow the user to upload a PSV file
    uploaded_file = st.file_uploader("Choose a PSV file", type="psv")

    if uploaded_file is not None:
        # Read the file as a dataframe
        df = pd.read_csv(uploaded_file, delimiter='|')
        st.write("File Content:")
        st.write(df)

        # Prepare the data for prediction
        features = df.replace([np.nan, np.inf, -np.inf], None).values.tolist()

        # Send the data to the /predict endpoint
        #url = os.getenv("PREDICT_API_URL")
        url = f"{streamlit_uri}/predict"
        if url is None:
            st.error("PREDICT_API_URL environment variable is not set.")
            return

        try:
            response = requests.post(url, json={"data": features})
            response.raise_for_status()  # Raise an error for bad status codes
            predictions = response.json().get("predictions")
            st.write("Predictions:")
            st.write(predictions)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()
