import os
import numpy as np
import pandas as pd
import streamlit as st
import requests

def main():
    st.title("PSV File Upload and Prediction")

    ## Asking the user to upload a file in the form of PSV file
    uploaded_file = st.file_uploader("Choose a PSV file", type="psv")

    if uploaded_file is not None:
        ## Reading the file in the form of dataframe and the delimiter 'pipe'
        df = pd.read_csv(uploaded_file, delimiter='|')
        st.write("File Content:")
        st.write(df)

        features = df.replace([np.nan, np.inf, -np.inf], None).values.tolist()

        # Send the data to the /predict endpoint
        #url = os.getenv("PREDICT_API_URL") ## commented since url is given directly
        url = "https://sepsis-predict-3wcd2ryf5q-uc.a.run.app/predict"
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
