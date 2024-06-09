## INTRODUCTION 
The Early Sepsis Prediction Project is a pioneering initiative designed to transform the landscape of sepsis management in clinical settings through the integration of advanced machine learning (ML) techniques and robust operational processes. Sepsis, a life-threatening response to infection, requires prompt and accurate detection to improve patient outcomes and reduce mortality rates. This project addresses this critical healthcare challenge by facilitating the seamless deployment and management of ML models tailored for early sepsis prediction.

## DESCRIPTION 
The goal of the Early Sepsis Prediction Project is to leverage machine learning models to accurately predict whether patients have sepsis. This entails two primary objectives:

Prediction Aim: Utilize advanced ML models to predict the presence of sepsis in patients, facilitating early detection and timely intervention.

Operational Excellence: Ensure the continuous monitoring of the ML pipeline to uphold model accuracy, reliability, and integration within clinical workflows.

## DATASET 
Link to dataset used - https://physionet.org/content/challenge-2019/1.0.0/.

DATA DESCRIPTION - 
The data repository for the Early Sepsis Prediction Project is structured to support detailed and temporal analysis for each subject involved in the study. The repository contains one file per subject, ensuring that all relevant data for an individual patient is contained within a single file for ease of access and analysis. Each training data file is organized as a table, where the columns represent different types of measurements taken over time. The last column in each row of the table represents the sepsis label, which indicates whether the patient was diagnosed with sepsis at that time point (0 for no sepsis, 1 for sepsis). This binary labeling is crucial for training and evaluating the machine learning models used in the project.

## PREREQUISITES 
Following are the prerequisites of our project:

- Airflow
- Docker
- Google Cloud Platform
- Python
- Streamlit

## DAG 1 - DATA PRE-PROCESSING 
DAG-1 is the inaugural phase of the Early Sepsis Prediction Project, centered on the essential task of data preprocessing. This phase is critical as it transforms raw data into a structured format suitable for machine learning analysis. The key activities within DAG-1 include converting data to CSV format, validating the data, splitting it into training and testing sets, scaling, and cleaning.

Out of the total 40,000 data points available, DAG-1 utilizes 24,000 data points. These are divided into two subsets: 18,000 data points are used for training the machine learning models, while the remaining 6,000 are reserved for testing their performance. This strategic division ensures that the models are trained effectively and evaluated rigorously.

DAG-1 comprises 17 distinct processes, each designed to enhance the quality and usability of the data. The processes are:

1) .PSV TO .CSV - This process converts .PSV files to .CSV format using the Google BigQuery operator. The resulting .CSV files are then saved back to GCP.

2) SCHEMA GENERATION - The input for this process is the merged .CSV file generated from the previous step. This process constructs a schema and compiles statistics, which are subsequently used for data validation to prevent model crashes caused by incorrect data entry. The schema includes column names and data types, while the statistics encompass null count, minimum, maximum, mean, median, and standard deviation for each column. This schema is then produced in a .JSON file.

3) PUSH .JSON TO GCP - This process pushes .JSON schema file from the previous process to GCP. 

4) PULLING .JSON FOR DATA VALIDATION - This process retrieves the .JSON file from GCP for data validation. The validation is performed according to the schema defined in the previous process. The outcome of this validation is a boolean value: True or False. If each data point validates as True, the process proceeds; otherwise, it halts to prevent the model from crashing.

5) TRAIN - TEST SPLIT - This process pulls the .CSV file and splits it into training and testing sets. The resulting sets are then saved locally.

6) PRE - PROCESSING - After the train-test split, X-train and X-test undergo preprocessing. Two separate processes are created: one for preprocessing the training set and another for preprocessing the test set.

7) SCALING TRAIN AND TEST SETS - In this process, the pre-processed training and test datasets are scaled. Separate procedures are established: one for scaling the training set and another for scaling the test set.

8) PUSHING FILES TO GCP - Five distinct processes are established to upload X_train, X_test, Y_train, Y_test, and the scaler to GCP. These files are uploaded in .pkl format for efficiency.

9) CLEANING - Finally, all the .pkl files are cleaned and finalized for modeling in DAG-2.

10) TRIGGER DAG - 2 - This process is used to trigger DAG - 2 after successful completion of DAG - 1.  

## Installation

### Dependencies

The entire project can be run just using Docker. Install it here https://docs.docker.com/engine/install/ 



