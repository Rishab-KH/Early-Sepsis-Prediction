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

## Data Card
- Size: 1552210 rows × 42 columns [From 40,000 Patients]
- Data Types

| Variable | Role    | Type   | Description                       | Unit                  |
|----------|---------|--------|-----------------------------------|-----------------------|
| HR       | Feature | float64| Heart rate                        | beats per minute      |
| O2Sat    | Feature | float64| Pulse oximetry                    | %                     |
| Temp     | Feature | float64| Temperature                       | Deg C                 |
| SBP      | Feature | float64| Systolic BP                       | mm Hg                 |
| MAP      | Feature | float64| Mean arterial pressure            | mm Hg                 |
| DBP      | Feature | float64| Diastolic BP                      | mm Hg                 |
| Resp     | Feature | float64| Respiration rate                  | breaths per minute    |
| EtCO2    | Feature | float64| End tidal carbon dioxide          | mm Hg                 |
| BaseExcess      | Feature | float64| Measure of excess bicarbonate                      | mmol/L                 |
| HCO3            | Feature | float64| Bicarbonate                                        | mmol/L                 |
| FiO2            | Feature | float64| Fraction of inspired oxygen                        | %                      |
| pH              | Feature | float64| pH level (not applicable)                          | N/A                    |
| PaCO2           | Feature | float64| Partial pressure of carbon dioxide from arterial blood | mm Hg              |
| SaO2            | Feature | float64| Oxygen saturation from arterial blood              | %                      |
| AST             | Feature | float64| Aspartate transaminase                             | IU/L                   |
| BUN             | Feature | float64| Blood urea nitrogen                                | mg/dL                  |
| Alkalinephos    | Feature | float64| Alkaline phosphatase                               | IU/L                   |
| Calcium         | Feature | float64| Calcium level                                      | mg/dL                  |
| Chloride        | Feature | float64| Chloride                                           | mmol/L                 |
| Creatinine      | Feature | float64| Creatinine                                         | mg/dL                  |
| Bilirubin_direct| Feature | float64| Bilirubin direct                                   | mg/dL                  |
| Glucose         | Feature | float64| Serum glucose                                      | mg/dL                  |
| Lactate         | Feature | float64| Lactic acid                                        | mg/dL                  |
| Magnesium       | Feature | float64| Magnesium                                          | mmol/L                 |
| Phosphate       | Feature | float64| Phosphate                                          | mg/dL                  |
| Potassium       | Feature | float64| Potassium                                          | mmol/L                 |
| Bilirubin_total | Feature | float64| Total bilirubin                                    | mg/dL                  |
| TroponinI       | Feature | float64| Troponin I                                         | ng/mL                  |
| Hct             | Feature | float64| Hematocrit                                         | %                      |
| Hgb             | Feature | float64| Hemoglobin                                         | g/dL                   |
| PTT             | Feature | float64| Partial thromboplastin time                        | seconds                |
| WBC             | Feature | float64| Leukocyte count                                    | count*10^3/µL          |
| Fibrinogen      | Feature | float64| Fibrinogen level                                   | mg/dL                  |
| Platelets       | Feature | float64| Platelet count                                     | count*10^3/µL          |
| Age         | Feature | float64| Years (100 for patients 90 or above)                | years                  |
| Gender      | Feature | int64| Female (0) or Male (1)                              | categorical            |
| Unit1       | Feature | float64| Administrative identifier for ICU unit (MICoult64)  | categorical            |
| Unit2       | Feature | float64| Administrative identifier for ICU unit (SICU)       | categorical            |
| HospAdmTime | Feature | float64| Hours between hospital admit and ICU admit          | hours                  |
| ICULOS      | Feature | int64| ICU length-of-stay                                  | hours                  |
| Patient ID  | ID      | String| ID of the patient                                   | ID |
| SepsisLabel | Target  | int64| For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6 and 0 if t< tsepsis−6, For non-sepsis patients SepsisLabel is 0 | 0 or 1 |

## PREREQUISITES 
Following are the prerequisites of our project:

- Airflow
- Docker
- Google Cloud Platform
- Python
- Streamlit

## DAG 1 - DATA PRE-PROCESSING 

![image](https://github.com/Rishab-KH/IE7374-Sepsis-Classification/assets/47169600/fc7a1343-7ca6-45e8-bd0e-125871216448)

![image](https://github.com/Rishab-KH/IE7374-Sepsis-Classification/assets/47169600/6ce104aa-10c6-4232-a9dd-53a07e394031)


DAG-1 is the inaugural phase of the Early Sepsis Prediction Project, centered on the essential task of data preprocessing. This phase is critical as it transforms raw data into a structured format suitable for machine learning analysis. The key activities within DAG-1 include converting data to CSV format, validating the data, splitting it into training and testing sets, scaling, and cleaning.

Out of the total 40,000 patient files from two hospitals, DAG-1 utilizes 24,000 files. These are split into two subsets: 18,000 patient files are used for training the machine learning models, while the remaining 6,000 are reserved for testing model performance. This strategic division ensures effective training and rigorous evaluation of the models. The remaining 16,000 patient files will be sent in batches, simulating real-time data, and will be processed later as part of the Deployment pipeline (DAG-3) to validate the models' performance and detect any potential data or concept drift over time.

Patient File Description:
 Format: Each patient file is in .PSV format.
 Content: Contains hourly records for that particular patient.
 Duration: For each patient, there may be records for up to 60 hours, resulting in up to 60 records per file.

In DAG-1, each task is carefully logged and stored locally to help developers track and identify any abnormal behavior.

DAG-1 comprises 19 distinct tasks, each designed to enhance the quality and usability of the data. The airflow task are:

1) .PSV TO .CSV - This task converts individual patient files from .PSV files to .CSV format using the Google BigQuery operator. The resulting .CSV files are then saved back to GCP.

2) IF Schema Exists - This task uses an IF statement to check whether schema for validation exists in our Google Cloud Bucket or not. The workflow further proceeds only if schema exists. 

3) SCHEMA GENERATION - The input for this task is the merged .CSV file generated from the previous task. If the validation schema doesn't exists, This task constructs a schema and compiles statistics, which are subsequently used for data validation to prevent model crashes caused by incorrect data entry. The schema includes column names and data types, while the statistics encompass null count, minimum, maximum, mean, median, and standard deviation for each column. This schema is then produced in a .JSON file.

4) PUSH .JSON TO GCP - This task pushes .JSON schema file from the previous process to GCP. 

5) PULLING .JSON FOR DATA VALIDATION - This task retrieves the .JSON file from GCP for data validation. The validation is performed according to the schema defined in the previous process. The outcome of this validation is a boolean value: True or False. If each data point passes schema  and statastics validation it retuens True, the process proceeds; otherwise, it halts the further processing of DAG-1 thereby preventing the pipeline from crashing.

6) PREPARING EMAIL CONTENT - This task creates an email notifying the developer that the data validation has failed and the workflow has stopped working to avoid model from failing. Data validation is said to fail when the data does not satisfy the pre-defined schema and data distribution. 

7) EMAIL VALIDATION FAILED - This process is triggered when the previous process is successful. The email created by airflow using the email content in the previous task is sent to the developers to alert them in case of data anomaly. 

8) TRAIN - TEST SPLIT - This process pulls the .CSV file and splits it into training and testing sets. The resulting sets are then saved locally.

9) PRE - PROCESSING - After the train-test split, X-train and X-test undergo preprocessing. Two separate processes are created: one for preprocessing the training set and another for preprocessing the test set.

10) SCALING TRAIN AND TEST SETS - In this task, the pre-processed training and test datasets are scaled. Separate procedures are established: one for scaling the training set and another for scaling the test set.

11) PUSHING FILES TO GCP - Five distinct tasks are established to upload X_train, X_test, Y_train, Y_test, and the scaler to GCP. These files are uploaded in .pkl format for efficiency.

12) CLEANING - Finally, all the .pkl files are cleaned and finalized for modeling in DAG-2.

13) TRIGGER DAG - 2 - This task is used to trigger DAG - 2 after successful completion of DAG - 1.


### GCS Bucket Folder Structure
![image](https://github.com/Rishab-KH/IE7374-Sepsis-Classification/assets/40423823/5ecbd9dd-cb8d-4a2f-ad84-ef792a1d11b9)
The [link](https://console.cloud.google.com/storage/browser/sepsis-prediction-mlops/data/processed_data) to our bucket


## Installation

You can directly view our server hosted Airflow instance [here](http://35.193.213.112:8080/home)

Contact [Our Team](mailto:raju.d@northeastern.edu?cc=senthil.sh@northeastern.edu,khuba.r@northeastern.edu,sarda.h@northeastern.edu,dube.ra@northeastern.edu) for credentials

### Dependencies
- git
- Docker, Installation guide [here](https://docs.docker.com/engine/install/) 

### Steps

Clone the repository:
```shell
git clone https://github.com/Rishab-KH/IE7374-Sepsis-Classification.git
```

You will need a `development.env` file in your root directory which has details about the SMTP server and Airflow connections to GCP.

Below is a template how it might look like

```shell
AIRFLOW_UID=50000
AIRFLOW__SMTP__SMTP_HOST=<smtp_host>
AIRFLOW__SMTP__SMTP_STARTTLS=True
AIRFLOW__SMTP__SMTP_SSL=False
AIRFLOW__SMTP__SMTP_USER=<your_email>
AIRFLOW__SMTP__SMTP_PASSWORD=<your app password>
AIRFLOW__SMTP__SMTP_PORT=<smtp_port>
AIRFLOW__SMTP__SMTP_MAIL_FROM=<your_email>
AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT=google-cloud-platform://?extra__google_cloud_platform__key_path=%2Fkeys%2Fkey.json&extra__google_cloud_platform__scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform&extra__google_cloud_platform__project=airflow&extra__google_cloud_platform__num_retries=5
```

Contact [Our Team](mailto:raju.d@northeastern.edu?cc=senthil.sh@northeastern.edu,khuba.r@northeastern.edu,sarda.h@northeastern.edu,dube.ra@northeastern.edu)
 for the `development.env` that we are using

To build the docker image:
```shell
docker compose build
```

To Run the docker image:
```shell
docker compose up
```

To Run the docker image in background:
```shell
docker compose up -d
```

To stop the docker image:
```shell
docker compose down
```

Once the process is running, login with credentials `airflow` and `airflow`
```shell
http://locahost:8080
```

Note: Make sure to run the above `docker compose` commands inside your project root, i.e wherever the `docker-compose.yml` file is placed

## Peformance Optimization

In our initial attempt to manage a substantial dataset consisting of 40,000 individual files, we employed a PythonOperator within Apache Airflow to manually fetch and sequentially append each file from the cloud. This approach, however, turned out to be highly resource-intensive and inefficient, as evidenced by the prolonged execution times depicted in the Gantt chart below. Python's handling of the data did not meet our performance expectations, necessitating a search for more robust processing solutions.

![image](https://github.com/Rishab-KH/IE7374-Sepsis-Classification/assets/47169600/a36366fd-0906-4fc5-8795-9965867c66e1)
The above Gantt chart highlights the bottleneck in our pipeline

### Improvements with Google Cloud Platform
#### Optimized Data Processing
To enhance performance and manage the extensive dataset of 40,000 files more effectively, we transitioned from a sequential Python-based processing approach to a more robust solution leveraging Google Cloud's BigTable and Cloud Storage. This migration aimed to utilize the scalable infrastructure of GCP to expedite data handling and analysis.

##### Creating an External Table
The first step in our optimized pipeline involved creating an external table in BigQuery, which directly references the raw data stored in Google Cloud Storage. This approach eliminates the need to load data into BigQuery, reducing both the time and computational overhead associated with importing data.
```sql
CREATE OR REPLACE EXTERNAL TABLE sepsis.dataset_temporary (
    HR FLOAT64, O2Sat FLOAT64, Temp FLOAT64, SBP FLOAT64, MAP FLOAT64, DBP FLOAT64,
    Resp FLOAT64, EtCO2 FLOAT64, BaseExcess FLOAT64, HCO3 FLOAT64, FiO2 FLOAT64,
    pH FLOAT64, PaCO2 FLOAT64, SaO2 FLOAT64, AST FLOAT64, BUN FLOAT64,
    Alkalinephos FLOAT64, Calcium FLOAT64, Chloride FLOAT64, Creatinine FLOAT64,
    Bilirubin_direct FLOAT64, Glucose FLOAT64, Lactate FLOAT64, Magnesium FLOAT64,
    Phosphate FLOAT64, Potassium FLOAT64, Bilirubin_total FLOAT64, TroponinI FLOAT64,
    Hct FLOAT64, Hgb FLOAT64, PTT FLOAT64, WBC FLOAT64, Fibrinogen FLOAT64,
    Platelets FLOAT64, Age FLOAT64, Gender INT64, Unit1 FLOAT64, Unit2 FLOAT64,
    HospAdmTime FLOAT64, ICULOS INT64, SepsisLabel INT64
)
OPTIONS (
    format = 'CSV',
    uris = ['gs://sepsis-prediction-mlops/data/initial/*.psv'],
    skip_leading_rows = 1,
    field_delimiter="|"
);
```

##### Data Export
After configuring the external table to reference our raw data, we performed transformations directly within BigQuery and exported the final dataset back to Cloud Storage. This method is highly efficient due to BigQuery's powerful data processing capabilities.

```sql
EXPORT DATA OPTIONS(
    uri='gs://sepsis-prediction-mlops/data/modified_data/finalDataset-*.csv',
    format='CSV',
    overwrite=true,
    header=true,
    field_delimiter=','
) AS
SELECT *, REGEXP_EXTRACT(_FILE_NAME, r'([^/]+)\.psv$') AS Patient_ID
FROM sepsis.dataset_temporary LIMIT 9223372036854775807;
```
![image](https://github.com/Rishab-KH/IE7374-Sepsis-Classification/assets/47169600/4af31f53-2e9c-45d1-b12d-9dbff672ac1d)
The above Gantt chart highlights significant improvement in speeds of processing the PSV to CSV data, removing the bottleneck 

By leveraging Google Cloud's powerful data warehousing and storage solutions, we have significantly reduced the time and resources required to process large-scale datasets. This optimized approach not only speeds up the data processing workflow but also enhances the scalability and manageability of our sepsis prediction project.

## Our Team

[Sharanya Senthil](mailto:senthil.sh@northeastern.edu)

[Rishab Khuba](mailto:khuba.r@northeastern.edu)

[Hrithik Sarda](mailto:sarda.h@northeastern.edu)

[Raghavi Dube](mailto:dube.ra@northeastern.edu)

[Deril Raju](mailto:raju.d@northeastern.edu)

