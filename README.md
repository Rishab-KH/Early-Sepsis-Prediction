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
- Size: 1552210 rows × 42 columns
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

## Our Team

[Sharanya Senthil](mailto:senthil.sh@northeastern.edu)

[Rishab Khuba](mailto:khuba.r@northeastern.edu)

[Hrithik Sarda](mailto:sarda.h@northeastern.edu)

[Raghavi Dube](mailto:dube.ra@northeastern.edu)

[Deril Raju](mailto:raju.d@northeastern.edu)

