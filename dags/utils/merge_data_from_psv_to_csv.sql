CREATE OR REPLACE EXTERNAL TABLE sepsis.dataset_temporary (
    HR FLOAT64,
    O2Sat FLOAT64,
    Temp FLOAT64,
    SBP FLOAT64,
    MAP FLOAT64,
    DBP FLOAT64,
    Resp FLOAT64,
    EtCO2 FLOAT64,
    BaseExcess FLOAT64,
    HCO3 FLOAT64,
    FiO2 FLOAT64,
    pH FLOAT64,
    PaCO2 FLOAT64,
    SaO2 FLOAT64,
    AST FLOAT64,
    BUN FLOAT64,
    Alkalinephos FLOAT64,
    Calcium FLOAT64,
    Chloride FLOAT64,
    Creatinine FLOAT64,
    Bilirubin_direct FLOAT64,
    Glucose FLOAT64,
    Lactate FLOAT64,
    Magnesium FLOAT64,
    Phosphate FLOAT64,
    Potassium FLOAT64,
    Bilirubin_total FLOAT64,
    TroponinI FLOAT64,
    Hct FLOAT64,
    Hgb FLOAT64,
    PTT FLOAT64,
    WBC FLOAT64,
    Fibrinogen FLOAT64,
    Platelets FLOAT64,
    Age FLOAT64,
    Gender INT64,
    Unit1 FLOAT64,
    Unit2 FLOAT64,
    HospAdmTime FLOAT64,
    ICULOS INT64,
    SepsisLabel INT64
)
OPTIONS (
format = 'CSV',
uris = ['gs://sepsis-prediction-mlops/data/initial/*.psv'],
skip_leading_rows = 1,
field_delimiter="|"
);

EXPORT DATA OPTIONS(
uri='gs://sepsis-prediction-mlops/data/modified_data/finalDataset-*.csv',
format='CSV',
overwrite=true,
header=true,
field_delimiter=',') AS
SELECT *, REGEXP_EXTRACT(_FILE_NAME, r'([^/]+)\.psv$') AS Patient_ID
FROM sepsis.dataset_temporary LIMIT 9223372036854775807;

DROP TABLE IF EXISTS sepsis.dataset_temporary;