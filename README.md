# IAI600 – Diabetes Health Indicators – Group 3

Machine Learning Project | University West | Fall 2026

## Project Overview

This project investigates whether machine learning models can classify diabetes-related status using health and lifestyle indicators.

We compare two supervised learning algorithms:

- Logistic Regression
- Random Forest

## Dataset

**Source:** [Diabetes Health Indicators – Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

The dataset originates from the 2015 Behavioral Risk Factor Surveillance System (BRFSS).

**Dataset file:** `diabetes_binary_health_indicators_BRFSS2015.csv`

| Property | Value |
|---|---|
| Observations | 253,680 |
| Input features | 21 |
| Target variable | Diabetes_binary |
| Class 0 | 86.07% |
| Class 1 | 13.93% |

## Methodology

1. Dataset inspection and Exploratory Data Analysis (EDA)
2. Data quality checks and preprocessing
3. Stratified train/test split (80/20)
4. Logistic Regression training
5. Random Forest training
6. Model evaluation and comparison
7. Feature importance analysis

## Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8634 | 0.8594 |
| Precision (Class 1) | 0.5338 | 0.4876 |
| Recall (Class 1) | 0.1532 | 0.1784 |
| F1-score (Class 1) | 0.2381 | 0.2612 |
| ROC-AUC | 0.8209 | 0.7944 |
| Average Precision | 0.4022 | 0.3678 |

### Main Findings

**Logistic Regression** achieved higher ROC-AUC and Average Precision.

**Random Forest** achieved slightly higher recall and F1-score for Class 1.

Both models struggled to identify the minority class at the default classification threshold.

## How to Run

1. Download `Diabetes_Project.ipynb` from this repository.
2. Download the CSV dataset from Kaggle.
3. Install Python, Jupyter Notebook, Pandas, Matplotlib and Scikit-Learn.
4. Place the CSV in an accessible folder.
5. Adjust the CSV file path in the notebook if necessary.
6. Run all notebook cells.

## Limitations

- Survey data is self-reported.
- The positive class combines prediabetes and diabetes.
- The dataset is imbalanced.
- Models were evaluated using a single train/test split.
- Feature importance does not establish causation.
- The models are not intended for medical diagnosis.

## Group Members

- Sravani Manda
- Ndifreke Gregory Bassey
- Maju Anand
- Daniel Sandström Hasselblad

## Acknowledgment

ChatGPT was used to support learning, Python development, debugging and documentation. All group members are responsible for understanding and validating the submitted work.

## Course Information

**Course:** IAI600 – Introduction to Applied Artificial Intelligence and Machine Learning

**University:** University West (Högskolan Väst), Sweden

**Semester:** Fall 2026
