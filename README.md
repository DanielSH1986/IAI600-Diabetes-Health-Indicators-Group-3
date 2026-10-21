IAI600 – Diabetes Health Indicators – Group 3

Project Overview

This project was developed as part of the course Introduction to Applied Artificial Intelligence and Machine Learning (IAI600) at University West, Sweden.

The objective is to investigate whether machine learning algorithms can classify diabetes-related status based on health and lifestyle indicators.

We compare two supervised machine learning algorithms:

Logistic Regression

Random Forest

Dataset

Source: Diabetes Health Indicators Dataset – Kaggle

The dataset is derived from the 2015 Behavioral Risk Factor Surveillance System (BRFSS).

We use the file diabetes_binary_health_indicators_BRFSS2015.csv.

Observations: 253,680

Input features: 21

Target variable: Diabetes_binary

Class 0: No diabetes (86.07%)

Class 1: Prediabetes or diabetes (13.93%)

Methodology

The project follows these main steps:

Dataset inspection and Exploratory Data Analysis (EDA).

Data-quality checks, including missing values and duplicate records.

Data preprocessing using StandardScaler, OneHotEncoder, and binary feature passthrough for Logistic Regression.

Stratified training/test split (80% / 20%).

Training Logistic Regression and Random Forest.

Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and Average Precision.

Feature importance analysis using Random Forest.

Model Results

Metric

Logistic Regression

Random Forest

Accuracy

0.8634

0.8594

Precision (Class 1)

0.5338

0.4876

Recall (Class 1)

0.1532

0.1784

F1-score (Class 1)

0.2381

0.2612

ROC-AUC

0.8209

0.7944

Average Precision

0.4022

0.3678

Logistic Regression achieved higher ROC-AUC and Average Precision, while Random Forest achieved slightly higher recall and F1-score for Class 1.

Both models showed limited recall for the minority class at the default classification threshold.

How to Run the Project

Download Diabetes_Project.ipynb from this repository.

Download diabetes_binary_health_indicators_BRFSS2015.csv from the Kaggle link above.

Install Python, Jupyter Notebook, Pandas, NumPy, Matplotlib, and Scikit-Learn.

Place the dataset in a folder accessible to the notebook.

Update the pd.read_csv() file path if necessary.

Open the notebook in Jupyter and run all cells.

Project Limitations

The dataset contains self-reported survey information.

The target combines prediabetes and diabetes into one positive class.

The dataset has a substantial class imbalance.

Model evaluation was performed using a single stratified 80/20 split.

The models were not systematically hyperparameter-tuned.

Feature importance does not establish causation.

The models are educational classifiers and are not intended for medical diagnosis.

Group Members

Sravani Manda

Ndifreke Gregory Bassey

Maju Anand

Daniel Sandström Hasselblad

Acknowledgment

Generative AI tools, including ChatGPT, were used to support learning, explain machine learning concepts, assist with Python programming, and help document the project. The group members remain responsible for reviewing, understanding, and validating the submitted work.

Course

IAI600 – Introduction to Applied Artificial Intelligence and Machine Learning

University West (Högskolan Väst), Sweden

Fall 2026
