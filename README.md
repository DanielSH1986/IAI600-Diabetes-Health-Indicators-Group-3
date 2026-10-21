# IAI600 – Diabetes Health Indicators – Group 3

Machine Learning Project | University West | Fall 2026

## Project Overview

This project investigates whether machine learning models can predict recorded diabetes-related status using health and lifestyle indicators from the 2015 Behavioral Risk Factor Surveillance System (BRFSS).

### Research Question

**To what extent can a reduced set of BRFSS health indicators predict recorded diabetes status, and which indicators contribute most to predictive performance?**

To investigate this question, we compare two supervised machine learning algorithms:

- Logistic Regression
- Random Forest

Each algorithm is trained using two different sets of input features:

- **Full feature set:** All 21 BRFSS health indicators.
- **Reduced feature set:** Five indicators selected using Random Forest feature importance calculated from the training data.

The selected indicators are BMI, Age, Income, PhysHlth, and Education.

The objective is to investigate the trade-off between reducing the number of health indicators and maintaining predictive performance.

## Dataset

**Source:** [Diabetes Health Indicators – Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

The dataset originates from the 2015 Behavioral Risk Factor Surveillance System (BRFSS).

**Dataset file:** `diabetes_binary_health_indicators_BRFSS2015.csv`

| Property | Value |
|---|---|
| Observations | 253,680 |
| Input features | 21 |
| Target variable | `Diabetes_binary` |
| Class 0 (No diabetes) | 86.07% |
| Class 1 (Prediabetes or diabetes) | 13.93% |
| Missing values | 0 |

The dataset is imbalanced, with considerably fewer observations belonging to Class 1 than Class 0.

## Methodology

The project follows these steps:

1. Dataset loading and inspection.
2. Exploratory Data Analysis (EDA) to investigate relationships between health indicators and diabetes-related status.
3. Data quality checks, including missing values, duplicate records, and feature ranges.
4. Stratified training/test split (80/20).
5. Data preprocessing, including standardization and one-hot encoding for Logistic Regression.
6. Training Logistic Regression using all 21 indicators.
7. Training Random Forest using all 21 indicators.
8. Model evaluation using six performance metrics.
9. Feature importance analysis using the trained Random Forest model.
10. Selection of the five highest-ranked indicators from the training-fitted Random Forest.
11. Training new Logistic Regression and Random Forest models using only five indicators.
12. Comparing the predictive performance of all four models.

### Selected Health Indicators

The five indicators selected for the reduced-feature experiment were:

| Indicator | Description |
|---|---|
| BMI | Body Mass Index |
| Age | Age category |
| Income | Income category |
| PhysHlth | Days of poor physical health |
| Education | Education level |

The features were ranked using Random Forest's impurity-based feature importance.

Feature selection was based on a model fitted to the training data, not on test labels. However, feature importance does not establish causality or measure each indicator's independent contribution to predictive performance.

## Experimental Setup

**Programming language:** Python

**Development environment:** Jupyter Notebook

**Main libraries:**

- Pandas
- Matplotlib
- Scikit-Learn

**Training and testing:**

- Training observations: 202,944 (80%)
- Test observations: 50,736 (20%)
- Stratified splitting to preserve the class distribution.
- `random_state=42` for reproducibility.

**Model configuration:**

- Logistic Regression: `max_iter=1000`.
- Random Forest: `n_estimators=100`, `random_state=42`, `n_jobs=-1`.

The same training and test observations were used in both the full-feature and reduced-feature experiments.

## Model Performance

The following table compares the performance of all four models on the held-out test set.

| Metric | Logistic Regression (21) | Logistic Regression (5) | Random Forest (21) | Random Forest (5) |
|---|---:|---:|---:|---:|
| Accuracy | 0.8634 | 0.8590 | 0.8594 | 0.8429 |
| Precision (Class 1) | 0.5338 | 0.4582 | 0.4876 | 0.3520 |
| Recall (Class 1) | 0.1532 | 0.0644 | 0.1784 | 0.1515 |
| F1-score (Class 1) | 0.2381 | 0.1129 | 0.2612 | 0.2118 |
| ROC-AUC | 0.8209 | 0.7675 | 0.7944 | 0.7085 |
| Average Precision | 0.4022 | 0.3253 | 0.3678 | 0.2725 |

### Main Findings

**1. Logistic Regression achieved the highest ROC-AUC.**

The Logistic Regression model using all 21 indicators achieved the best ROC-AUC (0.8209) and Average Precision (0.4022).

**2. Reducing the number of indicators decreased predictive performance.**

For Logistic Regression, ROC-AUC decreased from 0.8209 to 0.7675 when reducing the number of indicators from 21 to 5.

For Random Forest, ROC-AUC decreased from 0.7944 to 0.7085.

**3. The five selected indicators retain useful predictive information.**

The reduced Logistic Regression model achieved a ROC-AUC of 0.7675, demonstrating that the five indicators provide meaningful information for distinguishing between the two classes.

However, both reduced models performed worse than their full-feature counterparts.

**4. Minority-class detection remained challenging.**

The reduced Logistic Regression model achieved a recall of only 6.44%, compared with 15.32% for the full-feature model.

This demonstrates that overall accuracy alone is insufficient when evaluating models on imbalanced datasets.

**5. BMI, Age, and Income were the highest-ranked indicators.**

Random Forest feature importance identified the following five indicators as most important in the full-feature model:

1. BMI
2. Age
3. Income
4. PhysHlth
5. Education

These rankings reflect the model's internal feature-importance measure and should not be interpreted as causal relationships.

### Interpretation

The results suggest that a reduced set of five BRFSS indicators can provide useful discrimination between diabetes-related classes, but reducing the number of features causes a measurable loss of predictive performance.

The findings demonstrate a trade-off between model simplicity and predictive capability.

Because the original test results were examined before the reduced-feature experiment was performed, this comparison should be regarded as exploratory. Further validation would be needed to establish how well the findings generalize.

## How to Run

1. Download or clone this GitHub repository.
2. Ensure that `Diabetes_Project.ipynb` and `diabetes_binary_health_indicators_BRFSS2015.csv` are located in the same folder.
3. Install Python and Jupyter Notebook.
4. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

5. Open `Diabetes_Project.ipynb` in Jupyter Notebook.
6. Select **Kernel → Restart Kernel and Run All Cells**.
7. Review the EDA visualizations, trained models, feature importance, and four-model comparison table.

The notebook uses the following dataset path:

```python
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
```

## Limitations

- The dataset is based on self-reported survey information.
- Class 1 combines respondents reporting prediabetes or diabetes.
- The dataset is imbalanced, with approximately 13.93% belonging to Class 1.
- Both algorithms showed limited recall for the minority class.
- The evaluation used a single stratified train/test split.
- The reduced-feature comparison is exploratory because the original test results were already examined.
- No systematic hyperparameter tuning or cross-validation was performed.
- Feature importance rankings do not establish causal relationships.
- The models are intended for educational analysis, not medical diagnosis or clinical decision-making.

## Possible Future Improvements

Future work could investigate:

- Different numbers of selected features, such as 5, 10, and 15.
- Cross-validation and more systematic feature selection.
- Hyperparameter tuning.
- Class weighting to address the imbalanced dataset.
- Classification threshold adjustment to improve minority-class recall.
- Permutation importance to further investigate indicator contributions.
- Validation on an independent dataset.

## Conclusion

The project investigated whether reducing the number of BRFSS health indicators from 21 to 5 could maintain predictive performance when classifying diabetes-related status.

The results demonstrate that five selected indicators contain meaningful predictive information, but both Logistic Regression and Random Forest performed better when trained using all 21 indicators.

Logistic Regression using all 21 features achieved the strongest overall ranking performance, with a ROC-AUC of 0.8209.

The study illustrates the importance of feature selection, appropriate evaluation metrics, and careful interpretation when applying machine learning to imbalanced health-related datasets.

## Group Members

- Sravani Manda
- Ndifreke Gregory Bassey
- Maju Anand
- Daniel Sandström Hasselblad

## Acknowledgment

ChatGPT was used to support learning, Python development, debugging, analysis interpretation, and documentation. All group members are responsible for understanding, reviewing, and validating the submitted work.

## Course Information

**Course:** IAI600 – Introduction to Applied Artificial Intelligence and Machine Learning

**University:** University West (Högskolan Väst), Sweden

**Semester:** Fall 2026
