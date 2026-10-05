# Bank Personal Loan Prediction using PyCaret

## 📌 Project Overview

This project predicts whether a customer is likely to take a **Personal Loan** based on customer-related information.

This is a **Binary Classification** problem where:

* `0` → Customer did not take the Personal Loan
* `1` → Customer took the Personal Loan

The project uses **PyCaret**, an AutoML library in Python, to automate model comparison, training, tuning, prediction, and evaluation.

---

## 🎯 Objective

The main objective is to build a machine learning classification model that can identify customers who are likely to take a personal loan.

This type of model can help banks identify potential customers for targeted marketing campaigns and loan-related offers.

---

## 📂 Dataset

Dataset used:

`Bank_Loan_Granting.csv`

The dataset contains **5,000 customer records**.

### Target Distribution

| Personal Loan |     Count | Percentage |
| ------------- | --------: | ---------: |
| 0 – No Loan   |     4,520 |      90.4% |
| 1 – Loan      |       480 |       9.6% |
| **Total**     | **5,000** |   **100%** |

The target variable is imbalanced, with class `0` representing 90.4% of the dataset and class `1` representing 9.6%.

Therefore, model evaluation considers not only accuracy but also **precision, recall, F1-score, and ROC-AUC**.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyCaret
* Google Colab

---

## 🔄 Machine Learning Workflow

The project follows these steps:

```text
Load Dataset
     ↓
Data Understanding
     ↓
Data Preprocessing
     ↓
Check Target Distribution
     ↓
PyCaret Setup
     ↓
Compare Classification Models
     ↓
Create Random Forest
     ↓
Tune Random Forest
     ↓
Generate Predictions
     ↓
Evaluate Model
     ↓
Feature Importance
     ↓
ROC-AUC Analysis
```

---

## 🧹 Data Preprocessing

### 1. Load the dataset

```python
import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/Bank_Loan_Granting.csv')
```

### 2. Handle `CCAvg`

The `CCAvg` column contains values using `/` instead of `.` for decimal representation.

For example:

```text
2/3
```

is converted to:

```text
2.3
```

Code:

```python
df["CCAvg"] = df["CCAvg"].str.replace('/', '.')
df["CCAvg"] = df["CCAvg"].astype(float)
```

### 3. Remove unwanted column

```python
df = df.iloc[:, 1:]
```

### 4. Check target distribution

```python
print(df["Personal Loan"].value_counts())

print(df["Personal Loan"].value_counts(normalize=True) * 100)
```

---

# 🤖 PyCaret Implementation

## 1. Setup

```python
from pycaret.classification import *

clf = setup(
    data=df,
    target='Personal Loan',
    session_id=34
)
```

`setup()` initializes the PyCaret classification environment and prepares the dataset for machine learning.

---

## 2. Compare Machine Learning Models

```python
best_model = compare_models()
```

`compare_models()` trains and compares multiple classification algorithms using cross-validation.

This allows us to identify which models perform well on the dataset.

---

## 3. Create Random Forest

```python
rf = create_model('rf')
```

A Random Forest classifier is created using PyCaret.

---

## 4. Tune Random Forest

```python
tuned_rf = tune_model(rf)
```

Hyperparameters are automatically tuned to improve model performance.

---

## 5. Evaluate the Model

```python
evaluate_model(tuned_rf)
```

This provides different evaluation plots, including:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance
* Learning Curve
* Validation Curve

---

## 6. Generate Predictions

```python
predictions = predict_model(tuned_rf)

predictions.head()
```

The prediction output contains the actual target, predicted label, and prediction score.

Example:

| Personal Loan | prediction_label | prediction_score |
| ------------: | ---------------: | ---------------: |
|             0 |                0 |             0.98 |
|             1 |                1 |             0.91 |
|             0 |                0 |             0.99 |

---

# 📊 Model Evaluation

The model is evaluated using:

### Accuracy

Measures the percentage of total predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many customers predicted as loan customers actually belong to class `1`.

### Recall

Measures how many actual loan customers were successfully identified by the model.

For this project, **class 1 recall is particularly important** because identifying potential loan customers is the business objective.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures how well the model separates class `0` and class `1` across different classification thresholds.

---

## 📋 Classification Report

The model can be evaluated using:

```python
from sklearn.metrics import classification_report

print(classification_report(
    predictions['Personal Loan'],
    predictions['prediction_label']
))
```

The classification report provides:

```text
Precision
Recall
F1-score
Support
```

for both classes.

---

## 🔲 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    predictions['Personal Loan'],
    predictions['prediction_label']
)

print(cm)
```

The confusion matrix contains:

```text
True Negative (TN)
False Positive (FP)
False Negative (FN)
True Positive (TP)
```

For this business problem, **False Negatives are important** because they represent customers who actually took a loan but were predicted as not taking one.

---

# 📈 Feature Importance

```python
plot_model(tuned_rf, plot='feature')
```

Feature importance helps identify which customer attributes have the greatest influence on the Random Forest predictions.

---

# 📉 ROC-AUC Curve

```python
plot_model(tuned_rf, plot='auc')
```

The ROC-AUC curve evaluates the model's ability to distinguish between:

```text
Class 0 → No Personal Loan
Class 1 → Personal Loan
```

---

# 📊 Confusion Matrix Visualization

```python
plot_model(tuned_rf, plot='confusion_matrix')
```

---

# 🚀 How to Run the Project

### Step 1 — Clone the repository

```bash
git clone <your-github-repository-url>
```

### Step 2 — Open the project

Open the project in:

* Jupyter Notebook
* Google Colab
* VS Code

### Step 3 — Install dependencies

```bash
pip install pandas numpy scikit-learn pycaret
```

For Google Colab:

```python
!pip install pycaret
```

### Step 4 — Place the dataset

Keep the dataset inside:

```text
Datasets/
└── Bank_Loan_Granting.csv
```

### Step 5 — Run the notebook

Run the cells sequentially to:

1. Load the data
2. Preprocess the data
3. Set up PyCaret
4. Compare models
5. Create Random Forest
6. Tune the model
7. Generate predictions
8. Evaluate the model

---

# 📁 Project Structure

```text
Bank-Loan-Prediction/
│
├── Datasets/
│   └── Bank_Loan_Granting.csv
│
├── notebooks/
│   └── Bank_Loan_PyCaret.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# 💡 Key Learning Outcomes

Through this project, I learned:

* Classification problem identification
* Data loading and exploration
* Data preprocessing
* Handling numeric data
* Target variable analysis
* Class imbalance
* PyCaret `setup()`
* Model comparison using `compare_models()`
* Random Forest classification
* Hyperparameter tuning
* Model prediction
* Confusion Matrix
* Precision
* Recall
* F1-score
* ROC-AUC
* Feature importance
* AutoML workflow

---

# 🔍 Manual Scikit-learn vs PyCaret

### Manual approach

```text
train_test_split()
        ↓
RandomForestClassifier()
        ↓
fit()
        ↓
predict()
        ↓
accuracy_score()
        ↓
classification_report()
```

### PyCaret approach

```text
setup()
   ↓
compare_models()
   ↓
create_model()
   ↓
tune_model()
   ↓
predict_model()
   ↓
evaluate_model()
```

PyCaret reduces the amount of repetitive code required for model experimentation while still allowing the underlying machine learning concepts to be understood and analyzed.

---

# 📌 Conclusion

This project demonstrates an end-to-end **Bank Personal Loan Classification** workflow using PyCaret.

The dataset is highly imbalanced, so model performance should not be judged using accuracy alone. Precision, recall, F1-score, ROC-AUC, and the confusion matrix provide a more complete understanding of the model's performance, particularly for the minority class `Personal Loan = 1`.

---

## 👩‍💻 Author

**Ananthi P**

Machine Learning | GenAI 
