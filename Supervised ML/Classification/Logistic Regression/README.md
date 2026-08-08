# 📊 Logistic Regression

## Overview

Logistic Regression is one of the most widely used supervised Machine Learning algorithms for solving **classification problems**. Unlike Linear Regression, which predicts continuous numerical values, Logistic Regression predicts **categorical outcomes**, making it an ideal algorithm for binary classification tasks.

Logistic Regression estimates the probability that an observation belongs to a particular class using the **Sigmoid (Logistic) Function**. Based on this probability, the model classifies observations into one of two categories.

This project demonstrates how to build, preprocess, train, evaluate, and visualize a Logistic Regression model using **Python** and **Scikit-learn** on the **Social Network Ads Dataset**. The project covers data preprocessing, feature scaling, model training, probability prediction, confusion matrix, ROC Curve, and various classification evaluation metrics.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Classification?
- What is Logistic Regression?
- Why Use Logistic Regression?
- Sigmoid Function
- Difference Between Linear Regression and Logistic Regression
- Assumptions of Logistic Regression
- Project Workflow
- Dataset Information
- Technologies Used
- Project Structure
- Step-by-Step Notebook Explanation (Cells 1–10)

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions or decisions without being explicitly programmed.

Instead of following predefined rules, Machine Learning algorithms identify relationships within data and continuously improve their performance as more data becomes available.

Machine Learning is broadly divided into three categories:

- **Supervised Learning**
- **Unsupervised Learning**
- **Reinforcement Learning**

Since the **Social Network Ads Dataset** contains labeled data with a target variable (**Purchased**), **Logistic Regression** belongs to the **Supervised Learning** category.

---

# 📊 What is Classification?

Classification is a supervised Machine Learning technique used to predict **categorical outputs**.

Instead of predicting numerical values, classification algorithms assign data into predefined classes.

Some common classification problems include:

- Email Spam Detection
- Disease Prediction
- Customer Purchase Prediction
- Loan Approval Prediction
- Credit Card Fraud Detection
- Sentiment Analysis

The objective of classification is to learn the relationship between input features and categorical target labels.

---

# 📈 What is Logistic Regression?

Logistic Regression is a supervised Machine Learning algorithm used for **binary classification** problems.

Rather than predicting a continuous value, Logistic Regression predicts the probability that an observation belongs to a particular class.

The predicted probability always lies between **0 and 1**.

The algorithm uses the **Sigmoid Function** to transform the linear output into a probability.

---

### In this project

**Independent Variables (Features):**

- Gender
- Age
- Estimated Salary

**Dependent Variable (Target):**

- Purchased

The model learns the relationship between customer characteristics and purchasing behavior to predict whether a customer will purchase a product.

---

# 🎯 Why Use Logistic Regression?

Logistic Regression is particularly useful when:

- The target variable is categorical.
- Binary classification is required.
- Probability estimation is important.
- The model should be simple and interpretable.
- Fast training and prediction are required.

Common applications include:

- Customer Purchase Prediction
- Medical Diagnosis
- Spam Email Detection
- Credit Risk Analysis
- Fraud Detection
- Employee Attrition Prediction
- Marketing Analytics
- Customer Churn Prediction

---

# 📐 Sigmoid Function

The mathematical equation of the Sigmoid Function is:

```text
P(Y=1)=1/(1+e^(-z))
```

### Where

| Symbol | Description |
|---------|-------------|
| P(Y=1) | Probability of belonging to Class 1 |
| e | Euler's Number |
| z | Linear Combination of Features |

The Sigmoid Function converts any real-valued number into a probability between **0 and 1**, making it suitable for binary classification.

---

# 🔄 Difference Between Linear Regression and Logistic Regression

| Feature | Linear Regression | Logistic Regression |
|----------|-------------------|---------------------|
| Problem Type | Regression | Classification |
| Output | Continuous Value | Probability / Class |
| Target Variable | Numerical | Categorical |
| Function Used | Linear Equation | Sigmoid Function |
| Output Range | Any Value | 0 to 1 |
| Evaluation Metrics | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |

Logistic Regression should be preferred whenever the target variable contains categories instead of continuous numerical values.

---

# 📋 Assumptions of Logistic Regression

For Logistic Regression to perform effectively, the following assumptions should generally hold:

- The dependent variable should be categorical.
- Observations should be independent.
- There should be little or no multicollinearity among features.
- Independent variables should have a linear relationship with the log odds.
- Outliers should be minimal.
- Features should preferably be scaled when their ranges differ significantly.

---

# 🔄 Project Workflow

```text
Import Libraries
        │
        ▼
Load Dataset
        │
        ▼
Explore Dataset
        │
        ▼
Check Missing Values
        │
        ▼
Encode Categorical Features
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Train Logistic Regression Model
        │
        ▼
Predict Classes
        │
        ▼
Evaluate Model
        │
        ▼
Visualize Results
```
# 📂 Dataset Information

**Dataset Name:** Social_Network_Ads.csv

The dataset contains demographic information about customers and whether they purchased a product after viewing a social network advertisement.

## Dataset Summary

| Property | Value |
|----------|-------|
| Number of Rows | 400 |
| Number of Columns | 5 |
| Missing Values | No |
| Numerical Features | 2 |
| Categorical Features | 1 |
| Target Variable | Purchased |

---

## Features

| Feature | Description |
|----------|-------------|
| User ID | Unique identifier for each customer |
| Gender | Gender of the customer |
| Age | Age of the customer |
| Estimated Salary | Annual estimated salary |
| Purchased | Whether the customer purchased the product (0 = No, 1 = Yes) |

The **Purchased** column is the dependent variable (target), while **Gender**, **Age**, and **Estimated Salary** are used as independent variables for model training. The **User ID** column is only an identifier and is not used for prediction.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 📁 Project Structure

```text
Logistic-Regression/
│
├── Logistic Regression.ipynb
├── Social_Network_Ads.csv
├── README.md
├── requirements.txt
└── LICENSE
```

---

# 📖 Step-by-Step Notebook Explanation

---

## 🔹 Cell 1 – Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
```

### Purpose

Each library has a specific responsibility.

| Library | Purpose |
|----------|----------|
| NumPy | Numerical computations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| train_test_split | Split dataset into training and testing sets |
| LabelEncoder | Convert categorical values into numerical values |
| StandardScaler | Standardize feature values |
| LogisticRegression | Train the Logistic Regression model |

---

## 🔹 Cell 2 – Load Dataset

```python
df = pd.read_csv("Social_Network_Ads.csv")
```

The dataset is loaded into a Pandas DataFrame.

After loading, the first five rows are displayed using:

```python
df.head()
```

Displaying the first few rows helps verify that the dataset has been loaded correctly and provides an overview of its structure.

---

## 🔹 Cell 3 – Display Dataset Information

```python
df.info()
```

This provides:

- Number of rows
- Number of columns
- Data types
- Missing values
- Memory usage

Understanding the dataset structure before preprocessing and model training is an essential step in every Machine Learning project.

---

## 🔹 Cell 4 – Statistical Summary

```python
df.describe()
```

This generates summary statistics such as:

- Count
- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles

These statistics help understand the distribution of numerical features and identify possible outliers.

---

## 🔹 Cell 5 – Check Missing Values

```python
df.isnull().sum()
```

Missing values can negatively affect the performance of Machine Learning models.

This step checks every column to ensure that the dataset is complete before preprocessing.

For this dataset, all columns contain complete data with no missing values.

---

## 🔹 Cell 6 – Dataset Shape

```python
df.shape
```

This returns:

- Number of rows
- Number of columns

For this dataset:

- Rows: **400**
- Columns: **5**

Knowing the dataset dimensions helps understand the size of the training data before building the model.

---

## 🔹 Cell 7 – Separate Features and Target

```python
X = df.iloc[:,1:4]
y = df.iloc[:,-1]
```

The dataset is divided into:

- **X (Independent Variables):** Gender, Age, Estimated Salary
- **y (Dependent Variable):** Purchased

Separating the features and target variable is necessary before preprocessing and training the Logistic Regression model.

---

## 🔹 Cell 8 – Encode the Gender Column

```python
encoder = LabelEncoder()

X["Gender"] = encoder.fit_transform(X["Gender"])
```

Machine Learning models require numerical input.

Since the **Gender** column contains categorical values, Label Encoding converts it into numerical values that the model can process.

---

## 🔹 Cell 9 – Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

The dataset is divided into two parts:

- **Training Set (75%)** – Used to train the Logistic Regression model.
- **Testing Set (25%)** – Used to evaluate the model on unseen data.

Using separate training and testing datasets helps determine how well the model generalizes to new observations.

---

## 🔹 Cell 10 – Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Feature Scaling is one of the most important preprocessing steps for Logistic Regression.

Since **Age** and **Estimated Salary** have different numerical ranges, StandardScaler transforms every feature so that they have:

- Mean = 0
- Standard Deviation = 1

Scaling ensures that all features contribute equally during model training and improves the convergence and performance of the Logistic Regression algorithm.
---

## 🔹 Cell 11 – Train the Logistic Regression Model

```python
model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)
```

After preprocessing the dataset, the Logistic Regression model is trained using the training dataset.

During training, the model learns the relationship between the independent variables (**Gender**, **Age**, and **Estimated Salary**) and the target variable (**Purchased**).

Once trained, the model can predict whether a new customer is likely to purchase the advertised product.

---

## 🔹 Cell 12 – Predict Test Data

```python
y_pred = model.predict(X_test)
```

The trained Logistic Regression model predicts the class labels for the unseen testing dataset.

The predicted values are stored in **`y_pred`**, which are later compared with the actual target values to evaluate the model's performance.

---

## 🔹 Cell 13 – Predict Class Probabilities

```python
y_prob = model.predict_proba(X_test)
```

Instead of predicting only the class label, Logistic Regression can also estimate the probability that each observation belongs to each class.

The output contains:

- Probability of Class 0 (Not Purchased)
- Probability of Class 1 (Purchased)

These probability values are useful when confidence scores are required instead of only class predictions.

---

## 🔹 Cell 14 – Compare Actual and Predicted Values

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.head()
```

This step creates a comparison table containing:

- Actual Purchase Status
- Predicted Purchase Status

Comparing these values provides a simple way to understand how accurately the Logistic Regression model classifies customers.

---

## 🔹 Cell 15 – Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The Confusion Matrix summarizes the classification performance of the model.

It contains:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

A heatmap visualization is used to clearly display the prediction results.

---

## 🔹 Cell 16 – Accuracy Score

```python
accuracy = accuracy_score(y_test, y_pred)
```

Accuracy measures the proportion of correctly classified observations.

### Formula

```text
Accuracy = (TP + TN) / Total Predictions
```

A higher accuracy indicates better classification performance.

---

## 🔹 Cell 17 – Classification Report

```python
print(classification_report(y_test, y_pred))
```

The Classification Report provides several important evaluation metrics:

- Precision
- Recall
- F1-Score
- Support

These metrics give a more detailed understanding of model performance than accuracy alone.

---

## 🔹 Cell 18 – ROC Curve and AUC Score

```python
roc_curve()
roc_auc_score()
```

The ROC (Receiver Operating Characteristic) Curve illustrates the performance of the classifier at different classification thresholds.

The AUC (Area Under Curve) Score measures how well the model distinguishes between the two classes.

Interpretation:

- AUC = 1 → Perfect Classification
- AUC = 0.5 → Random Guessing

Higher AUC values indicate a better classifier.

---

## 🔹 Cell 19 – Predict a New Customer

```python
new_customer = [[1,35,60000]]

prediction = model.predict(new_customer)
```

Finally, the trained Logistic Regression model is used to predict whether a new customer is likely to purchase the advertised product.

Before prediction, the customer data is standardized using the same **StandardScaler** that was fitted on the training dataset.

This ensures consistency between training data and new input data.

---

## 🔹 Cell 20 – Decision Boundary Visualization

The final step visualizes the decision boundary learned by the Logistic Regression model.

The graph displays:

- Class 0 observations
- Class 1 observations
- Decision Boundary

This visualization helps understand how Logistic Regression separates the two classes using the selected features.

---

# 📊 Model Evaluation Metrics

After training the Logistic Regression model, it is important to evaluate its classification performance.

In this project, the following evaluation metrics are used:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score

These metrics provide a comprehensive understanding of how well the classifier performs on unseen data.

---

## 📈 Accuracy

Accuracy measures the percentage of correctly classified observations.

### Formula

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Interpretation

- Higher Accuracy indicates better classification performance.
- Accuracy is most useful when the dataset is balanced.

---

## 📈 Precision

Precision measures how many predicted positive observations are actually positive.

### Formula

```text
Precision = TP / (TP + FP)
```

### Interpretation

High Precision means fewer False Positives.

---

## 📈 Recall

Recall measures how many actual positive observations are correctly identified.

### Formula

```text
Recall = TP / (TP + FN)
```

### Interpretation

High Recall means fewer False Negatives.

---

## 📈 F1-Score

The F1-Score is the harmonic mean of Precision and Recall.

### Formula

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Interpretation

A higher F1-Score indicates a better balance between Precision and Recall.

---

## 📈 ROC Curve

The ROC Curve compares:

- True Positive Rate
- False Positive Rate

A curve closer to the top-left corner represents better classification performance.

---

## 📈 AUC Score

The Area Under the ROC Curve (AUC) measures the classifier's ability to distinguish between positive and negative classes.

| AUC Score | Performance |
|-----------|-------------|
| 1.0 | Excellent |
| 0.9 – 1.0 | Outstanding |
| 0.8 – 0.9 | Very Good |
| 0.7 – 0.8 | Good |
| Below 0.5 | Poor |

---

# 🌍 Real-World Applications

Logistic Regression is widely used in real-world binary classification problems.

Some common applications include:

- Customer Purchase Prediction
- Email Spam Detection
- Disease Diagnosis
- Credit Card Fraud Detection
- Loan Approval Prediction
- Employee Attrition Prediction
- Customer Churn Prediction
- Marketing Campaign Analysis
- Risk Assessment
- Healthcare Analytics

---

# ✅ Advantages of Logistic Regression

- Simple and easy to implement.
- Works well for binary classification problems.
- Provides probability estimates.
- Computationally efficient.
- Easy to interpret.
- Performs well on linearly separable datasets.
- Supported by Scikit-learn.

---

# ❌ Limitations of Logistic Regression

- Suitable mainly for linear decision boundaries.
- Performance decreases with highly complex relationships.
- Sensitive to outliers.
- Requires feature scaling for optimal performance.
- Not suitable for complex non-linear datasets without feature engineering.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Logistic-Regression.git
```

Navigate to the project directory:

```bash
cd Logistic-Regression
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Logistic Regression.ipynb
```

---

# 📦 Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

Install all dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand the concept of Logistic Regression.
- Understand binary classification problems.
- Load and explore datasets using Pandas.
- Encode categorical variables using Label Encoding.
- Apply Feature Scaling using StandardScaler.
- Split datasets into training and testing sets.
- Train a Logistic Regression model.
- Predict class labels and probabilities.
- Evaluate classification models using Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC Score.
- Visualize the decision boundary of a Logistic Regression classifier.
- Apply Logistic Regression to real-world customer purchase prediction problems.

---

# 📝 Conclusion

Logistic Regression is one of the most fundamental and widely used classification algorithms in Machine Learning. It predicts the probability of an observation belonging to a particular class using the Sigmoid Function, making it highly effective for binary classification tasks.

In this project, the Social Network Ads dataset was explored, categorical data was encoded, numerical features were standardized, a Logistic Regression model was trained, predictions were generated, classification metrics were evaluated, and the decision boundary was visualized.

The project demonstrates a complete Machine Learning classification workflow and provides a strong foundation for understanding more advanced classification algorithms such as Support Vector Machines, Decision Trees, Random Forests, and Neural Networks.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
