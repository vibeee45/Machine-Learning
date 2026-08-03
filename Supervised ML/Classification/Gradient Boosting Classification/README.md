# 🌳 Gradient Boosting Classification

A comprehensive Machine Learning project demonstrating **Gradient Boosting Classification** using the **Social_Network_Ads** dataset. This project covers the complete machine learning workflow, including data preprocessing, feature scaling, model training, prediction, performance evaluation, ROC analysis, and feature importance visualization.

The notebook is designed for beginners as well as intermediate learners who want to understand how **Gradient Boosting** builds a powerful predictive model by sequentially minimizing errors using gradient descent.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Classification?
- What is Gradient Boosting?
- Why Use Gradient Boosting?
- Ensemble Learning
- Boosting
- Weak Learners
- Gradient Descent
- Residual Errors
- Learning Rate
- Difference Between AdaBoost and Gradient Boosting
- Advantages of Gradient Boosting
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually defining rules, Machine Learning algorithms automatically identify relationships within the data and use those patterns to make predictions for new observations.

Machine Learning is widely used in:

- Healthcare
- Banking
- Finance
- Recommendation Systems
- Fraud Detection
- Customer Analytics
- Image Recognition
- Email Spam Detection
- Stock Market Prediction
- Marketing

Machine Learning algorithms are mainly classified into:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Gradient Boosting is a **Supervised Machine Learning Algorithm** because it learns from labeled training data to classify new observations.

---

# 🎯 What is Classification?

Classification is a supervised learning technique used to predict discrete categories or class labels.

The objective is to assign each observation to one of the predefined classes using input features.

Examples include:

- Email Spam Detection
- Disease Diagnosis
- Credit Card Fraud Detection
- Customer Purchase Prediction
- Loan Approval
- Sentiment Analysis

In this project, the model predicts whether a customer will purchase a product after viewing a social network advertisement.

Target Classes:

- **0 → Not Purchased**
- **1 → Purchased**

---

# 🌳 What is Gradient Boosting?

Gradient Boosting is a powerful **ensemble learning algorithm** that combines multiple weak learners to create a highly accurate predictive model.

Unlike Bagging algorithms, Gradient Boosting trains weak learners **sequentially**.

Each new Decision Tree is trained to correct the prediction errors (residuals) made by the previous trees.

Instead of assigning greater importance to misclassified observations like AdaBoost, Gradient Boosting minimizes a **loss function** using **Gradient Descent**.

By continuously reducing prediction errors, the model becomes increasingly accurate with each iteration.

Gradient Boosting is widely used for both **classification** and **regression** problems.

---

# ⭐ Why Use Gradient Boosting?

Gradient Boosting is one of the most accurate classical Machine Learning algorithms.

Some important advantages include:

- Excellent prediction accuracy.
- Learns from previous mistakes.
- Handles complex non-linear relationships.
- Supports both classification and regression.
- Reduces prediction bias.
- Provides feature importance scores.
- Performs exceptionally well on structured datasets.

---

# 🌐 Ensemble Learning

Ensemble Learning is a Machine Learning technique that combines predictions from multiple models to produce better performance than a single model.

Instead of relying on one classifier, several models work together to improve prediction accuracy.

The major ensemble techniques include:

- Bagging
- Boosting
- Stacking

Gradient Boosting belongs to the **Boosting** family.

---

# 🔄 Boosting

Boosting is an ensemble learning technique where weak learners are trained **one after another**.

Each learner attempts to reduce the prediction errors made by the previous learners.

The general process is:

1. Train the first Decision Tree.
2. Calculate prediction errors.
3. Train the next tree to reduce those errors.
4. Repeat the process until all trees are trained.
5. Combine all learners to produce the final prediction.

This sequential learning process significantly improves model performance.

---

# 🌱 Weak Learners

A **Weak Learner** is a simple predictive model that performs only slightly better than random guessing.

Gradient Boosting generally uses **shallow Decision Trees** as weak learners.

Although each tree individually has limited predictive capability, combining many weak learners results in a highly accurate model.

---

# 📉 Gradient Descent

Gradient Descent is an optimization algorithm used to minimize prediction errors.

Gradient Boosting applies Gradient Descent to minimize the chosen loss function.

Instead of directly predicting the target variable, each new Decision Tree learns how to reduce the remaining errors produced by previous trees.

This iterative optimization process is what gives Gradient Boosting its high predictive power.

---

# 📊 Residual Errors

Residual Errors represent the difference between the actual values and the predicted values.

### Formula

```text
Residual = Actual Value − Predicted Value
```

Gradient Boosting trains every new Decision Tree to predict these residual errors.

As more trees are added, the residuals become smaller, improving the overall accuracy of the model.

---

# ⚙️ Learning Rate

The **Learning Rate** controls how much each new Decision Tree contributes to the final prediction.

### Small Learning Rate

- Slower learning.
- Better generalization.
- Usually requires more trees.

### Large Learning Rate

- Faster learning.
- Higher risk of overfitting.
- Requires fewer trees.

Choosing an appropriate learning rate is essential for achieving optimal performance.

---

# ⚖️ Difference Between AdaBoost and Gradient Boosting

| AdaBoost | Gradient Boosting |
|----------|-------------------|
| Focuses on misclassified observations | Focuses on minimizing residual errors |
| Uses sample weights | Uses Gradient Descent |
| Sequential learning | Sequential learning |
| Simpler implementation | More flexible and powerful |
| Faster training | Generally slower but often more accurate |
| Mainly classification | Supports both classification and regression |

---

# ✅ Advantages of Gradient Boosting

- High prediction accuracy.
- Handles complex datasets effectively.
- Learns sequentially from previous errors.
- Supports classification and regression.
- Provides feature importance.
- Reduces prediction bias.
- Highly flexible through hyperparameter tuning.

---

# 📌 Assumptions of Gradient Boosting

Although Gradient Boosting is highly flexible, better performance is achieved when:

- Training data is representative.
- Features contain meaningful information.
- Noise and outliers are reasonably limited.
- Hyperparameters are properly tuned.
- Sufficient training data is available.

Feature Scaling is included in this project to maintain consistency with the other classification notebooks.

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
Select Features & Target
        │
        ▼
Encode Categorical Data
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Train Gradient Boosting Model
        │
        ▼
Predict Test Data
        │
        ▼
Evaluate Performance
        │
        ▼
Confusion Matrix
        │
        ▼
ROC Curve & AUC
        │
        ▼
Predict New Customer
        │
        ▼
Feature Importance
        │
        ▼
Display Hyperparameters
```
# 📂 Dataset Information

**Dataset Name:** Social_Network_Ads.csv

The dataset contains demographic information about customers and whether they purchased a product after viewing a social network advertisement.

The objective of this project is to build a **Gradient Boosting Classification** model that predicts whether a customer will purchase the advertised product based on their demographic details.

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
| Estimated Salary | Estimated annual salary |
| Purchased | Target Variable (0 = Not Purchased, 1 = Purchased) |

In this project, the **User ID** column is removed because it serves only as a unique identifier and has no predictive value.

The Gradient Boosting model is trained using the following input features:

- Gender
- Age
- Estimated Salary

The model predicts whether the customer purchased the advertised product.

---

# 🛠 Technologies Used

The following technologies and libraries are used throughout this project:

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
Gradient-Boosting-Classification/
│
├── Gradient Boosting Classification.ipynb
├── Social_Network_Ads.csv
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
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
from sklearn.ensemble import GradientBoostingClassifier
```

### Purpose

The required Python libraries are imported before beginning data preprocessing and model development.

| Library | Purpose |
|----------|----------|
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| train_test_split | Split the dataset into training and testing sets |
| LabelEncoder | Encode categorical variables |
| StandardScaler | Standardize numerical features |
| GradientBoostingClassifier | Build the Gradient Boosting Classification model |

---

## 🔹 Cell 2 – Load the Dataset

```python
df = pd.read_csv("Social_Network_Ads.csv")
```

The dataset is loaded into a Pandas DataFrame.

After loading, the first five rows are displayed using:

```python
df.head()
```

Displaying the first few records confirms that the dataset has been imported correctly and provides an overview of its structure.

---

## 🔹 Cell 3 – Dataset Information

```python
df.info()
```

This function displays useful information about the dataset, including:

- Number of rows
- Number of columns
- Data types
- Missing values
- Memory usage

Understanding the dataset structure is an essential preprocessing step before training the Machine Learning model.

---

## 🔹 Cell 4 – Statistical Summary

```python
df.describe()
```

This function generates descriptive statistics for the numerical columns.

The summary includes:

- Count
- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles

These statistics help understand the distribution of the data and identify possible outliers.

---

## 🔹 Cell 5 – Check Missing Values

```python
df.isnull().sum()
```

Before training the model, the dataset is checked for missing values.

This step verifies that each feature contains complete information.

The Social Network Ads dataset contains **no missing values**, making it suitable for training without additional preprocessing.

---

## 🔹 Cell 6 – Dataset Shape

```python
df.shape
```

The shape function returns the dimensions of the dataset.

For this dataset:

- Number of Rows: **400**
- Number of Columns: **5**

This provides a quick overview of the available training data.

---

## 🔹 Cell 7 – Feature and Target Selection

```python
if "User ID" in df.columns:
    df = df.drop("User ID", axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

The **User ID** column is removed because it serves only as a unique identifier and does not contribute to customer purchase prediction.

The remaining data is divided into:

**Features (X):**

- Gender
- Age
- Estimated Salary

**Target (y):**

- Purchased

Separating the independent variables from the target variable prepares the dataset for model training.

---

## 🔹 Cell 8 – Encode the Gender Column

```python
encoder = LabelEncoder()

X["Gender"] = encoder.fit_transform(X["Gender"])
```

Machine Learning algorithms require numerical input.

The **Gender** column is converted into numerical values using Label Encoding.

After encoding:

- Female → 0
- Male → 1

This transformation enables the Gradient Boosting model to process categorical information.

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

The dataset is divided into:

- **Training Set (75%)** – Used to train the Gradient Boosting model.
- **Testing Set (25%)** – Used to evaluate the model on unseen observations.

Using separate datasets helps measure how well the trained model generalizes to new data.

---

## 🔹 Cell 10 – Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Feature Scaling is performed using **StandardScaler**.

Although Gradient Boosting is based on Decision Trees and is generally less sensitive to feature scaling, standardizing numerical features provides a consistent preprocessing workflow across all classification projects in this repository.

After scaling:

- Mean = 0
- Standard Deviation = 1

The scaled data is then used for training and evaluating the Gradient Boosting classifier.
---

## 🔹 Cell 11 – Train the Gradient Boosting Classifier

```python
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)
```

After preprocessing and feature scaling, the Gradient Boosting model is trained using the training dataset.

In this project:

- **100 Decision Trees** are used as weak learners.
- **Learning Rate = 0.1**
- **Random State = 42** ensures reproducible results.

Gradient Boosting trains Decision Trees sequentially, where each new tree learns to reduce the residual errors produced by the previous trees.

By minimizing these residuals, the model gradually improves its prediction accuracy.

---

## 🔹 Cell 12 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained Gradient Boosting model predicts whether each customer in the testing dataset purchased the advertised product.

The predicted labels are stored in **`y_pred`**, which are later compared with the actual values to evaluate the model's performance.

---

## 🔹 Cell 13 – Compare Actual and Predicted Values

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.head()
```

A comparison table is created containing:

- Actual Purchase Status
- Predicted Purchase Status

This comparison provides a quick overview of how accurately the model classifies unseen observations.

---

## 🔹 Cell 14 – Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The Confusion Matrix summarizes the prediction performance of the Gradient Boosting classifier.

It consists of:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

A heatmap visualization provides an intuitive representation of these classification results.

---

## 🔹 Cell 15 – Accuracy Score

```python
accuracy = accuracy_score(y_test, y_pred)
```

Accuracy measures the percentage of correctly classified observations.

### Formula

```text
Accuracy = (TP + TN) / Total Predictions
```

Higher accuracy indicates better overall classification performance.

---

## 🔹 Cell 16 – Classification Report

```python
print(classification_report(y_test, y_pred))
```

The Classification Report provides detailed evaluation metrics including:

- Precision
- Recall
- F1-Score
- Support

These metrics provide deeper insight into the model's performance than accuracy alone.

---

## 🔹 Cell 17 – ROC Curve and AUC Score

```python
roc_curve()
roc_auc_score()
```

The ROC (Receiver Operating Characteristic) Curve illustrates how effectively the Gradient Boosting model distinguishes between the two target classes.

The Area Under the Curve (AUC) summarizes the classifier's ability to separate positive and negative classes.

Interpretation:

- AUC = 1.0 → Perfect Classifier
- AUC = 0.5 → Random Guessing

A higher AUC value indicates better classification performance.

---

## 🔹 Cell 18 – Predict a New Customer

```python
new_customer = pd.DataFrame({
    "Gender":[1],
    "Age":[35],
    "EstimatedSalary":[60000]
})

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)
```

A new customer's information is first standardized using the previously fitted **StandardScaler**.

The scaled data is then passed to the trained Gradient Boosting model to predict whether the customer is likely to purchase the advertised product.

This demonstrates how the trained model can be used in real-world prediction scenarios.

---

## 🔹 Cell 19 – Feature Importance

```python
importance = model.feature_importances_
```

Gradient Boosting provides **Feature Importance Scores**, which indicate how much each feature contributes to the model's predictions.

This notebook displays:

- A table of feature importance values.
- A bar chart comparing the importance of each feature.

Features with higher importance scores have a greater influence on the model's decision-making process.

---

## 🔹 Cell 20 – Gradient Boosting Hyperparameters

```python
model.get_params()
```

The final notebook cell displays the hyperparameters used to train the Gradient Boosting classifier.

Some important hyperparameters include:

- **n_estimators** – Number of Decision Trees used during training.
- **learning_rate** – Controls how much each tree contributes to the final model.
- **loss** – Specifies the loss function optimized during boosting.
- **criterion** – Function used to measure the quality of a split.
- **random_state** – Ensures reproducible model training.

Understanding these hyperparameters is essential for optimizing Gradient Boosting performance.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Gradient Boosting classifier.

These include:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score

Together, these metrics provide a comprehensive evaluation of the classification model.

---

## 📈 Accuracy

Accuracy measures the proportion of correctly classified observations.

### Formula

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Interpretation

- Higher Accuracy indicates better prediction performance.
- Most useful when the dataset is balanced.

---

## 📈 Precision

Precision measures how many predicted positive observations are actually positive.

### Formula

```text
Precision = TP / (TP + FP)
```

### Interpretation

Higher Precision indicates fewer False Positives.

---

## 📈 Recall

Recall measures how many actual positive observations are correctly identified.

### Formula

```text
Recall = TP / (TP + FN)
```

### Interpretation

Higher Recall indicates fewer False Negatives.

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

A curve closer to the upper-left corner represents a stronger classifier.

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

Gradient Boosting is widely used in many real-world applications, including:

- Fraud Detection
- Customer Churn Prediction
- Credit Risk Assessment
- Medical Diagnosis
- Customer Purchase Prediction
- Recommendation Systems
- Marketing Analytics
- Financial Forecasting
- Insurance Risk Analysis
- Predictive Analytics

---

# ✅ Advantages of Gradient Boosting

- Very high prediction accuracy.
- Handles complex non-linear relationships.
- Learns sequentially from previous errors.
- Supports both classification and regression.
- Provides feature importance scores.
- Highly flexible through hyperparameter tuning.
- Performs exceptionally well on structured datasets.

---

# ❌ Limitations of Gradient Boosting

- Training can be slower than simpler algorithms.
- Sensitive to hyperparameter selection.
- May overfit if too many trees are used.
- Computationally more expensive than Decision Trees.
- Requires careful tuning for optimal performance.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Gradient-Boosting-Classification.git
```

Navigate to the project directory:

```bash
cd Gradient-Boosting-Classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Gradient Boosting Classification.ipynb
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

Install all required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand Ensemble Learning and Gradient Boosting.
- Learn how Gradient Boosting minimizes residual errors using Gradient Descent.
- Understand the concepts of Weak Learners and Learning Rate.
- Build a Gradient Boosting classifier using Scikit-learn.
- Evaluate classification models using Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC.
- Interpret Feature Importance scores.
- Apply Gradient Boosting to real-world classification problems.

---

# 📝 Conclusion

Gradient Boosting is one of the most powerful ensemble learning algorithms for supervised Machine Learning. By sequentially training Decision Trees to minimize residual errors using Gradient Descent, it achieves excellent predictive performance on a wide variety of classification problems.

In this project, the Social Network Ads dataset was explored, preprocessed, feature-scaled, used to train a Gradient Boosting classifier, evaluated using multiple classification metrics, and analyzed through feature importance and model hyperparameters.

This project provides a solid foundation for understanding advanced boosting algorithms and prepares learners for modern gradient boosting frameworks such as **XGBoost**, **LightGBM**, and **CatBoost**.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
