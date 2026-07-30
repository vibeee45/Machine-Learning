# 🛡️ Support Vector Machine (SVM) Classification

A comprehensive Machine Learning project demonstrating **Support Vector Machine (SVM) Classification** using the **Social_Network_Ads** dataset. This project covers the complete machine learning workflow, including data preprocessing, feature scaling, model training, prediction, performance evaluation, ROC analysis, and decision boundary visualization.

The notebook is designed for beginners as well as intermediate learners who want to understand how Support Vector Machines classify data by finding the optimal hyperplane that maximizes the margin between different classes.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Classification?
- What is Support Vector Machine (SVM)?
- Why Use SVM?
- Hyperplane
- Support Vectors
- Margin
- Hard Margin vs Soft Margin
- Kernel Trick
- Types of Kernels
- Difference Between Logistic Regression and SVM
- Advantages of SVM
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

Support Vector Machine (SVM) is a **Supervised Machine Learning Algorithm** because it learns from labeled training data to classify new observations.

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

# 🛡️ What is Support Vector Machine (SVM)?

Support Vector Machine (SVM) is a powerful supervised Machine Learning algorithm used for both classification and regression problems. However, it is primarily used for classification tasks.

The main objective of SVM is to find the **optimal separating boundary (hyperplane)** that divides different classes while maximizing the distance between the closest data points of each class.

Unlike Decision Trees or Random Forest, SVM focuses on finding the best possible decision boundary rather than creating multiple rules or trees.

SVM performs exceptionally well on:

- High-dimensional datasets
- Small and medium-sized datasets
- Binary classification problems
- Linearly and non-linearly separable data

---

# ⭐ Why Use SVM?

Support Vector Machine is one of the most powerful classification algorithms because it focuses on maximizing the margin between different classes.

Some important advantages include:

- High classification accuracy.
- Effective in high-dimensional spaces.
- Works well when the number of features is large.
- Can solve both linear and non-linear problems.
- Less prone to overfitting due to margin maximization.
- Supports multiple kernel functions.
- Performs well even with limited training data.

---

# 📏 Hyperplane

A **Hyperplane** is the decision boundary that separates different classes.

For a dataset with:

- 2 Features → Hyperplane is a line.
- 3 Features → Hyperplane is a plane.
- More than 3 Features → Hyperplane exists in higher-dimensional space.

The goal of SVM is to identify the hyperplane that best separates the classes while maximizing the margin.

---

# 📍 Support Vectors

Support Vectors are the data points that lie closest to the separating hyperplane.

These points are the most important observations because they determine the position and orientation of the decision boundary.

If the support vectors change, the hyperplane also changes.

The performance of SVM depends primarily on these support vectors rather than the entire dataset.

---

# 📐 Margin

The **Margin** is the distance between the separating hyperplane and the nearest data points from each class.

SVM always attempts to maximize this margin.

A larger margin generally leads to:

- Better generalization
- Improved prediction accuracy
- Lower risk of overfitting

---

# ⚖️ Hard Margin vs Soft Margin

| Hard Margin | Soft Margin |
|--------------|-------------|
| Assumes perfectly separable data | Allows some misclassification |
| No classification errors allowed | Small classification errors are acceptable |
| Sensitive to outliers | More robust to noisy data |
| Rarely used in real-world datasets | Commonly used in practical applications |

Most real-world datasets use **Soft Margin SVM** because data is rarely perfectly separable.

---

# 🎯 Kernel Trick

Many real-world datasets cannot be separated using a straight line.

The **Kernel Trick** transforms the original data into a higher-dimensional feature space where it becomes easier to separate using a linear hyperplane.

Instead of explicitly transforming the data, SVM computes similarities between observations using kernel functions, making the process computationally efficient.

---

# 🌐 Types of Kernels

Support Vector Machines support several kernel functions depending on the nature of the data.

## 1. Linear Kernel

Used when the dataset is linearly separable.

Suitable for:

- Text Classification
- High-dimensional datasets

---

## 2. Polynomial Kernel

Creates curved decision boundaries.

Suitable for:

- Complex relationships
- Pattern Recognition

---

## 3. Radial Basis Function (RBF) Kernel

The most commonly used kernel.

Suitable for:

- Non-linear datasets
- Complex decision boundaries

---

## 4. Sigmoid Kernel

Behaves similarly to a Neural Network activation function.

Suitable for certain specialized classification tasks.

---

# ⚖️ Difference Between Logistic Regression and SVM

| Logistic Regression | Support Vector Machine |
|----------------------|------------------------|
| Predicts probabilities | Finds the optimal hyperplane |
| Uses the Sigmoid function | Maximizes the margin |
| Suitable for linearly separable data | Handles both linear and non-linear data |
| Faster on very large datasets | Better for high-dimensional datasets |
| Outputs probability estimates | Outputs class labels (probabilities optional) |
| Less effective with complex boundaries | Excellent at handling complex boundaries using kernels |

---

# ✅ Advantages of SVM

- High classification accuracy.
- Effective in high-dimensional feature spaces.
- Works well with small datasets.
- Supports multiple kernel functions.
- Robust against overfitting.
- Memory efficient because it uses only support vectors.
- Effective for binary classification problems.
- Strong performance on complex datasets.

---

# 📌 Assumptions of SVM

Although SVM is highly flexible, better performance is achieved when:

- Data is properly scaled.
- Features are relevant.
- Classes are reasonably separable.
- Noise is limited.
- Appropriate kernel and hyperparameters are selected.

Unlike Decision Trees and Random Forest, **Feature Scaling is essential for SVM** because it relies on distance calculations.

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
Train SVM Model
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
Decision Boundary Visualization
        │
        ▼
Display Hyperparameters
```
# 📂 Dataset Information

**Dataset Name:** Social_Network_Ads.csv

The dataset contains demographic information about customers and whether they purchased a product after viewing a social network advertisement.

The objective of this project is to build a **Support Vector Machine (SVM) Classification** model that predicts whether a customer will purchase the advertised product based on their demographic details.

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

In this project, the **User ID** column is removed because it is only an identifier and does not contribute to the prediction process.

The Support Vector Machine model is trained using the following input features:

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
Support-Vector-Machine-Classification/
│
├── Support Vector Machine.ipynb
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
from sklearn.svm import SVC
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
| SVC | Build the Support Vector Machine Classification model |

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

This information helps understand the structure of the dataset before preprocessing.

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

These statistics help understand data distribution and identify potential outliers.

---

## 🔹 Cell 5 – Check Missing Values

```python
df.isnull().sum()
```

Before training the model, the dataset is checked for missing values.

This ensures that every feature contains complete information.

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

This provides a quick overview of the dataset size.

---

## 🔹 Cell 7 – Feature and Target Selection

```python
if "User ID" in df.columns:
    df = df.drop("User ID", axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

The **User ID** column is removed because it serves only as a unique identifier and does not contribute to predicting customer purchases.

The remaining dataset is divided into:

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

This transformation enables the SVM model to process categorical information.

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

- **Training Set (75%)** – Used to train the SVM model.
- **Testing Set (25%)** – Used to evaluate the model on unseen data.

Separating the dataset helps measure how well the trained model generalizes to new observations.

---

## 🔹 Cell 10 – Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Feature Scaling is a critical preprocessing step for Support Vector Machine.

Unlike Decision Trees and Random Forest, SVM calculates distances between observations while finding the optimal hyperplane.

Without scaling:

- Features with larger numerical values dominate the model.
- The decision boundary may become biased.

StandardScaler transforms each numerical feature so that it has:

- Mean = 0
- Standard Deviation = 1

This improves model performance, convergence, and classification accuracy.
---

## 🔹 Cell 11 – Train the Support Vector Machine (SVM) Classifier

```python
model = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)
```

After preprocessing and scaling the dataset, the Support Vector Machine model is trained using the training data.

In this project:

- **Linear Kernel** is used.
- **Probability estimates** are enabled for ROC Curve analysis.
- **random_state = 42** ensures reproducible results.

During training, the SVM algorithm identifies the **optimal hyperplane** that maximizes the margin between the two classes.

---

## 🔹 Cell 12 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained SVM model predicts whether each customer in the testing dataset purchased the advertised product.

The predicted labels are stored in **`y_pred`**, which are later compared with the actual values to evaluate model performance.

---

## 🔹 Cell 13 – Compare Actual and Predicted Values

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.head()
```

A comparison table is created to display:

- Actual Purchase Status
- Predicted Purchase Status

This makes it easy to observe how accurately the SVM classifier predicts customer purchases.

---

## 🔹 Cell 14 – Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The Confusion Matrix summarizes the prediction performance of the SVM classifier.

It consists of:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

A heatmap visualization is used to make these results easier to interpret.

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

The ROC (Receiver Operating Characteristic) Curve illustrates how effectively the SVM model distinguishes between the two target classes.

The Area Under the Curve (AUC) summarizes the model's classification capability.

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

The scaled data is then passed to the trained SVM model to predict whether the customer is likely to purchase the advertised product.

This demonstrates how the trained model can be used in real-world prediction scenarios.

---

## 🔹 Cell 19 – Decision Boundary Visualization

```python
# Decision Boundary Visualization
```

One of the most useful visualizations for Support Vector Machines is the **Decision Boundary**.

This plot illustrates:

- The separating hyperplane learned by the model.
- The regions classified as each class.
- The distribution of training observations.

Decision Boundary visualization helps understand how the SVM classifier separates different classes after feature scaling.

---

## 🔹 Cell 20 – SVM Hyperparameters

```python
model.get_params()
```

The final notebook cell displays the hyperparameters used to train the Support Vector Machine model.

Some important hyperparameters include:

- **kernel** – Specifies the kernel function (Linear in this project).
- **C** – Regularization parameter controlling the trade-off between maximizing the margin and minimizing classification errors.
- **gamma** – Controls the influence of individual training samples (used mainly with non-linear kernels).
- **probability** – Enables probability estimation.
- **random_state** – Ensures reproducible model training.

Understanding these hyperparameters is essential for optimizing SVM performance.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Support Vector Machine classifier.

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

Support Vector Machine is widely used in many real-world domains, including:

- Email Spam Detection
- Face Recognition
- Image Classification
- Handwritten Digit Recognition
- Medical Diagnosis
- Credit Risk Analysis
- Customer Purchase Prediction
- Text Classification
- Sentiment Analysis
- Bioinformatics

---

# ✅ Advantages of SVM

- High classification accuracy.
- Effective in high-dimensional feature spaces.
- Works well with small and medium-sized datasets.
- Supports linear and non-linear classification.
- Robust against overfitting through margin maximization.
- Uses only support vectors, making it memory efficient.
- Flexible through multiple kernel functions.

---

# ❌ Limitations of SVM

- Requires feature scaling.
- Training can be slow for very large datasets.
- Choosing the correct kernel can be challenging.
- Hyperparameter tuning is important for optimal performance.
- Decision boundary is less interpretable than Decision Trees.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Support-Vector-Machine-Classification.git
```

Navigate to the project directory:

```bash
cd Support-Vector-Machine-Classification
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
Support Vector Machine.ipynb
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

- Understand the working principle of Support Vector Machines.
- Understand Hyperplanes, Support Vectors, and Margins.
- Learn the importance of Feature Scaling.
- Build an SVM Classification model using Scikit-learn.
- Evaluate classification models using Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC.
- Visualize the SVM Decision Boundary.
- Understand important SVM hyperparameters.
- Apply SVM Classification to real-world prediction problems.

---

# 📝 Conclusion

Support Vector Machine (SVM) is one of the most powerful supervised Machine Learning algorithms for classification tasks. By identifying the optimal hyperplane and maximizing the margin between classes, SVM achieves strong predictive performance and excellent generalization.

In this project, the Social Network Ads dataset was explored, preprocessed, feature-scaled, used to train an SVM classifier, evaluated using multiple classification metrics, visualized through a decision boundary, and analyzed through its hyperparameters.

This project provides a solid foundation for learning advanced kernel-based methods and tackling complex classification problems.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
