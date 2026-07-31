# 🤝 K-Nearest Neighbors (KNN) Classification

A comprehensive Machine Learning project demonstrating **K-Nearest Neighbors (KNN) Classification** using the **Social_Network_Ads** dataset. This project covers the complete machine learning workflow, including data preprocessing, feature scaling, model training, prediction, performance evaluation, ROC analysis, and determining the optimal value of **K**.

The notebook is designed for beginners as well as intermediate learners who want to understand how KNN classifies data by measuring the distance between data points and assigning the class based on the majority of the nearest neighbors.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Classification?
- What is K-Nearest Neighbors (KNN)?
- Why Use KNN?
- Instance-Based Learning
- Lazy Learning
- Distance Metrics
- Euclidean Distance
- Manhattan Distance
- Minkowski Distance
- Choosing the Value of K
- Difference Between Logistic Regression and KNN
- Advantages of KNN
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually writing rules for every possible scenario, Machine Learning algorithms automatically identify relationships within the data and use those patterns to make predictions for new observations.

Machine Learning is widely used in:

- Healthcare
- Banking
- Finance
- Fraud Detection
- Recommendation Systems
- Marketing
- Customer Analytics
- Image Recognition
- Email Spam Detection
- Autonomous Vehicles

Machine Learning algorithms are mainly classified into:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

K-Nearest Neighbors (KNN) belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 🎯 What is Classification?

Classification is a supervised learning technique used to predict discrete categories or class labels.

The objective is to assign each observation to one of the predefined classes based on the available input features.

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

# 🤝 What is K-Nearest Neighbors (KNN)?

K-Nearest Neighbors (KNN) is one of the simplest and most widely used supervised Machine Learning algorithms for classification and regression tasks.

Unlike many Machine Learning algorithms, KNN does not build a mathematical model during training.

Instead, it stores the entire training dataset and predicts the class of a new observation by identifying the **K nearest data points** based on a chosen distance metric.

The predicted class is determined using **Majority Voting**, where the class that appears most frequently among the nearest neighbors becomes the final prediction.

---

# ⭐ Why Use KNN?

K-Nearest Neighbors is popular because of its simplicity and effectiveness.

Some important advantages include:

- Easy to understand and implement.
- No assumptions about data distribution.
- Works well for multi-class classification.
- Naturally handles non-linear decision boundaries.
- No explicit training phase.
- Effective on small and medium-sized datasets.
- Can be used for both classification and regression.

---

# 📌 Instance-Based Learning

KNN is known as an **Instance-Based Learning** algorithm because it stores all training examples instead of learning mathematical parameters.

Whenever a new observation is provided, the algorithm compares it with all stored training instances and finds the nearest neighbors.

Prediction depends entirely on the stored examples.

---

# 💤 Lazy Learning

KNN is also called a **Lazy Learning Algorithm**.

Unlike Logistic Regression or Decision Trees, KNN does not learn a predictive model during the training phase.

Instead:

- Training simply stores the dataset.
- Most computation happens during prediction.
- Classification occurs only when new data is provided.

Because of this behavior, KNN has:

- Very fast training.
- Relatively slower prediction.

---

# 📏 Distance Metrics

The performance of KNN depends on measuring the distance between observations.

Several distance metrics can be used depending on the problem.

The most common distance metrics are:

- Euclidean Distance
- Manhattan Distance
- Minkowski Distance

---

# 📐 Euclidean Distance

Euclidean Distance is the most commonly used distance metric in KNN.

It measures the straight-line distance between two observations.

### Formula

```text
d = √[(x₂ − x₁)² + (y₂ − y₁)²]
```

Euclidean Distance is commonly used when:

- Features are continuous.
- Feature Scaling has been applied.
- Data is represented in geometric space.

---

# 📐 Manhattan Distance

Manhattan Distance measures the distance by moving only horizontally and vertically, similar to navigating city streets.

### Formula

```text
d = |x₂ − x₁| + |y₂ − y₁|
```

Manhattan Distance is often preferred when:

- Data contains many dimensions.
- Outliers should have less influence.
- Movement is restricted to grid-like paths.

---

# 📐 Minkowski Distance

Minkowski Distance is a generalized distance metric.

It combines Euclidean and Manhattan distances into a single formula.

### Formula

```text
d = ( Σ |xᵢ − yᵢ|ᵖ )^(1/p)
```

Where:

- **p = 1** → Manhattan Distance
- **p = 2** → Euclidean Distance

In this project:

- **Metric = Minkowski**
- **p = 2**

which makes the algorithm use **Euclidean Distance**.

---

# 🎯 Choosing the Value of K

The value of **K** determines how many nearest neighbors are considered before making a prediction.

### Small K

- More flexible decision boundary.
- Sensitive to noise.
- Higher risk of overfitting.

### Large K

- Smoother decision boundary.
- Less sensitive to noise.
- May underfit the data.

Selecting an appropriate value of **K** is essential for achieving good model performance.

In this project, different values of **K** are evaluated and compared using an **Accuracy vs K** graph.

---

# ⚖️ Difference Between Logistic Regression and KNN

| Logistic Regression | K-Nearest Neighbors |
|----------------------|--------------------|
| Parametric Algorithm | Non-Parametric Algorithm |
| Learns a mathematical model | Stores the training data |
| Fast prediction | Slower prediction |
| Assumes linear decision boundary | Captures non-linear boundaries |
| Requires model training | No explicit training phase |
| Uses Sigmoid Function | Uses Distance Metrics |

---

# ✅ Advantages of KNN

- Simple and easy to understand.
- No training phase.
- Naturally supports multi-class classification.
- Handles non-linear decision boundaries.
- Flexible and versatile.
- Effective for small datasets.
- Easy to update with new observations.
- No assumptions about data distribution.

---

# 📌 Assumptions of KNN

Although KNN is simple, good performance depends on:

- Proper Feature Scaling.
- Appropriate value of **K**.
- Relevant input features.
- Balanced dataset.
- Meaningful distance calculations.

Since KNN is a **distance-based algorithm**, **Feature Scaling is mandatory** before training the model.

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
Train KNN Model
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
Find Best Value of K
        │
        ▼
Display Hyperparameters
```
# 📂 Dataset Information

**Dataset Name:** Social_Network_Ads.csv

The dataset contains demographic information about customers and whether they purchased a product after viewing a social network advertisement.

The objective of this project is to build a **K-Nearest Neighbors (KNN) Classification** model that predicts whether a customer will purchase the advertised product based on their demographic details.

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

The K-Nearest Neighbors model is trained using the following input features:

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
K-Nearest-Neighbors-Classification/
│
├── K-Nearest Neighbors.ipynb
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
from sklearn.neighbors import KNeighborsClassifier
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
| KNeighborsClassifier | Build the K-Nearest Neighbors Classification model |

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

Understanding the dataset structure is an essential step before preprocessing and model training.

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

These statistics help understand data distribution and identify possible outliers.

---

## 🔹 Cell 5 – Check Missing Values

```python
df.isnull().sum()
```

Before training the model, the dataset is checked for missing values.

This ensures that each feature contains complete information.

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

This transformation allows the KNN algorithm to process categorical information.

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

- **Training Set (75%)** – Used to train the KNN model.
- **Testing Set (25%)** – Used to evaluate the model on unseen observations.

Using separate datasets helps evaluate how well the trained model generalizes to new data.

---

## 🔹 Cell 10 – Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Feature Scaling is an essential preprocessing step for K-Nearest Neighbors.

Since KNN calculates the distance between observations, features with larger numerical values can dominate the distance calculations if scaling is not applied.

StandardScaler transforms each numerical feature so that it has:

- Mean = 0
- Standard Deviation = 1

This ensures that every feature contributes equally to the distance calculations, resulting in improved classification performance.
---

## 🔹 Cell 11 – Train the K-Nearest Neighbors (KNN) Classifier

```python
model = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)

model.fit(X_train, y_train)
```

After preprocessing and feature scaling, the K-Nearest Neighbors model is trained using the training dataset.

In this project:

- **Number of Neighbors (K) = 5**
- **Minkowski Distance** is used as the distance metric.
- **p = 2**, which means the algorithm uses **Euclidean Distance**.

Unlike many Machine Learning algorithms, KNN does not create a mathematical model. Instead, it stores the training data and uses it during prediction.

---

## 🔹 Cell 12 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained KNN model predicts whether each customer in the testing dataset purchased the advertised product.

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

A comparison table is created containing:

- Actual Purchase Status
- Predicted Purchase Status

This helps compare the model's predictions with the true class labels.

---

## 🔹 Cell 14 – Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The Confusion Matrix summarizes the prediction performance of the KNN classifier.

It consists of:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

A heatmap visualization provides an intuitive representation of these results.

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

These metrics provide a more comprehensive evaluation than accuracy alone.

---

## 🔹 Cell 17 – ROC Curve and AUC Score

```python
roc_curve()
roc_auc_score()
```

The ROC (Receiver Operating Characteristic) Curve illustrates how effectively the KNN model distinguishes between the two target classes.

The Area Under the Curve (AUC) summarizes the classifier's overall performance.

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

The scaled data is then passed to the trained KNN model to predict whether the customer is likely to purchase the advertised product.

This demonstrates how the trained model can be used in real-world prediction scenarios.

---

## 🔹 Cell 19 – Choosing the Best Value of K

```python
accuracy_scores = []

for k in range(1, 21):
    ...
```

Selecting an appropriate value of **K** is one of the most important aspects of building a KNN model.

In this notebook:

- K values from **1 to 20** are evaluated.
- The model is trained for each value of K.
- Accuracy is calculated for every model.
- An **Accuracy vs Number of Neighbors (K)** graph is plotted.

The value of **K** that produces the highest accuracy is selected as the most suitable choice for this dataset.

This visualization helps understand how the number of neighbors influences model performance.

---

## 🔹 Cell 20 – KNN Hyperparameters

```python
model.get_params()
```

The final notebook cell displays the hyperparameters used to train the KNN model.

Some important hyperparameters include:

- **n_neighbors** – Number of nearest neighbors considered during prediction.
- **metric** – Distance metric used to calculate similarity between observations.
- **p** – Determines the type of Minkowski distance (1 = Manhattan, 2 = Euclidean).
- **weights** – Specifies whether all neighbors contribute equally or closer neighbors receive greater importance.
- **algorithm** – Algorithm used to compute nearest neighbors efficiently.

Understanding these hyperparameters is essential for optimizing KNN performance.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the K-Nearest Neighbors classifier.

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

K-Nearest Neighbors is widely used in many real-world applications, including:

- Recommendation Systems
- Medical Diagnosis
- Image Recognition
- Handwritten Digit Recognition
- Customer Purchase Prediction
- Credit Risk Analysis
- Fraud Detection
- Pattern Recognition
- Document Classification
- Customer Segmentation

---

# ✅ Advantages of KNN

- Simple and easy to understand.
- No explicit training phase.
- Naturally handles multi-class classification.
- Works well with non-linear data.
- Flexible and easy to implement.
- No assumptions about data distribution.
- Easy to update with new observations.

---

# ❌ Limitations of KNN

- Prediction can be slow for large datasets.
- Requires feature scaling.
- Sensitive to irrelevant features.
- Choosing the correct value of K can be challenging.
- Memory-intensive because the entire training dataset must be stored.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/K-Nearest-Neighbors-Classification.git
```

Navigate to the project directory:

```bash
cd K-Nearest-Neighbors-Classification
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
K-Nearest Neighbors.ipynb
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

- Understand the working principle of K-Nearest Neighbors.
- Learn the concepts of Lazy Learning and Instance-Based Learning.
- Understand Euclidean, Manhattan, and Minkowski Distance.
- Apply Feature Scaling correctly.
- Build a KNN Classification model using Scikit-learn.
- Evaluate classification models using Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC.
- Determine the optimal value of K using an Accuracy vs K graph.
- Apply KNN Classification to real-world prediction problems.

---

# 📝 Conclusion

K-Nearest Neighbors (KNN) is one of the simplest yet highly effective supervised Machine Learning algorithms. By classifying observations based on the majority vote of the nearest neighbors, it provides an intuitive approach to solving classification problems.

In this project, the Social Network Ads dataset was explored, preprocessed, feature-scaled, used to train a KNN classifier, evaluated using multiple classification metrics, and analyzed by identifying the optimal value of K through an accuracy comparison graph.

This project provides a strong foundation for understanding distance-based Machine Learning algorithms and prepares learners for more advanced classification techniques.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
