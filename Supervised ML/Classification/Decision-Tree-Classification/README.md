# 🌳 Decision Tree Classification

A comprehensive Machine Learning project demonstrating **Decision Tree Classification** using the **Social Network Ads** dataset. This project covers the complete machine learning workflow, including data preprocessing, model training, prediction, evaluation, visualization of the decision tree, and feature importance analysis.

The notebook is designed for beginners as well as intermediate learners who want to understand how Decision Tree algorithms work and how they can be applied to real-world classification problems.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Classification?
- What is Decision Tree Classification?
- Why Use Decision Trees?
- Entropy
- Gini Index
- Information Gain
- How Does a Decision Tree Work?
- Difference Between Logistic Regression and Decision Tree
- Advantages of Decision Trees
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data without being explicitly programmed.

Instead of writing specific rules for every problem, a Machine Learning algorithm learns from historical data and makes predictions or decisions on new, unseen data.

Machine Learning is widely used in:

- Healthcare
- Banking
- Finance
- Marketing
- Recommendation Systems
- Fraud Detection
- Autonomous Vehicles
- Image Recognition
- Natural Language Processing

Machine Learning algorithms are generally divided into three categories:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Decision Tree Classification belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 🎯 What is Classification?

Classification is a supervised learning technique used to predict discrete categories or class labels.

The goal of a classification model is to assign each observation to one of the predefined classes based on the input features.

Examples include:

- Email Spam Detection
- Disease Prediction
- Customer Purchase Prediction
- Credit Card Fraud Detection
- Loan Approval
- Sentiment Analysis

In this project, the model predicts whether a customer will purchase a product after seeing a social media advertisement.

Target Classes:

- **0 → Not Purchased**
- **1 → Purchased**

---

# 🌳 What is Decision Tree Classification?

Decision Tree Classification is a supervised Machine Learning algorithm that classifies data by learning a series of decision rules from the training dataset.

The algorithm builds a tree-like structure where:

- The top node is called the **Root Node**.
- Internal nodes represent decision conditions.
- Branches represent the outcomes of those decisions.
- The final nodes are called **Leaf Nodes**, which represent the predicted class.

During training, the algorithm repeatedly splits the dataset into smaller subsets based on the feature that provides the best separation between classes.

These splits continue until a stopping condition is reached, such as a maximum tree depth or a minimum number of samples in a node.

---

# ⭐ Why Use Decision Trees?

Decision Trees are one of the most popular classification algorithms because they are simple, intuitive, and easy to interpret.

Some key reasons for using Decision Trees include:

- Easy to understand and visualize.
- Requires very little data preprocessing.
- Handles both numerical and categorical data.
- Captures non-linear relationships.
- Performs automatic feature selection.
- Can model complex decision boundaries.
- Works well for both classification and regression tasks.

---

# 📊 Entropy

Entropy is a measure of impurity or randomness in a dataset.

It indicates how mixed the class labels are within a node.

- Low Entropy → Pure node
- High Entropy → Mixed node

### Formula

```text
Entropy(S) = - Σ pᵢ log₂(pᵢ)
```

Where:

- **pᵢ** is the probability of each class.

Interpretation:

- Entropy = 0 → Perfectly pure node.
- Higher entropy indicates greater uncertainty.

The Decision Tree algorithm aims to reduce entropy after every split.

---

# 📈 Gini Index

The Gini Index is another measure of impurity used by Decision Tree algorithms.

It measures how often a randomly selected element would be incorrectly classified if it were randomly labeled according to the class distribution.

### Formula

```text
Gini = 1 − Σ(pᵢ²)
```

Where:

- **pᵢ** is the probability of each class.

Interpretation:

- Gini = 0 → Perfectly pure node.
- Lower Gini values indicate better splits.

Many Decision Tree implementations, including Scikit-learn, use the Gini Index as the default splitting criterion.

---

# 📉 Information Gain

Information Gain measures the reduction in entropy achieved after splitting the dataset.

The feature with the highest Information Gain is selected for splitting because it provides the maximum reduction in uncertainty.

### Formula

```text
Information Gain = Entropy(Parent) − Weighted Entropy(Children)
```

Higher Information Gain indicates a better feature for creating a split.

The Decision Tree continues selecting the feature with the highest Information Gain until the stopping criteria are met.

---

# ⚙️ How Does a Decision Tree Work?

The Decision Tree algorithm follows these steps:

1. Start with the entire dataset as the root node.
2. Calculate the impurity (Entropy or Gini Index).
3. Evaluate every feature to determine the best split.
4. Select the feature with the highest Information Gain or lowest Gini impurity.
5. Split the dataset into child nodes.
6. Repeat the process recursively for each child node.
7. Stop when the node becomes pure or another stopping condition is satisfied.
8. Assign a class label to each leaf node.

The final tree represents a sequence of decision rules used for classification.

---

# ⚖️ Difference Between Logistic Regression and Decision Tree

| Logistic Regression | Decision Tree |
|---------------------|---------------|
| Linear Classification Algorithm | Tree-Based Classification Algorithm |
| Creates a linear decision boundary | Creates non-linear decision boundaries |
| Requires feature scaling for best performance | Does not require feature scaling |
| Uses the Sigmoid Function | Uses recursive data splitting |
| Predicts probabilities | Predicts decision rules |
| Assumes a linear relationship | Can capture complex non-linear relationships |
| Less prone to overfitting | Can overfit if not properly controlled |

---

# 📌 Assumptions of Decision Tree

Although Decision Trees make fewer assumptions than many Machine Learning algorithms, they still have some important considerations:

- Training data should represent the real-world problem.
- Features should contain meaningful information.
- Excessive noise may reduce performance.
- Very deep trees may lead to overfitting.
- Proper pruning or parameter tuning improves generalization.

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
Train Decision Tree Model
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
Visualize Decision Tree
        │
        ▼
Feature Importance
```
# 📂 Dataset Information

**Dataset Name:** Social_Network_Ads.csv

The dataset contains demographic information about customers and whether they purchased a product after viewing a social network advertisement.

The objective of this project is to build a Decision Tree Classification model that predicts whether a customer will purchase the advertised product based on their demographic details.

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
| Purchased | Target variable (0 = Not Purchased, 1 = Purchased) |

In this project, the **User ID** column is removed because it is only an identifier and does not contribute to predicting customer purchases.

The model uses **Gender**, **Age**, and **Estimated Salary** as input features to predict whether a customer purchased the product.

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
Decision-Tree-Classification/
│
├── Decision Tree Classification.ipynb
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
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
```

### Purpose

The required Python libraries are imported before beginning data analysis and model development.

| Library | Purpose |
|----------|----------|
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| train_test_split | Split data into training and testing datasets |
| LabelEncoder | Convert categorical values into numerical values |
| DecisionTreeClassifier | Build the Decision Tree Classification model |

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

Displaying the first few records helps verify that the dataset has been imported correctly and provides an overview of its structure.

---

## 🔹 Cell 3 – Dataset Information

```python
df.info()
```

This function provides useful information about the dataset, including:

- Number of rows
- Number of columns
- Data types
- Missing values
- Memory usage

Understanding the dataset structure is an important preprocessing step before training a Machine Learning model.

---

## 🔹 Cell 4 – Statistical Summary

```python
df.describe()
```

This function generates descriptive statistics for all numerical features.

The output includes:

- Count
- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles

These statistics help understand the overall distribution of the dataset and identify potential outliers.

---

## 🔹 Cell 5 – Check Missing Values

```python
df.isnull().sum()
```

Before training any Machine Learning model, it is important to verify that the dataset does not contain missing values.

This step checks each column individually.

For the Social Network Ads dataset, all columns are complete with no missing values.

---

## 🔹 Cell 6 – Dataset Shape

```python
df.shape
```

The shape function returns the dimensions of the dataset.

For this dataset:

- Number of Rows: **400**
- Number of Columns: **5**

Understanding dataset dimensions helps estimate the amount of available training data.

---

## 🔹 Cell 7 – Feature and Target Selection

```python
df = df.drop("User ID", axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

The **User ID** column is removed because it is simply a unique identifier and has no predictive value.

The remaining data is divided into:

- **Features (X):**
  - Gender
  - Age
  - Estimated Salary

- **Target (y):**
  - Purchased

Separating the independent variables from the target variable prepares the dataset for model training.

---

## 🔹 Cell 8 – Encode the Gender Column

```python
encoder = LabelEncoder()

X["Gender"] = encoder.fit_transform(X["Gender"])
```

Machine Learning algorithms require numerical input.

Since the **Gender** column contains categorical values, Label Encoding converts these categories into numerical values that the Decision Tree algorithm can process.

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

The dataset is divided into two subsets:

- **Training Set (75%)** – Used to train the Decision Tree model.
- **Testing Set (25%)** – Used to evaluate the model's performance on unseen data.

Using separate training and testing datasets helps assess how well the model generalizes to new observations.

---

## 🔹 Cell 10 – Feature Scaling

```python
print("Decision Tree Classification does not require Feature Scaling.")
```

Unlike algorithms such as Logistic Regression, K-Nearest Neighbors, and Support Vector Machines, Decision Trees do not require feature scaling.

Decision Trees split data based on feature values rather than calculating distances between data points.

Because of this:

- Standardization is unnecessary.
- Normalization is unnecessary.
- Features with different scales do not affect the model's performance.

This makes Decision Trees one of the simplest Machine Learning algorithms in terms of data preprocessing.
---

## 🔹 Cell 11 – Train the Decision Tree Classifier

```python
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X_train, y_train)
```

After preprocessing the dataset, the Decision Tree Classification model is trained using the training dataset.

In this project, the **Entropy** criterion is used to determine the best feature for splitting the data at each node.

During training, the algorithm recursively creates decision rules until the stopping conditions are satisfied.

---

## 🔹 Cell 12 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained Decision Tree model predicts whether each customer in the testing dataset purchased the advertised product.

The predicted class labels are stored in **`y_pred`**, which are later compared with the actual values to evaluate model performance.

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

This provides a quick overview of the model's predictions and helps identify correctly and incorrectly classified observations.

---

## 🔹 Cell 14 – Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The Confusion Matrix summarizes the performance of the Decision Tree classifier.

It contains:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

A heatmap visualization makes it easier to interpret the classification results.

---

## 🔹 Cell 15 – Accuracy Score

```python
accuracy = accuracy_score(y_test, y_pred)
```

The Accuracy Score measures the percentage of correctly classified observations.

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

The Classification Report provides several important evaluation metrics:

- Precision
- Recall
- F1-Score
- Support

These metrics offer a more detailed evaluation of the classifier than accuracy alone, especially when class distributions are imbalanced.

---

## 🔹 Cell 17 – ROC Curve and AUC Score

```python
roc_curve()
roc_auc_score()
```

The ROC (Receiver Operating Characteristic) Curve illustrates how well the Decision Tree classifier distinguishes between the two target classes.

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

prediction = model.predict(new_customer)
```

After training, the model is used to predict whether a new customer is likely to purchase the advertised product based on the provided demographic information.

This demonstrates how the trained Decision Tree model can be applied to real-world prediction tasks.

---

## 🔹 Cell 19 – Visualize the Decision Tree

```python
plot_tree(model)
```

One of the biggest advantages of Decision Trees is their interpretability.

The complete tree structure is visualized using Scikit-learn's `plot_tree()` function.

The visualization displays:

- Decision nodes
- Splitting conditions
- Class labels
- Sample counts
- Predicted classes

This allows users to understand exactly how the model makes classification decisions.

---

## 🔹 Cell 20 – Feature Importance

```python
model.feature_importances_
```

Decision Trees automatically calculate the importance of each feature based on how much it contributes to reducing impurity.

The feature importance values are displayed in both a table and a bar chart.

Features with higher importance have a greater influence on the model's predictions.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Decision Tree Classification model.

These include:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score

Together, these metrics provide a comprehensive understanding of the classifier's effectiveness.

---

## 📈 Accuracy

Accuracy measures the proportion of correctly classified observations.

### Formula

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Interpretation

- High Accuracy indicates that the classifier makes correct predictions for most observations.
- Accuracy is most informative when the dataset is balanced.

---

## 📈 Precision

Precision measures how many predicted positive observations are actually positive.

### Formula

```text
Precision = TP / (TP + FP)
```

### Interpretation

Higher Precision means fewer False Positives.

---

## 📈 Recall

Recall measures how many actual positive observations are correctly identified.

### Formula

```text
Recall = TP / (TP + FN)
```

### Interpretation

Higher Recall means fewer False Negatives.

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

- True Positive Rate (Sensitivity)
- False Positive Rate

A curve closer to the upper-left corner indicates a stronger classifier.

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

Decision Tree Classification is widely used across many industries, including:

- Customer Purchase Prediction
- Medical Diagnosis
- Credit Risk Analysis
- Loan Approval Systems
- Fraud Detection
- Employee Attrition Prediction
- Customer Churn Analysis
- Spam Email Detection
- Marketing Campaign Analysis
- Insurance Claim Prediction

---

# ✅ Advantages of Decision Trees

- Simple and easy to understand.
- Produces highly interpretable decision rules.
- Handles both numerical and categorical data.
- Requires minimal data preprocessing.
- Does not require feature scaling.
- Captures non-linear relationships.
- Performs automatic feature selection.
- Supports both classification and regression tasks.

---

# ❌ Limitations of Decision Trees

- Can easily overfit the training data.
- Small changes in data may produce a different tree.
- Deep trees may become complex.
- Can be biased toward dominant classes.
- Usually performs better when combined into ensemble methods such as Random Forest.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Decision-Tree-Classification.git
```

Navigate to the project directory:

```bash
cd Decision-Tree-Classification
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
Decision Tree Classification.ipynb
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

Install all required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand the fundamentals of Decision Tree Classification.
- Explore and preprocess a real-world dataset.
- Encode categorical features using Label Encoding.
- Split data into training and testing sets.
- Train a Decision Tree Classification model.
- Make predictions on unseen data.
- Evaluate model performance using Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC.
- Interpret the model using a Decision Tree visualization.
- Analyze feature importance to understand which variables influence predictions the most.
- Apply Decision Tree Classification to real-world classification problems.

---

# 📝 Conclusion

Decision Tree Classification is one of the most intuitive and interpretable supervised Machine Learning algorithms. It learns decision rules directly from the data by recursively splitting the dataset into smaller subsets, making it effective for solving binary and multi-class classification problems.

In this project, the Social Network Ads dataset was explored, the data was preprocessed, a Decision Tree model was trained using the Entropy criterion, predictions were generated, classification metrics were evaluated, the decision tree structure was visualized, and feature importance was analyzed.

This project demonstrates the complete workflow of building and evaluating a Decision Tree Classification model and provides a strong foundation for learning advanced ensemble algorithms such as Random Forest, AdaBoost, Gradient Boosting, and XGBoost.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
