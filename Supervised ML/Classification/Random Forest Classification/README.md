# 🌲 Random Forest Classification

A comprehensive Machine Learning project demonstrating **Random Forest Classification** using the **Social_Network_Ads** dataset. This project covers the complete machine learning workflow, including data preprocessing, model training, prediction, performance evaluation, ROC analysis, feature importance visualization, and hyperparameter inspection.

The notebook is designed for beginners as well as intermediate learners who want to understand how ensemble learning improves the performance of Decision Trees by combining multiple trees into a single robust model.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Classification?
- What is Random Forest Classification?
- Why Use Random Forest?
- Ensemble Learning
- Bootstrap Sampling
- Bagging
- Majority Voting
- Difference Between Decision Tree and Random Forest
- Advantages of Random Forest
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually defining rules, Machine Learning algorithms automatically identify relationships within the data and use those patterns to predict outcomes for new observations.

Machine Learning is widely used in:

- Healthcare
- Banking
- Finance
- Marketing
- Recommendation Systems
- Fraud Detection
- Image Recognition
- Natural Language Processing
- Customer Analytics

Machine Learning algorithms are mainly classified into:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Random Forest Classification belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 🎯 What is Classification?

Classification is a supervised learning technique used to predict discrete categories or class labels.

The objective is to assign each observation to one of the predefined classes based on the available input features.

Examples include:

- Email Spam Detection
- Disease Prediction
- Credit Card Fraud Detection
- Customer Purchase Prediction
- Loan Approval
- Sentiment Analysis

In this project, the model predicts whether a customer will purchase a product after viewing a social network advertisement.

Target Classes:

- **0 → Not Purchased**
- **1 → Purchased**

---

# 🌲 What is Random Forest Classification?

Random Forest Classification is an ensemble Machine Learning algorithm that combines multiple Decision Trees to produce a more accurate and stable prediction.

Instead of relying on a single Decision Tree, Random Forest creates many trees using different subsets of the training data and randomly selected features.

Each tree makes its own prediction, and the final prediction is determined using **Majority Voting**.

This approach significantly improves prediction accuracy while reducing overfitting, making Random Forest one of the most widely used classification algorithms in Machine Learning.

---

# ⭐ Why Use Random Forest?

Random Forest is one of the most powerful and versatile Machine Learning algorithms because it combines the predictions of multiple Decision Trees.

Some key advantages include:

- High prediction accuracy.
- Reduces overfitting compared to a single Decision Tree.
- Handles both numerical and categorical data.
- Works well on large datasets.
- Automatically estimates feature importance.
- Requires minimal preprocessing.
- Robust against noisy data and outliers.

---

# 🌐 Ensemble Learning

Ensemble Learning is a Machine Learning technique where multiple models are combined to produce better predictions than a single model.

Instead of depending on one classifier, ensemble methods aggregate predictions from several models to improve accuracy and stability.

Common ensemble techniques include:

- Bagging
- Boosting
- Stacking

Random Forest is based on the **Bagging** technique.

---

# 📦 Bootstrap Sampling

Bootstrap Sampling is the process of creating multiple training datasets by randomly sampling observations **with replacement** from the original dataset.

This means:

- Some observations may appear multiple times.
- Some observations may not appear at all.

Each bootstrap sample is used to train one Decision Tree.

This diversity among the trees helps improve the overall performance of the Random Forest model.

---

# 🌳 Bagging (Bootstrap Aggregating)

Bagging is an ensemble learning technique that combines predictions from multiple models trained on different bootstrap samples.

The process is:

1. Create multiple bootstrap datasets.
2. Train one Decision Tree on each dataset.
3. Collect predictions from all trees.
4. Combine the predictions using Majority Voting.

Bagging reduces variance and improves the model's ability to generalize to unseen data.

---

# 🗳️ Majority Voting

For classification problems, every Decision Tree in the Random Forest predicts a class label.

The final prediction is the class that receives the highest number of votes from all the trees.

Example:

| Tree | Prediction |
|------|------------|
| Tree 1 | Purchased |
| Tree 2 | Purchased |
| Tree 3 | Not Purchased |
| Tree 4 | Purchased |
| Tree 5 | Purchased |

Final Prediction:

**Purchased (4 out of 5 votes)**

This voting mechanism makes Random Forest more reliable than a single Decision Tree.

---

# ⚖️ Difference Between Decision Tree and Random Forest

| Decision Tree | Random Forest |
|---------------|---------------|
| Uses a single tree | Uses multiple Decision Trees |
| More prone to overfitting | Reduces overfitting |
| Lower prediction stability | Higher prediction stability |
| Faster training | Slightly slower due to multiple trees |
| Easier to interpret | Less interpretable because many trees are used |
| Does not use bagging | Uses Bootstrap Aggregating (Bagging) |
| Single prediction | Majority Voting from multiple trees |

---

# ✅ Advantages of Random Forest

- High prediction accuracy.
- Less prone to overfitting.
- Handles large datasets efficiently.
- Works well with missing and noisy data.
- Provides feature importance scores.
- Supports parallel training.
- Performs well for both classification and regression.
- Requires little data preprocessing.

---

# 📌 Assumptions of Random Forest

Although Random Forest is flexible, good performance depends on:

- Representative training data.
- Relevant input features.
- Proper selection of hyperparameters.
- Sufficient number of Decision Trees.
- Balanced class distribution for reliable classification.

Unlike many algorithms, Random Forest does **not** require feature scaling or assume a linear relationship between variables.

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
Train Random Forest Model
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
Feature Importance
        │
        ▼
Hyperparameter Analysis
```
# 📂 Dataset Information

**Dataset Name:** Social_Network_Ads.csv

The dataset contains demographic information about customers and whether they purchased a product after viewing a social network advertisement.

The objective of this project is to build a **Random Forest Classification** model that predicts whether a customer will purchase the advertised product based on their demographic details.

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

In this project, the **User ID** column is removed because it is only an identifier and has no predictive value.

The Random Forest model is trained using the following input features:

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
Random-Forest-Classification/
│
├── Random Forest Classification.ipynb
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
from sklearn.ensemble import RandomForestClassifier
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
| LabelEncoder | Convert categorical values into numerical values |
| RandomForestClassifier | Build the Random Forest Classification model |

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

Displaying the first few records verifies that the dataset has been imported correctly and provides an overview of its structure.

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

This function generates descriptive statistics for the numerical features.

The output includes:

- Count
- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles

These statistics help understand the distribution of the data and identify potential outliers.

---

## 🔹 Cell 5 – Check Missing Values

```python
df.isnull().sum()
```

Before training the model, it is important to verify that the dataset does not contain missing values.

This step checks each column individually.

For the Social Network Ads dataset, all columns are complete and contain no missing values.

---

## 🔹 Cell 6 – Dataset Shape

```python
df.shape
```

The shape function returns the dimensions of the dataset.

For this dataset:

- Number of Rows: **400**
- Number of Columns: **5**

Understanding the dataset dimensions provides an overview of the available training data.

---

## 🔹 Cell 7 – Feature and Target Selection

```python
if "User ID" in df.columns:
    df = df.drop("User ID", axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

The **User ID** column is removed because it serves only as a unique identifier and does not contribute to prediction.

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

Since the **Gender** column contains categorical values, Label Encoding converts these categories into numerical values that the Random Forest algorithm can process.

After encoding:

- Female → 0
- Male → 1

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

- **Training Set (75%)** – Used to train the Random Forest model.
- **Testing Set (25%)** – Used to evaluate the model's performance on unseen data.

Using separate training and testing datasets helps determine how well the model generalizes to new observations.

---

## 🔹 Cell 10 – Feature Scaling

```python
print("Random Forest Classification does not require Feature Scaling.")
```

Unlike algorithms such as Logistic Regression, Support Vector Machine, and K-Nearest Neighbors, Random Forest does not require feature scaling.

Random Forest builds multiple Decision Trees by splitting the data based on feature values rather than calculating distances.

Therefore:

- Standardization is unnecessary.
- Normalization is unnecessary.
- Different feature scales do not affect model performance.

This makes Random Forest one of the easiest Machine Learning algorithms to prepare data for before training.
---

## 🔹 Cell 11 – Train the Random Forest Classifier

```python
model = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    random_state=42
)

model.fit(X_train, y_train)
```

After preprocessing the dataset, the Random Forest Classification model is trained using the training dataset.

In this project:

- **100 Decision Trees** are created.
- **Entropy** is used as the splitting criterion.
- Each tree is trained using a different bootstrap sample.
- The final prediction is obtained through **Majority Voting**.

This ensemble approach improves prediction accuracy and reduces overfitting compared to a single Decision Tree.

---

## 🔹 Cell 12 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained Random Forest model predicts whether each customer in the testing dataset purchased the advertised product.

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

A comparison table is created containing:

- Actual Purchase Status
- Predicted Purchase Status

This allows a quick comparison between the model's predictions and the true labels.

---

## 🔹 Cell 14 – Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The Confusion Matrix summarizes the performance of the Random Forest classifier.

It consists of:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

A heatmap visualization makes these results easier to interpret.

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

These metrics provide a more complete evaluation than accuracy alone.

---

## 🔹 Cell 17 – ROC Curve and AUC Score

```python
roc_curve()
roc_auc_score()
```

The ROC (Receiver Operating Characteristic) Curve illustrates how effectively the Random Forest model distinguishes between the two target classes.

The Area Under the Curve (AUC) summarizes the model's overall classification ability.

Interpretation:

- AUC = 1.0 → Perfect Classifier
- AUC = 0.5 → Random Guessing

Higher AUC values indicate better classification performance.

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

After training, the Random Forest model predicts whether a new customer is likely to purchase the advertised product based on the provided demographic information.

This demonstrates how the trained model can be used for real-world prediction tasks.

---

## 🔹 Cell 19 – Feature Importance

```python
model.feature_importances_
```

One of the major advantages of Random Forest is its ability to estimate feature importance automatically.

Each feature receives an importance score based on how much it contributes to reducing impurity across all Decision Trees.

The feature importance values are displayed both as a table and as a bar chart for easier interpretation.

---

## 🔹 Cell 20 – Random Forest Hyperparameters

```python
model.get_params()
```

The final notebook cell displays the hyperparameters used to train the Random Forest model.

Some important hyperparameters include:

- **n_estimators** – Number of Decision Trees.
- **criterion** – Splitting criterion (Entropy).
- **random_state** – Ensures reproducible results.
- **max_depth** – Maximum depth of each tree.
- **max_features** – Number of features considered at each split.
- **min_samples_split** – Minimum samples required to split a node.
- **min_samples_leaf** – Minimum samples required at a leaf node.

Understanding these hyperparameters is essential for optimizing Random Forest performance.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Random Forest Classification model.

These include:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score

Together, these metrics provide a comprehensive understanding of the classifier's performance.

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

Random Forest Classification is widely used in many real-world domains, including:

- Customer Purchase Prediction
- Credit Risk Assessment
- Fraud Detection
- Disease Diagnosis
- Loan Approval Systems
- Employee Attrition Prediction
- Customer Churn Prediction
- Spam Email Detection
- Insurance Risk Analysis
- Marketing Campaign Optimization

---

# ✅ Advantages of Random Forest

- High prediction accuracy.
- Reduces overfitting.
- Handles large datasets efficiently.
- Supports both numerical and categorical data.
- Resistant to noisy data and outliers.
- Provides feature importance scores.
- Works well for both classification and regression.
- Requires minimal preprocessing.

---

# ❌ Limitations of Random Forest

- More computationally expensive than a single Decision Tree.
- Requires more memory.
- Slower training with a large number of trees.
- Less interpretable because it consists of many Decision Trees.
- Hyperparameter tuning may be required for optimal performance.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Random-Forest-Classification.git
```

Navigate to the project directory:

```bash
cd Random-Forest-Classification
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
Random Forest Classification.ipynb
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

- Understand the concept of Ensemble Learning.
- Understand Bootstrap Sampling and Bagging.
- Build a Random Forest Classification model.
- Preprocess and prepare a real-world dataset.
- Encode categorical variables.
- Split datasets into training and testing sets.
- Evaluate classification models using Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC.
- Analyze Feature Importance.
- Understand important Random Forest hyperparameters.
- Apply Random Forest Classification to real-world prediction problems.

---

# 📝 Conclusion

Random Forest Classification is one of the most powerful and reliable supervised Machine Learning algorithms. By combining multiple Decision Trees through Bootstrap Aggregating (Bagging) and Majority Voting, it achieves higher prediction accuracy while significantly reducing overfitting.

In this project, the Social Network Ads dataset was explored, the data was preprocessed, a Random Forest model was trained, predictions were generated, classification metrics were evaluated, feature importance was analyzed, and the model's hyperparameters were examined.

This project demonstrates a complete Machine Learning classification workflow and provides a strong foundation for advanced ensemble methods such as Gradient Boosting, AdaBoost, XGBoost, LightGBM, and CatBoost.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
