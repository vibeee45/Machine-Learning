# 🧠 Naive Bayes Classification

A comprehensive Machine Learning project demonstrating **Gaussian Naive Bayes Classification** using the **Social_Network_Ads** dataset. This project covers the complete machine learning workflow, including data preprocessing, feature scaling, model training, prediction, performance evaluation, ROC analysis, and probability estimation.

The notebook is designed for beginners as well as intermediate learners who want to understand how the **Naive Bayes algorithm** classifies data using **Bayes' Theorem** and probability instead of distance measures or decision boundaries.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Classification?
- What is Naive Bayes?
- Why Use Naive Bayes?
- Bayes' Theorem
- Prior Probability
- Likelihood
- Posterior Probability
- Conditional Probability
- Naive Assumption
- Types of Naive Bayes
- Difference Between Logistic Regression and Naive Bayes
- Advantages of Naive Bayes
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually defining rules, Machine Learning algorithms automatically identify relationships within the data and use those patterns to make predictions for unseen observations.

Machine Learning is widely used in:

- Healthcare
- Banking
- Finance
- Marketing
- Recommendation Systems
- Fraud Detection
- Customer Analytics
- Image Recognition
- Natural Language Processing
- Email Spam Detection

Machine Learning algorithms are mainly classified into:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Naive Bayes belongs to the **Supervised Learning** category because it learns from labeled training data.

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

# 📊 What is Naive Bayes?

Naive Bayes is a supervised Machine Learning algorithm based on **Bayes' Theorem**.

It predicts the probability that an observation belongs to a particular class by combining prior knowledge with the observed feature values.

The algorithm is called **"Naive"** because it assumes that all input features are independent of one another, even though this assumption may not always hold in real-world datasets.

Despite this simple assumption, Naive Bayes performs remarkably well for many classification problems.

---

# ⭐ Why Use Naive Bayes?

Naive Bayes is one of the fastest and most efficient classification algorithms.

Some important advantages include:

- Extremely fast training and prediction.
- Easy to implement.
- Performs well on small datasets.
- Works efficiently with high-dimensional data.
- Produces probability estimates.
- Less computationally expensive than many other classifiers.
- Effective for binary and multi-class classification.

---

# 📖 Bayes' Theorem

Bayes' Theorem is the mathematical foundation of the Naive Bayes algorithm.

It calculates the probability of an event based on prior knowledge and observed evidence.

### Formula

```text
P(A|B) = ( P(B|A) × P(A) ) / P(B)
```

Where:

- **P(A|B)** → Posterior Probability
- **P(B|A)** → Likelihood
- **P(A)** → Prior Probability
- **P(B)** → Evidence

Bayes' Theorem updates the probability of a class whenever new evidence becomes available.

---

# 📌 Prior Probability

Prior Probability represents the probability of an event occurring before observing any evidence.

It reflects the initial belief about a class based solely on historical data.

Example:

If 60 out of 100 customers purchased a product,

```text
Prior Probability = 60 / 100 = 0.60
```

---

# 📌 Likelihood

Likelihood measures the probability of observing a particular feature value given that the observation belongs to a specific class.

It answers the question:

> "If the customer purchased the product, how likely is this feature value?"

Likelihood plays a central role in Bayes' Theorem.

---

# 📌 Posterior Probability

Posterior Probability is the updated probability obtained after combining the prior probability with the observed evidence.

The Naive Bayes classifier calculates the posterior probability for each class and assigns the observation to the class with the highest probability.

---

# 📌 Conditional Probability

Conditional Probability measures the probability of one event occurring given that another event has already occurred.

It is represented as:

```text
P(A|B)
```

which means:

**Probability of A given B**

Naive Bayes repeatedly computes conditional probabilities while making predictions.

---

# 📌 Naive Assumption

The Naive Bayes algorithm assumes that all input features are **conditionally independent** given the target class.

For example:

- Age
- Gender
- Estimated Salary

are assumed to be independent when predicting whether a customer purchased a product.

Although this assumption is often unrealistic, Naive Bayes still performs exceptionally well on many real-world datasets.

---

# 🌐 Types of Naive Bayes

Naive Bayes has several variants depending on the nature of the input data.

## 1. Gaussian Naive Bayes

Suitable for continuous numerical features.

Assumes that each feature follows a **Gaussian (Normal) Distribution**.

This project uses **Gaussian Naive Bayes** because the dataset contains numerical features such as Age and Estimated Salary.

---

## 2. Multinomial Naive Bayes

Suitable for count-based data.

Commonly used in:

- Text Classification
- Document Classification
- Spam Detection

---

## 3. Bernoulli Naive Bayes

Suitable for binary features.

Commonly used when each feature has only two possible values, such as:

- Yes / No
- True / False
- 0 / 1

---

# ⚖️ Difference Between Logistic Regression and Naive Bayes

| Logistic Regression | Naive Bayes |
|----------------------|-------------|
| Learns a mathematical decision boundary | Uses Bayes' Theorem |
| Does not assume feature independence | Assumes independent features |
| Optimization-based algorithm | Probability-based algorithm |
| Slower training | Very fast training |
| Better for complex relationships | Better for probabilistic classification |
| Requires optimization | No iterative optimization required |

---

# ✅ Advantages of Naive Bayes

- Very fast training.
- Very fast prediction.
- Easy to implement.
- Works well with high-dimensional datasets.
- Produces probability estimates.
- Handles multi-class classification naturally.
- Requires relatively small amounts of training data.
- Performs well in many real-world classification tasks.

---

# 📌 Assumptions of Naive Bayes

Although Naive Bayes is highly effective, good performance depends on:

- Features being reasonably independent.
- Representative training data.
- Appropriate feature preprocessing.
- Continuous features following a Gaussian distribution (for Gaussian Naive Bayes).

Feature Scaling is included in this project to maintain consistency with the other classification notebooks and to standardize the numerical features.

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
Train Gaussian Naive Bayes Model
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
Probability Visualization
        │
        ▼
Display Model Parameters
```
# 📂 Dataset Information

**Dataset Name:** Social_Network_Ads.csv

The dataset contains demographic information about customers and whether they purchased a product after viewing a social network advertisement.

The objective of this project is to build a **Gaussian Naive Bayes Classification** model that predicts whether a customer will purchase the advertised product based on their demographic details.

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

The Gaussian Naive Bayes model is trained using the following input features:

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
Naive-Bayes-Classification/
│
├── Naive Bayes Classification.ipynb
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
from sklearn.naive_bayes import GaussianNB
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
| GaussianNB | Build the Gaussian Naive Bayes Classification model |

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

This transformation allows the Gaussian Naive Bayes model to process categorical information.

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

- **Training Set (75%)** – Used to train the Gaussian Naive Bayes model.
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

Although Gaussian Naive Bayes is less sensitive to feature scaling than distance-based algorithms, standardizing numerical features helps maintain consistency across the workflow and ensures that features with larger numerical values do not dominate the learning process.

After scaling:

- Mean = 0
- Standard Deviation = 1

The scaled data is then used for training and evaluating the Gaussian Naive Bayes classifier.
---

## 🔹 Cell 11 – Train the Gaussian Naive Bayes Classifier

```python
model = GaussianNB()

model.fit(X_train, y_train)
```

After preprocessing and feature scaling, the Gaussian Naive Bayes model is trained using the training dataset.

In this project:

- **Gaussian Naive Bayes** is used.
- Continuous numerical features are assumed to follow a **Gaussian (Normal) Distribution**.
- The model learns the probability distribution of each feature for every target class.

Unlike many Machine Learning algorithms, Gaussian Naive Bayes predicts classes by calculating probabilities using **Bayes' Theorem**.

---

## 🔹 Cell 12 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained Gaussian Naive Bayes model predicts whether each customer in the testing dataset purchased the advertised product.

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

This provides a quick overview of how accurately the model classified the testing observations.

---

## 🔹 Cell 14 – Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The Confusion Matrix summarizes the classification performance of the Gaussian Naive Bayes model.

It consists of:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

A heatmap visualization provides an intuitive representation of these prediction results.

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

The ROC (Receiver Operating Characteristic) Curve illustrates how effectively the Gaussian Naive Bayes model distinguishes between the two target classes.

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

The scaled data is then passed to the trained Gaussian Naive Bayes model to predict whether the customer is likely to purchase the advertised product.

This demonstrates how the trained model can be applied to real-world prediction scenarios.

---

## 🔹 Cell 19 – Class Probability Visualization

```python
probabilities = model.predict_proba(X_test[:10])
```

One of the key strengths of Gaussian Naive Bayes is that it predicts **class probabilities** rather than only class labels.

This notebook displays the prediction probabilities for the first ten test samples using both a table and a bar chart.

The visualization helps understand:

- Probability of **Not Purchased**
- Probability of **Purchased**
- Confidence of each prediction

This provides valuable insight into how the model makes classification decisions.

---

## 🔹 Cell 20 – Model Parameters

```python
model.get_params()
```

The final notebook cell displays the parameters used by the Gaussian Naive Bayes classifier.

Important parameters include:

- **priors** – Prior probabilities of each class (if specified).
- **var_smoothing** – Portion of the largest feature variance added for numerical stability.

Understanding these parameters helps improve model stability and performance when working with different datasets.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Gaussian Naive Bayes classifier.

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

Gaussian Naive Bayes is widely used in many real-world applications, including:

- Email Spam Detection
- Text Classification
- Sentiment Analysis
- Medical Diagnosis
- Document Categorization
- News Classification
- Customer Purchase Prediction
- Recommendation Systems
- Language Detection
- Fraud Detection

---

# ✅ Advantages of Gaussian Naive Bayes

- Simple and easy to implement.
- Extremely fast training and prediction.
- Works well with high-dimensional datasets.
- Produces probability estimates.
- Handles binary and multi-class classification.
- Performs well even with relatively small training datasets.
- Computationally efficient.

---

# ❌ Limitations of Gaussian Naive Bayes

- Assumes feature independence.
- Performance decreases when features are highly correlated.
- Assumes numerical features follow a Gaussian distribution.
- May not perform well on very complex decision boundaries.
- Sensitive to incorrect probability assumptions.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Naive-Bayes-Classification.git
```

Navigate to the project directory:

```bash
cd Naive-Bayes-Classification
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
Naive Bayes Classification.ipynb
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

- Understand Bayes' Theorem and probability-based classification.
- Learn Prior, Posterior, Likelihood, and Conditional Probability.
- Understand the Naive Assumption.
- Build a Gaussian Naive Bayes classifier using Scikit-learn.
- Evaluate classification models using Accuracy, Precision, Recall, F1-Score, ROC Curve, and AUC.
- Interpret prediction probabilities.
- Apply Gaussian Naive Bayes to real-world classification problems.

---

# 📝 Conclusion

Gaussian Naive Bayes is one of the fastest and most efficient supervised Machine Learning algorithms for probabilistic classification. By applying Bayes' Theorem and assuming conditional independence between features, it provides accurate predictions while remaining computationally efficient.

In this project, the Social Network Ads dataset was explored, preprocessed, feature-scaled, used to train a Gaussian Naive Bayes classifier, evaluated using multiple classification metrics, and analyzed through prediction probabilities and model parameters.

This project provides a strong foundation for understanding probability-based Machine Learning algorithms and their practical applications.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
