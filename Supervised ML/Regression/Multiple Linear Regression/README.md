# 📈 Multiple Linear Regression

## Overview

Multiple Linear Regression is one of the most widely used supervised machine learning algorithms for predicting continuous numerical values using **multiple independent variables**. Unlike Simple Linear Regression, which uses only one feature to make predictions, Multiple Linear Regression analyzes the combined effect of two or more independent variables on a dependent variable.

This project demonstrates how to build, preprocess, train, evaluate, and interpret a Multiple Linear Regression model using **Python** and **Scikit-learn** on the **50 Startups Dataset**. The project also introduces important preprocessing techniques such as **One-Hot Encoding**, **ColumnTransformer**, and **Pipeline**, which are essential when working with datasets containing categorical features.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Linear Regression?
- What is Multiple Linear Regression?
- Why Use Multiple Linear Regression?
- Regression Equation
- Assumptions of Multiple Linear Regression
- Project Workflow
- Dataset Information
- Technologies Used
- Project Structure
- Step-by-Step Notebook Explanation (Cells 1–11)

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

Instead of writing fixed rules for every possible situation, Machine Learning algorithms learn from historical data and improve their performance as they encounter more examples.

Machine Learning is mainly divided into three categories:

- **Supervised Learning**
- **Unsupervised Learning**
- **Reinforcement Learning**

Since the **50 Startups Dataset** contains labeled data (features and corresponding profit values), **Multiple Linear Regression** belongs to **Supervised Learning**.

---

# 📊 What is Linear Regression?

Linear Regression is a supervised learning algorithm used to predict continuous numerical values.

It models the relationship between one or more independent variables and a dependent variable by fitting the best possible straight line (or hyperplane in higher dimensions) through the data.

The primary objective of Linear Regression is to minimize the prediction error between the actual and predicted values.

Linear Regression is one of the most important algorithms because it forms the foundation for understanding many advanced Machine Learning techniques.

---

# 📈 What is Multiple Linear Regression?

Multiple Linear Regression is an extension of Simple Linear Regression.

Instead of using only one independent variable, Multiple Linear Regression uses **two or more independent variables** to predict a single dependent variable.

### In this project:

**Independent Variables (Features):**

- R&D Spend
- Administration
- Marketing Spend
- State

**Dependent Variable (Target):**

- Profit

The model learns how each feature contributes to the startup's profit and then predicts profit for unseen startups.

---

# 🎯 Why Use Multiple Linear Regression?

Multiple Linear Regression is used to:

- Predict continuous numerical values.
- Analyze relationships between multiple variables.
- Estimate business profits.
- Forecast sales and revenue.
- Predict house prices.
- Perform financial analysis.
- Understand feature importance.
- Build baseline regression models.

---

# 📐 Regression Equation

The mathematical equation of Multiple Linear Regression is:

```text
Ŷ = b₀ + b₁X₁ + b₂X₂ + b₃X₃ + ... + bₙXₙ
```

### Where

| Symbol | Description |
|---------|-------------|
| Ŷ | Predicted Value |
| X₁, X₂ ... Xₙ | Independent Variables |
| b₀ | Intercept |
| b₁, b₂ ... bₙ | Regression Coefficients |

Each coefficient represents the effect of its corresponding feature on the predicted output while keeping the other features constant.

---

# 📋 Assumptions of Multiple Linear Regression

For Multiple Linear Regression to perform well, the following assumptions should generally hold:

- Linear relationship between independent and dependent variables.
- Independence of observations.
- Homoscedasticity (constant variance of residuals).
- Normally distributed residuals.
- Minimal multicollinearity among independent variables.
- No significant outliers.

These assumptions help ensure that the model provides reliable and unbiased predictions.

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
Split Features & Target
        │
        ▼
Train-Test Split
        │
        ▼
One-Hot Encoding
        │
        ▼
ColumnTransformer
        │
        ▼
Pipeline
        │
        ▼
Train Model
        │
        ▼
Predict Values
        │
        ▼
Evaluate Model
        │
        ▼
Interpret Results
```

---

# 📂 Dataset Information

**Dataset Name:** 50_Startups.csv

The dataset contains information about startup expenditures and their corresponding profits.

## Dataset Summary

| Property | Value |
|----------|-------|
| Number of Rows | 50 |
| Number of Columns | 5 |
| Missing Values | No |
| Numerical Features | 3 |
| Categorical Features | 1 |
| Target Variable | Profit |

---

## Features

| Feature | Description |
|----------|-------------|
| R&D Spend | Amount invested in research and development |
| Administration | Administrative expenditure |
| Marketing Spend | Marketing expenditure |
| State | State where the startup operates |
| Profit | Annual profit of the startup |

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# 📁 Project Structure

```text
Multiple-Linear-Regression/
│
├── Multiple_Linear_Regression.ipynb
├── 50_Startups.csv
├── README.md
└── requirements.txt
```

---

# 📖 Step-by-Step Notebook Explanation

---

## 🔹 Cell 1 – Import Libraries

The first step of every Machine Learning project is importing the required libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Purpose

Each library has a specific responsibility.

| Library | Purpose |
|----------|----------|
| NumPy | Numerical computations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| train_test_split | Divide dataset into training and testing sets |
| OneHotEncoder | Encode categorical variables |
| ColumnTransformer | Apply preprocessing to selected columns |
| Pipeline | Combine preprocessing and model training |
| LinearRegression | Train regression model |
| MSE & R² | Evaluate model performance |

---

## 🔹 Cell 2 – Load Dataset

```python
dataset = pd.read_csv("50_Startups.csv")
```

The dataset is loaded into a Pandas DataFrame.

A DataFrame stores data in a tabular format consisting of rows and columns, making it easier to inspect, clean, and manipulate data.

---

## 🔹 Cell 3 – Display Dataset Information

```python
dataset.info()
```

This provides:

- Number of rows
- Number of columns
- Data types
- Missing values
- Memory usage

Understanding the dataset structure before training is an important step in every Machine Learning workflow.

---

## 🔹 Cell 4 – Statistical Summary

```python
dataset.describe()
```

This generates summary statistics such as:

- Count
- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles

These statistics help understand the distribution of numerical features.

---

## 🔹 Cell 5 – Check Missing Values

```python
dataset.isnull().sum()
```

Missing values can negatively affect model performance.

This step checks every column to ensure that no values are missing before training.

In this dataset, all columns contain complete data.

---

## 🔹 Cell 6 – Dataset Shape

```python
dataset.shape
```

This returns:

- Number of rows
- Number of columns

For this dataset:

- Rows: 50
- Columns: 5

---

## 🔹 Cell 7 – Separate Features and Target

```python
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
```

The dataset is divided into:

- **X (Independent Variables):** All columns except Profit
- **y (Dependent Variable):** Profit

The target variable is what the model learns to predict.

---

## 🔹 Cell 8 – Display Features and Target

```python
X.head()
y.head()
```

This step verifies that the feature matrix (`X`) and target variable (`y`) have been separated correctly before model training.

---

## 🔹 Cell 9 – Train-Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is divided into:

- **Training Set (80%)** – Used for learning patterns.
- **Testing Set (20%)** – Used for evaluating model performance.

Using separate training and testing data helps measure how well the model generalizes to unseen data.

---

## 🔹 Cell 10 – Data Preprocessing using ColumnTransformer

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), ["State"])
    ],
    remainder="passthrough"
)
```

The **State** column contains categorical values (text), which cannot be directly processed by the regression model.

To convert these categories into numerical form, **One-Hot Encoding** is applied.

`drop="first"` is used to avoid the **Dummy Variable Trap**, which can introduce multicollinearity by creating redundant features.

`ColumnTransformer` ensures that encoding is applied only to the `State` column while all remaining numerical columns pass through unchanged.

---

## 🔹 Cell 11 – Create the Machine Learning Pipeline

```python
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
```

A **Pipeline** combines preprocessing and model training into a single workflow.

When `fit()` or `predict()` is called, the pipeline automatically:

1. Applies One-Hot Encoding.
2. Transforms the dataset.
3. Trains or predicts using the Linear Regression model.

---

## 🔹 Cell 12 – Train the Model

```python
model.fit(X_train, y_train)
```

The `fit()` method trains the Multiple Linear Regression model using the training dataset.

Since we are using a Pipeline, the following operations happen automatically:

1. One-Hot Encoding is applied to the `State` column.
2. Numerical features remain unchanged.
3. The transformed data is passed to the Linear Regression algorithm.
4. The algorithm learns the relationship between the independent variables and the target variable (Profit).

After training, the model is ready to make predictions on unseen data.

---

## 🔹 Cell 13 – Predict the Test Set Results

```python
y_pred = model.predict(X_test)
```

The trained model predicts the profit for every startup in the testing dataset.

These predicted values are stored in `y_pred`.

The predictions are later compared with the actual profit values to evaluate the model's performance.

---

## 🔹 Cell 14 – Compare Actual vs Predicted Values

```python
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
```

This step creates a comparison table showing:

- Actual Profit
- Predicted Profit

Comparing these values provides a quick understanding of how closely the model's predictions match the real values.

Smaller differences indicate better prediction accuracy.

---

## 🔹 Cell 15 – Calculate Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

Mean Squared Error (MSE) measures the average squared difference between the actual and predicted values.

A lower MSE indicates that the model's predictions are closer to the actual values.

MSE is always greater than or equal to zero.

- Lower MSE → Better model
- Higher MSE → Poorer model

---

## 🔹 Cell 16 – Calculate R² Score

```python
r2 = r2_score(y_test, y_pred)
```

The R² Score (Coefficient of Determination) measures how well the regression model explains the variation in the target variable.

Its value ranges from **0 to 1**.

- **R² = 1** → Perfect prediction
- **R² close to 1** → Excellent model
- **R² close to 0** → Weak model

Higher R² values indicate better predictive performance.

---

## 🔹 Cell 17 – Display the Intercept

```python
model.named_steps["regressor"].intercept_
```

The intercept (`b₀`) is the constant term in the regression equation.

It represents the predicted profit when all independent variables have a value of zero.

Although this situation may not always occur in practice, the intercept is an essential part of the regression equation.

---

## 🔹 Cell 18 – Display Feature Names

```python
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
```

After One-Hot Encoding, the original categorical column (`State`) is converted into multiple numerical columns.

This step displays the transformed feature names used by the regression model.

It helps us understand how the preprocessing stage modified the dataset.

---

## 🔹 Cell 19 – Display Feature Coefficients

```python
coefficients = model.named_steps["regressor"].coef_
```

Each coefficient indicates the effect of its corresponding feature on the predicted profit.

- Positive coefficient → Profit increases as the feature increases.
- Negative coefficient → Profit decreases as the feature increases.

Larger absolute coefficient values generally indicate a stronger influence on the prediction.

---

## 🔹 Cell 20 – Predict Profit for a New Startup

```python
new_startup = pd.DataFrame({
    ...
})

prediction = model.predict(new_startup)
```

The trained model is used to predict the profit of a new startup using fresh input values.

The input must contain the same features used during training:

- State
- R&D Spend
- Administration
- Marketing Spend

Since the model uses a Pipeline, preprocessing is performed automatically before prediction.

---

## 🔹 Cell 21 – Residual Analysis

Residuals represent the difference between the actual and predicted values.

```
Residual = Actual − Predicted
```

Analyzing residuals helps determine whether the regression model satisfies its assumptions.

Ideally, residuals should appear randomly distributed around zero with no clear pattern.

This indicates that the model has captured the relationship in the data effectively.

---

## 🔹 Cell 22 – Training Score vs Testing Score

```python
model.score(X_train, y_train)
model.score(X_test, y_test)
```

This step compares the model's performance on:

- Training Dataset
- Testing Dataset

### Interpretation

- Similar training and testing scores indicate a well-generalized model.
- Very high training score but much lower testing score may indicate **overfitting**.
- Low scores on both datasets may indicate **underfitting**.

Comparing these scores helps evaluate how well the model performs on unseen data.

---

# 🔄 Difference Between Simple and Multiple Linear Regression

Although both algorithms belong to the Linear Regression family, they differ in the number of input features and preprocessing requirements.

| Feature | Simple Linear Regression | Multiple Linear Regression |
|----------|--------------------------|----------------------------|
| Independent Variables | One | Two or More |
| Regression Equation | Ŷ = b₀ + b₁X | Ŷ = b₀ + b₁X₁ + b₂X₂ + ... + bₙXₙ |
| Input Type | Usually NumPy Array | Pandas DataFrame |
| Categorical Features | Usually Not Present | Can Be Present |
| One-Hot Encoding | Not Required | Required (if categorical features exist) |
| ColumnTransformer | Not Required | Required |
| Pipeline | Optional | Recommended |

### Why Use a DataFrame in Multiple Linear Regression?

In Simple Linear Regression, the input generally contains only numerical values, so converting the data to a NumPy array using `.values` is sufficient.

```python
X = dataset.iloc[:, :-1].values
```

However, the Multiple Linear Regression dataset contains a categorical feature (`State`).

To preserve column names and allow `ColumnTransformer` and `OneHotEncoder` to identify the categorical column, the feature matrix is kept as a Pandas DataFrame.

```python
X = dataset.iloc[:, :-1]
```

This enables automatic preprocessing without manually selecting column indices.

---

# 📊 Model Evaluation Metrics

## 1. Mean Squared Error (MSE)

Mean Squared Error measures the average squared difference between actual and predicted values.

### Formula

```text
                Σ (yi − ŷi)²
MSE =  --------------------------
                  n
```

### Where

| Symbol | Description |
|--------|-------------|
| yi | Actual Value |
| ŷi | Predicted Value |
| n | Number of Observations |
| Σ | Summation |

### Interpretation

- Lower MSE indicates better predictions.
- MSE = 0 represents perfect predictions.

---

## 2. R² Score (Coefficient of Determination)

The R² Score measures how much of the variation in the target variable is explained by the regression model.

### Formula

```text
                Σ (yi − ŷi)²
R² = 1 − ------------------------------
          Σ (yi − ȳ)²
```

### Interpretation

| R² Score | Model Performance |
|----------|-------------------|
| 1.00 | Perfect Fit |
| 0.90 – 0.99 | Excellent |
| 0.70 – 0.89 | Good |
| 0.50 – 0.69 | Fair |
| Below 0.50 | Poor |

---

# 📚 Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# 🌍 Applications of Multiple Linear Regression

Multiple Linear Regression is widely used in:

- Startup Profit Prediction
- House Price Prediction
- Sales Forecasting
- Financial Analysis
- Marketing Analytics
- Business Intelligence
- Healthcare Cost Prediction
- Demand Forecasting
- Risk Assessment
- Economic Analysis

---

# ✅ Advantages

- Easy to understand and interpret.
- Uses multiple features for improved predictions.
- Helps identify feature importance.
- Fast to train and predict.
- Strong baseline regression model.
- Widely used in business analytics and research.

---

# ❌ Limitations

- Assumes a linear relationship between variables.
- Sensitive to outliers.
- Performance decreases if assumptions are violated.
- Requires preprocessing for categorical features.
- Multicollinearity can affect coefficient interpretation.

---

# 🚀 Installation & Usage

Clone the repository:

```bash
git clone https://github.com/vibeee45/Machine-Learning.git
```

Navigate to the project directory:

```bash
cd Machine-Learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
Multiple_Linear_Regression.ipynb
```

Run all notebook cells sequentially.

---

# 📦 Requirements

```
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Or install manually:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

---

# 🎓 Learning Outcomes

After completing this project, you will understand:

- Supervised Learning
- Linear Regression
- Multiple Linear Regression
- Feature and Target Selection
- Data Exploration
- Train-Test Split
- One-Hot Encoding
- ColumnTransformer
- Pipeline
- Model Training
- Prediction
- Mean Squared Error (MSE)
- R² Score
- Model Interpretation
- Residual Analysis
- Overfitting and Underfitting

---

# 🏁 Conclusion

Multiple Linear Regression is one of the most important supervised learning algorithms for predicting continuous numerical values using multiple independent variables.

In this project, we built a complete Machine Learning pipeline using the **50 Startups Dataset**, explored the data, handled categorical features through **One-Hot Encoding**, automated preprocessing with **ColumnTransformer** and **Pipeline**, trained a regression model, evaluated its performance using **Mean Squared Error (MSE)** and **R² Score**, interpreted feature coefficients, and predicted the profit of new startups.

Understanding Multiple Linear Regression provides a strong foundation for learning advanced regression techniques such as **Polynomial Regression**, **Support Vector Regression (SVR)**, **Decision Tree Regression**, and **Random Forest Regression**.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

GitHub: https://github.com/vibeee45

⭐ If you found this project helpful, consider giving the repository a **Star** on GitHub!

Using a Pipeline simplifies the code, reduces manual preprocessing, prevents data leakage, and ensures the same preprocessing steps are consistently applied during both training and prediction.

---
