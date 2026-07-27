# 📈 Polynomial Regression

## Overview

Polynomial Regression is one of the most widely used supervised Machine Learning algorithms for modeling **non-linear relationships** between an independent variable and a dependent variable. Unlike Simple Linear Regression, which fits a straight line through the data, Polynomial Regression transforms the original feature into polynomial terms, allowing the model to learn curved relationships.

Although the resulting model produces a non-linear curve, Polynomial Regression is still based on **Linear Regression**, because the algorithm remains linear with respect to its coefficients.

This project demonstrates how to build, preprocess, train, evaluate, and visualize a Polynomial Regression model using **Python** and **Scikit-learn** on the **Position Salaries Dataset**. The project introduces **PolynomialFeatures**, which automatically generates higher-degree features, enabling the model to fit complex real-world relationships more accurately than a simple linear model.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Regression?
- What is Polynomial Regression?
- Why Use Polynomial Regression?
- Polynomial Regression Equation
- Difference Between Linear and Polynomial Regression
- Assumptions of Polynomial Regression
- Project Workflow
- Dataset Information
- Technologies Used
- Project Structure
- Step-by-Step Notebook Explanation (Cells 1–10)

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

Instead of writing fixed rules for every possible situation, Machine Learning algorithms learn from historical data and improve their performance as they encounter more examples.

Machine Learning is mainly divided into three categories:

- **Supervised Learning**
- **Unsupervised Learning**
- **Reinforcement Learning**

Since the **Position Salaries Dataset** contains labeled data (Position Level and Salary), **Polynomial Regression** belongs to the **Supervised Learning** category.

---

# 📊 What is Regression?

Regression is a supervised Machine Learning technique used for predicting **continuous numerical values**.

Unlike classification algorithms, which predict categories, regression algorithms estimate numerical outputs such as:

- Salary Prediction
- House Price Prediction
- Sales Forecasting
- Population Growth
- Stock Price Prediction
- Demand Forecasting

The objective of regression is to learn the mathematical relationship between input variables and the target variable.

---

# 📈 What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression that models **non-linear relationships** between variables.

Instead of fitting only a straight line, Polynomial Regression transforms the original feature into polynomial terms such as:

- X²
- X³
- X⁴
- ...

These transformed features enable the regression model to fit a smooth curve that better represents complex data.

Although the resulting graph is curved, the underlying algorithm remains **Linear Regression** because it learns linear coefficients for the transformed features.

### In this project

**Independent Variable (Feature):**

- Position Level

**Dependent Variable (Target):**

- Salary

The model learns the relationship between an employee's position level and salary and predicts salaries for unseen position levels.

---

# 🎯 Why Use Polynomial Regression?

Polynomial Regression is useful whenever a straight line cannot accurately represent the relationship between variables.

It is commonly used for:

- Salary Prediction
- Population Growth Analysis
- Sales Forecasting
- Financial Trend Analysis
- Engineering Design
- Scientific Curve Fitting
- Medical Research
- Business Forecasting

Whenever the relationship between variables appears curved rather than linear, Polynomial Regression generally provides better predictions than Linear Regression.

---

# 📐 Polynomial Regression Equation

The mathematical equation of Polynomial Regression is:

```text
Ŷ = b₀ + b₁X + b₂X² + b₃X³ + ... + bₙXⁿ
```

### Where

| Symbol | Description |
|---------|-------------|
| Ŷ | Predicted Value |
| X | Independent Variable |
| X², X³ ... Xⁿ | Polynomial Features |
| b₀ | Intercept |
| b₁ ... bₙ | Regression Coefficients |

Each additional polynomial term allows the model to capture increasingly complex relationships between the input feature and the target variable.

---

# 🔄 Difference Between Linear Regression and Polynomial Regression

| Feature | Linear Regression | Polynomial Regression |
|----------|-------------------|-----------------------|
| Relationship | Linear | Non-linear |
| Prediction Curve | Straight Line | Curved Line |
| Feature Transformation | Not Required | Required |
| Handles Curved Data | No | Yes |
| Complexity | Low | Moderate |

Polynomial Regression should only be used when the dataset exhibits a non-linear relationship.

---

# 📋 Assumptions of Polynomial Regression

For Polynomial Regression to perform effectively, the following assumptions should generally hold:

- The relationship follows a polynomial pattern.
- Observations are independent.
- Errors have constant variance (Homoscedasticity).
- Residuals are approximately normally distributed.
- The polynomial degree is selected appropriately.
- Excessively high polynomial degrees should be avoided to prevent overfitting.

Selecting an appropriate polynomial degree is one of the most important factors affecting model performance.

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
Separate Features & Target
        │
        ▼
Generate Polynomial Features
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
Visualize Results
```

---

# 📂 Dataset Information

**Dataset Name:** Position_Salaries.csv

The dataset contains different employee positions, their numerical level, and the corresponding salaries.

## Dataset Summary

| Property | Value |
|----------|-------|
| Number of Rows | 10 |
| Number of Columns | 3 |
| Missing Values | No |
| Numerical Features | 1 |
| Text Features | 1 |
| Target Variable | Salary |

---

## Features

| Feature | Description |
|----------|-------------|
| Position | Employee Designation |
| Level | Position Level |
| Salary | Employee Salary |

The **Position** column is descriptive and is not used during model training.

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
Polynomial-Regression/
│
├── Polynomial Regression.ipynb
├── Position_Salaries.csv
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

from sklearn.preprocessing import PolynomialFeatures
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
| PolynomialFeatures | Generate polynomial features |
| LinearRegression | Train regression model |
| MSE & R² | Evaluate model performance |

---

## 🔹 Cell 2 – Load Dataset

```python
dataset = pd.read_csv("Position_Salaries.csv")
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

- Rows: **10**
- Columns: **3**

---

## 🔹 Cell 7 – Separate Features and Target

```python
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
```

The dataset is divided into:

- **X (Independent Variable):** Position Level
- **y (Dependent Variable):** Salary

The Position column is ignored because it is descriptive text and is not required for model training.

---

## 🔹 Cell 8 – Display Features and Target

```python
print(X)

print(y)
```

This step verifies that the feature matrix (`X`) and target variable (`y`) have been separated correctly before generating polynomial features.

Displaying the extracted values also helps ensure that the dataset has been prepared correctly for model training.

---

## 🔹 Cell 9 – Generate Polynomial Features

```python
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
```

`PolynomialFeatures` automatically transforms the original feature into higher-degree polynomial terms.

For a polynomial degree of **4**, the transformed dataset includes:

- Bias Term (1)
- Level
- Level²
- Level³
- Level⁴

These additional features enable the regression model to learn non-linear relationships that cannot be represented by a straight line.

---

## 🔹 Cell 10 – Display Polynomial Features

```python
print(X_poly)
```

After generating the polynomial features, this step displays the transformed feature matrix.

Displaying `X_poly` helps verify that the polynomial transformation has been successfully applied.

Each original value is now represented by multiple polynomial terms, which are later used by the Linear Regression model to learn a smooth non-linear relationship between **Position Level** and **Salary**.

---

## 🔹 Cell 11 – Display Polynomial Feature Names

```python
print(poly_reg.get_feature_names_out())
```

This step displays the names of all polynomial features generated by the `PolynomialFeatures` transformer.

For a polynomial degree of **4**, the generated features include:

- 1 (Bias Term)
- Level
- Level²
- Level³
- Level⁴

Displaying these feature names helps understand how the original feature has been transformed before training the model.

---

## 🔹 Cell 12 – Train the Polynomial Regression Model

```python
model = LinearRegression()
model.fit(X_poly, y)
```

After generating the polynomial features, the **Linear Regression** algorithm is trained using the transformed dataset.

Although the prediction curve is non-linear, the algorithm used is still **Linear Regression** because it learns a linear combination of the polynomial features.

During training, the model learns the coefficients that best represent the relationship between **Position Level** and **Salary**.

---

## 🔹 Cell 13 – Predict Training Data

```python
y_pred = model.predict(X_poly)
```

Once the model has been trained, it predicts the salary values for all position levels present in the dataset.

The predicted values are stored in **`y_pred`**, which will later be used to evaluate the model's performance and compare the predicted salaries with the actual salaries.

---

## 🔹 Cell 14 – Compare Actual and Predicted Values

```python
comparison = pd.DataFrame({
    "Actual Salary": y,
    "Predicted Salary": y_pred
})

comparison
```

This step creates a comparison table containing the actual salary values and the salaries predicted by the Polynomial Regression model.

Comparing both values provides a clear understanding of how accurately the model fits the training data.

A smaller difference between the actual and predicted values indicates better model performance.

---

## 🔹 Cell 15 – Evaluate the Model

```python
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

Two important evaluation metrics are calculated to measure the model's performance.

### Mean Squared Error (MSE)

Mean Squared Error measures the average squared difference between the actual values and the predicted values.

- Lower MSE indicates better predictions.
- An MSE value close to **0** represents a highly accurate model.

### R² Score

The R² Score measures how well the regression model explains the variation in the target variable.

Its value ranges from **0 to 1**.

- **1** → Perfect prediction
- **0** → Poor prediction

A higher R² Score indicates that the Polynomial Regression model fits the dataset effectively.

---

## 🔹 Cell 16 – Visualize the Polynomial Regression Curve

```python
plt.scatter(X, y, color='red')
plt.plot(X, model.predict(X_poly), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
```

Visualization is one of the most important steps in any Machine Learning project.

In this graph:

- **Red points** represent the actual salary values.
- **Blue curve** represents the Polynomial Regression model.

Unlike Linear Regression, which produces a straight line, Polynomial Regression creates a smooth curve that follows the trend of the data more accurately.

---

## 🔹 Cell 17 – Predict Salary for a New Position Level

```python
model.predict(poly_reg.transform([[6.5]]))
```

This step predicts the salary for a new employee whose **Position Level is 6.5**.

Since the model was trained using polynomial features, the new input is first transformed using `poly_reg.transform()` before prediction.

This demonstrates how the trained model can make predictions for unseen data.

---

## 🔹 Cell 18 – Display Intercept and Coefficients

```python
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

Finally, the learned model parameters are displayed.

### Intercept

The intercept represents the predicted salary when all polynomial feature values are zero.

### Coefficients

The coefficients indicate the contribution of each polynomial feature to the final prediction.

Together, the intercept and coefficients define the complete Polynomial Regression equation learned from the dataset.

---
# 🔄 Difference Between Simple, Multiple, and Polynomial Regression

Although all three algorithms belong to the **Linear Regression** family and are supervised Machine Learning algorithms used for predicting continuous numerical values, they differ in the number of input features, the type of relationship they model, and the preprocessing required before training.

| Feature | Simple Linear Regression | Multiple Linear Regression | Polynomial Regression |
|----------|--------------------------|----------------------------|----------------------|
| Number of Independent Variables | One | Two or More | Usually One (transformed into polynomial features) |
| Relationship | Linear | Linear | Non-linear |
| Prediction Shape | Straight Line | Hyperplane | Curved Line |
| Feature Transformation Required | No | No | Yes |
| Handles Multiple Features | No | Yes | No (in this project) |
| Handles Non-linear Data | No | No | Yes |
| Algorithm Used | Linear Regression | Linear Regression | Linear Regression |
| Data Complexity | Low | Medium | High |
| Risk of Overfitting | Low | Low | Higher (for large polynomial degrees) |
| Typical Applications | Simple Predictions | Multi-factor Predictions | Curved Trend Prediction |

## 📌 Simple Linear Regression

Simple Linear Regression is used when there is **only one independent variable** and **one dependent variable**. It assumes a **linear relationship** between the variables and fits the best possible straight line through the data.

### Example

Predicting a student's marks based only on the number of study hours.

---

## 📌 Multiple Linear Regression

Multiple Linear Regression is an extension of Simple Linear Regression where **two or more independent variables** are used to predict a single dependent variable.

It is suitable when the target variable depends on multiple factors.

### Example

Predicting house prices using:

- Area
- Number of Bedrooms
- Age of House
- Distance from City

---

## 📌 Polynomial Regression

Polynomial Regression is another extension of Linear Regression that models **non-linear relationships**.

Instead of adding more input features, it transforms the existing feature into higher-degree polynomial terms such as:

- X²
- X³
- X⁴
- ...

These transformed features allow the model to fit a smooth curve instead of a straight line.

### Example

Predicting employee salary based on position level, where salary increases non-linearly with higher positions.

---

## 🎯 Key Differences

- **Simple Linear Regression** uses **one independent variable** and fits a straight line.
- **Multiple Linear Regression** uses **multiple independent variables** but still fits a linear relationship.
- **Polynomial Regression** generally uses the same independent variable but transforms it into polynomial features to capture non-linear relationships.
- Simple and Multiple Linear Regression cannot accurately model curved data, whereas Polynomial Regression is specifically designed for such datasets.
- All three algorithms ultimately use the **Linear Regression algorithm** for training. The main difference lies in the number of input features and whether polynomial feature transformation is applied before training.

- # 📊 Model Evaluation Metrics

After training the Polynomial Regression model, it is important to evaluate how accurately it predicts the target values.

In this project, two widely used regression evaluation metrics are used:

- Mean Squared Error (MSE)
- R² Score (Coefficient of Determination)

These metrics help measure the prediction accuracy and overall performance of the model.

---

## 📉 Mean Squared Error (MSE)

Mean Squared Error measures the average squared difference between the actual values and the predicted values.

### Formula

```text
MSE = (1/n) Σ (Yi − Ŷi)²
```

### Where

| Symbol | Description |
|---------|-------------|
| Yi | Actual Value |
| Ŷi | Predicted Value |
| n | Number of Observations |

### Interpretation

- Lower MSE indicates better model performance.
- MSE equal to **0** means perfect prediction.
- Larger values indicate higher prediction errors.

Since errors are squared before averaging, larger errors receive greater penalties than smaller ones.

---

## 📈 R² Score (Coefficient of Determination)

The R² Score measures how well the regression model explains the variation in the dependent variable.

### Formula

```text
R² = 1 − (SSres / SStotal)
```

### Where

| Symbol | Description |
|---------|-------------|
| SSres | Sum of Squared Residuals |
| SStotal | Total Sum of Squares |

### Interpretation

| R² Score | Meaning |
|-----------|---------|
| 1.0 | Perfect Prediction |
| 0.9 – 1.0 | Excellent Model |
| 0.7 – 0.9 | Good Model |
| 0.5 – 0.7 | Moderate Model |
| Below 0.5 | Poor Model |

A higher R² Score indicates that the model explains a greater proportion of the variance in the target variable.

---

# 🎯 Choosing the Polynomial Degree

One of the most important decisions in Polynomial Regression is selecting the appropriate polynomial degree.

The degree determines the complexity of the regression curve.

| Degree | Model Behaviour |
|----------|----------------|
| 1 | Equivalent to Linear Regression |
| 2 | Captures simple curves |
| 3 | Models more complex relationships |
| 4 | Captures highly curved patterns |
| Very High Degree | May overfit the data |

In this project, a **degree of 4** is selected because it provides a good balance between flexibility and prediction accuracy for the Position Salaries dataset.

---

# ⚖️ Underfitting vs Overfitting

Selecting an inappropriate polynomial degree may lead to poor model performance.

## Underfitting

Underfitting occurs when the model is too simple to capture the underlying relationship between the variables.

Characteristics:

- High prediction error
- Poor accuracy
- Misses important patterns
- Degree too low

---

## Overfitting

Overfitting occurs when the polynomial degree is excessively high.

Characteristics:

- Fits training data extremely well
- Performs poorly on unseen data
- Learns noise instead of actual patterns
- Poor generalization

A balanced polynomial degree helps avoid both underfitting and overfitting.

---

# 🌍 Real-World Applications

Polynomial Regression is widely used in various industries where relationships between variables are non-linear.

Some common applications include:

- Salary Prediction
- Stock Market Analysis
- Population Growth Prediction
- Engineering Curve Fitting
- Medical Research
- Manufacturing Process Optimization
- Demand Forecasting
- Weather Forecasting
- Scientific Data Analysis
- Economic Trend Analysis

---

# ✅ Advantages of Polynomial Regression

- Models complex non-linear relationships.
- More flexible than Linear Regression.
- Easy to implement using Scikit-learn.
- Provides smooth prediction curves.
- Can significantly improve prediction accuracy.
- Works well for small and medium-sized datasets.
- Maintains the simplicity of Linear Regression while increasing flexibility.

---

# ❌ Limitations of Polynomial Regression

- Sensitive to outliers.
- Choosing the wrong polynomial degree may reduce performance.
- High-degree polynomials can lead to overfitting.
- Computational complexity increases with higher degrees.
- Difficult to interpret compared to Linear Regression.
- Performance may degrade for very large datasets with many polynomial features.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Polynomial-Regression.git
```

Navigate to the project directory:

```bash
cd Polynomial-Regression
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Polynomial Regression.ipynb
```

---

# 📦 Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Install all dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand the concept of Polynomial Regression.
- Differentiate between Simple, Multiple, and Polynomial Regression.
- Load and explore datasets using Pandas.
- Generate polynomial features using Scikit-learn.
- Train a Polynomial Regression model.
- Make predictions on unseen data.
- Evaluate regression models using MSE and R² Score.
- Visualize non-linear regression curves.
- Understand underfitting and overfitting.
- Apply Polynomial Regression to real-world prediction problems.

---

# 📝 Conclusion

Polynomial Regression is a powerful extension of Linear Regression that enables Machine Learning models to capture non-linear relationships between variables.

In this project, the Position Salaries dataset was analyzed, polynomial features were generated, a Polynomial Regression model was trained, predictions were made, evaluation metrics were calculated, and the regression curve was visualized.

The project demonstrates that transforming input features into polynomial terms allows Linear Regression to model complex patterns that cannot be represented using a simple straight line.

Polynomial Regression is particularly useful for datasets where the relationship between variables follows a curved trend, making it an important technique for predictive analytics and real-world Machine Learning applications.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project was developed as part of a Machine Learning learning journey to understand regression algorithms and their practical implementation using Python and Scikit-learn.
