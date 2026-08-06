# 🌲 Random Forest Regression

A comprehensive Machine Learning project demonstrating **Random Forest Regression** using the **Position_Salaries** dataset. This project covers the complete machine learning workflow, including data preprocessing, model training, prediction, performance evaluation, regression metrics, feature importance analysis, residual error analysis, and high-resolution regression curve visualization.

The notebook is designed for beginners as well as intermediate learners who want to understand how **Random Forest Regression** combines multiple Decision Trees to improve prediction accuracy and reduce overfitting through ensemble learning.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Regression?
- What is Random Forest Regression?
- Why Use Random Forest Regression?
- Ensemble Learning
- Bagging (Bootstrap Aggregating)
- Bootstrap Sampling
- Random Feature Selection
- Decision Trees in Random Forest
- Averaging Predictions
- Feature Importance
- Advantages of Random Forest Regression
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually writing rules, Machine Learning algorithms automatically identify relationships within the data and use those learned patterns to make predictions for unseen observations.

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
- Stock Market Prediction

Machine Learning algorithms are mainly classified into:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Random Forest Regression belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 📈 What is Regression?

Regression is a supervised learning technique used to predict **continuous numerical values**.

Unlike classification, where the output belongs to predefined categories, regression predicts numerical quantities.

Examples include:

- House Price Prediction
- Salary Prediction
- Sales Forecasting
- Insurance Cost Prediction
- Temperature Prediction
- Stock Price Prediction

In this project, the model predicts the **salary corresponding to a given position level**.

---

# 🌲 What is Random Forest Regression?

Random Forest Regression is an **ensemble learning algorithm** that combines multiple Decision Trees to predict continuous numerical values.

Instead of relying on a single Decision Tree, Random Forest builds many trees using different subsets of the training data and combines their predictions by calculating the average.

This ensemble approach improves prediction accuracy while reducing overfitting.

Random Forest is one of the most widely used regression algorithms because it is robust, accurate, and capable of modeling complex non-linear relationships.

---

# ⭐ Why Use Random Forest Regression?

Random Forest Regression is popular because it combines the strengths of many Decision Trees.

Some important advantages include:

- High prediction accuracy.
- Reduces overfitting compared to a single Decision Tree.
- Captures complex non-linear relationships.
- Works well with numerical and categorical data.
- Requires minimal preprocessing.
- Provides feature importance scores.
- Handles large datasets efficiently.

---

# 🌐 Ensemble Learning

Ensemble Learning is a Machine Learning technique where multiple models are combined to improve prediction performance.

Instead of relying on a single model, several models work together to produce more accurate and reliable predictions.

The three major ensemble techniques are:

- Bagging
- Boosting
- Stacking

Random Forest belongs to the **Bagging** family.

---

# 🌳 Bagging (Bootstrap Aggregating)

Bagging stands for **Bootstrap Aggregating**.

It is an ensemble learning technique where multiple Decision Trees are trained independently using different random subsets of the training dataset.

The predictions from all trees are then averaged to produce the final regression output.

Bagging reduces model variance and improves prediction stability.

---

# 📦 Bootstrap Sampling

Bootstrap Sampling is the process of creating multiple training datasets by randomly selecting observations **with replacement** from the original dataset.

Each Decision Tree receives a different bootstrap sample.

As a result:

- Some observations appear multiple times.
- Some observations are not selected.

This diversity among training datasets improves the overall robustness of the Random Forest.

---

# 🎯 Random Feature Selection

Random Forest introduces randomness not only in the training samples but also in the feature selection process.

At every split, only a random subset of features is considered.

Benefits include:

- Increased diversity among trees.
- Reduced correlation between trees.
- Better generalization.
- Lower risk of overfitting.

---

# 🌲 Decision Trees in Random Forest

Each tree within the Random Forest is an independent Decision Tree.

Every tree:

- Is trained on a different bootstrap sample.
- Uses random feature selection.
- Produces its own prediction.

The forest combines all these predictions to generate the final output.

---

# 📊 Averaging Predictions

For regression problems, the predictions from all Decision Trees are averaged.

### Formula

```text
Final Prediction =
(Prediction₁ + Prediction₂ + ... + Predictionₙ) / Number of Trees
```

Averaging reduces the influence of individual trees and produces more stable predictions.

---

# 📈 Feature Importance

Random Forest calculates **Feature Importance Scores** based on how much each feature contributes to reducing prediction error.

Features with higher importance values have a greater influence on the final prediction.

Feature importance helps identify the most significant variables affecting the target variable.

---

# ✅ Advantages of Random Forest Regression

- High prediction accuracy.
- Reduces overfitting.
- Handles non-linear relationships.
- Robust against noisy data.
- Works well with high-dimensional datasets.
- Provides feature importance.
- Minimal preprocessing required.
- Stable predictions due to averaging.

---

# 📌 Assumptions of Random Forest Regression

Random Forest Regression makes relatively few assumptions.

Better performance is achieved when:

- Training data is representative.
- Features contain meaningful information.
- A sufficient number of Decision Trees are used.
- Hyperparameters are properly tuned.
- Noise is reasonably limited.

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
Calculate MAE
        │
        ▼
Calculate MSE
        │
        ▼
Calculate RMSE
        │
        ▼
Calculate R² Score
        │
        ▼
Predict New Salary
        │
        ▼
Visualize Regression Curve
        │
        ▼
Display Hyperparameters
        │
        ▼
Feature Importance
        │
        ▼
Residual Error Plot
```
# 📂 Dataset Information

**Dataset Name:** Position_Salaries.csv

The dataset contains different job positions, their corresponding position levels, and salaries.

The objective of this project is to build a **Random Forest Regression** model that predicts the salary of an employee based on their position level.

## Dataset Summary

| Property | Value |
|----------|-------|
| Number of Rows | 10 |
| Number of Columns | 3 |
| Missing Values | No |
| Numerical Features | 2 |
| Categorical Features | 1 |
| Target Variable | Salary |

---

## Features

| Feature | Description |
|----------|-------------|
| Position | Job designation |
| Level | Position level (Independent Variable) |
| Salary | Annual salary (Target Variable) |

In this project, the **Position** column is removed because it is categorical and does not directly contribute to salary prediction.

The Random Forest Regression model is trained using:

- **Feature (X):** Level
- **Target (y):** Salary

The model predicts the salary corresponding to a given position level.

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
Random-Forest-Regression/
│
├── Random Forest Regression.ipynb
├── Position_Salaries.csv
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
from sklearn.ensemble import RandomForestRegressor
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
| RandomForestRegressor | Build the Random Forest Regression model |

---

## 🔹 Cell 2 – Load the Dataset

```python
df = pd.read_csv("Position_Salaries.csv")
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

Understanding the dataset structure is an essential preprocessing step before training the regression model.

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

These statistics help understand the distribution of the numerical variables.

---

## 🔹 Cell 5 – Check Missing Values

```python
df.isnull().sum()
```

Before training the model, the dataset is checked for missing values.

This step verifies that every feature contains complete information.

The Position Salaries dataset contains **no missing values**, making it suitable for training without additional preprocessing.

---

## 🔹 Cell 6 – Dataset Shape

```python
df.shape
```

The shape function returns the dimensions of the dataset.

For this dataset:

- Number of Rows: **10**
- Number of Columns: **3**

This provides a quick overview of the available training data.

---

## 🔹 Cell 7 – Feature and Target Selection

```python
if "Position" in df.columns:
    df = df.drop("Position", axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

The **Position** column is removed because it is a categorical identifier and is not directly used for regression.

The remaining data is divided into:

**Feature (X):**

- Level

**Target (y):**

- Salary

Separating the independent variable from the target variable prepares the dataset for model training.

---

## 🔹 Cell 8 – Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The dataset is divided into:

- **Training Set (80%)** – Used to train the Random Forest Regression model.
- **Testing Set (20%)** – Used to evaluate the model on unseen observations.

Using separate datasets helps measure how well the trained model generalizes to new data.

---

## 🔹 Cell 9 – Train the Random Forest Regressor

```python
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
```

The Random Forest Regression model is trained using the training dataset.

Instead of building a single Decision Tree, the algorithm constructs multiple Decision Trees using different bootstrap samples of the training data.

The final prediction is obtained by averaging the predictions from all the trees, resulting in improved accuracy and reduced overfitting.

---

## 🔹 Cell 10 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained Random Forest Regression model predicts salaries for the testing dataset.

The predicted salary values are stored in **`y_pred`**, which are later compared with the actual salaries to evaluate the model's performance using regression metrics.
---

## 🔹 Cell 11 – Compare Actual and Predicted Values

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.head()
```

A comparison table is created containing:

- Actual Salary
- Predicted Salary

This comparison provides a quick overview of how closely the Random Forest Regression model predicts the actual salary values.

---

## 🔹 Cell 12 – Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
```

Mean Absolute Error (MAE) measures the average absolute difference between the actual and predicted salary values.

### Formula

```text
MAE = (1/n) Σ |Actual − Predicted|
```

Lower MAE values indicate better prediction accuracy.

---

## 🔹 Cell 13 – Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

Mean Squared Error (MSE) measures the average squared difference between actual and predicted salaries.

### Formula

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

Since errors are squared, larger prediction errors receive greater penalties.

Lower MSE values indicate better model performance.

---

## 🔹 Cell 14 – Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

Root Mean Squared Error (RMSE) is the square root of the Mean Squared Error.

### Formula

```text
RMSE = √MSE
```

RMSE is expressed in the same unit as the target variable (salary), making it easier to interpret than MSE.

Smaller RMSE values indicate better predictive performance.

---

## 🔹 Cell 15 – R² Score

```python
r2 = r2_score(y_test, y_pred)
```

The **R² Score (Coefficient of Determination)** measures how well the Random Forest Regression model explains the variation in salary.

### Interpretation

- **1.0** → Perfect Prediction
- **0.0** → No Predictive Ability
- **Negative** → Model performs worse than predicting the mean

Higher R² values indicate better model performance.

---

## 🔹 Cell 16 – Predict a New Position Salary

```python
new_level = pd.DataFrame({
    "Level":[6.5]
})

predicted_salary = model.predict(new_level)
```

The trained Random Forest Regression model predicts the salary for a new employee with a **Position Level of 6.5**.

This demonstrates how the trained model can be applied to estimate salaries for unseen position levels.

---

## 🔹 Cell 17 – High-Resolution Random Forest Regression Curve

```python
X_grid = np.arange(
    min(X.values),
    max(X.values),
    0.01
)
```

A high-resolution regression curve is generated using closely spaced input values.

Since Random Forest Regression averages predictions from multiple Decision Trees, the resulting curve is generally smoother and more stable than that of a single Decision Tree Regression model.

This visualization illustrates how the ensemble model captures the relationship between position level and salary.

---

## 🔹 Cell 18 – Random Forest Hyperparameters

```python
model.get_params()
```

The final notebook cell displays the hyperparameters used to train the Random Forest Regression model.

Some important hyperparameters include:

- **n_estimators** – Number of Decision Trees in the forest.
- **criterion** – Function used to measure split quality.
- **max_depth** – Maximum depth of each Decision Tree.
- **max_features** – Number of features considered for each split.
- **random_state** – Ensures reproducible model training.

Understanding these hyperparameters helps optimize model performance.

---

## 🔹 Cell 19 – Feature Importance

```python
model.feature_importances_
```

Random Forest Regression calculates **Feature Importance Scores**, indicating how much each feature contributes to salary prediction.

This notebook displays:

- A table of feature importance values.
- A bar chart comparing feature importance.

Higher importance scores indicate greater influence on the model's predictions.

---

## 🔹 Cell 20 – Residual Error Plot

```python
residuals = y_test - y_pred
```

A Residual Error Plot is created to analyze the prediction errors made by the Random Forest Regression model.

Residuals represent the difference between the actual salary and the predicted salary.

### Formula

```text
Residual = Actual Salary − Predicted Salary
```

A good regression model produces residuals that are randomly distributed around zero without any clear pattern.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Random Forest Regression model.

These include:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Together, these metrics provide a comprehensive evaluation of the regression model.

---

## 📈 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

### Formula

```text
MAE = (1/n) Σ |Actual − Predicted|
```

### Interpretation

- Lower MAE indicates better prediction accuracy.
- Easy to understand because it is measured in the same unit as the target variable.

---

## 📈 Mean Squared Error (MSE)

MSE measures the average squared prediction error.

### Formula

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

### Interpretation

- Larger prediction errors receive greater penalties.
- Lower MSE indicates better model performance.

---

## 📈 Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

### Formula

```text
RMSE = √MSE
```

### Interpretation

- Expressed in the same unit as salary.
- Easier to interpret than MSE.
- Lower RMSE indicates more accurate predictions.

---

## 📈 R² Score

The R² Score measures the proportion of variance explained by the regression model.

### Formula

```text
R² = 1 − (Residual Sum of Squares / Total Sum of Squares)
```

### Interpretation

| R² Score | Performance |
|-----------|-------------|
| 1.0 | Perfect Fit |
| 0.9 – 1.0 | Excellent |
| 0.8 – 0.9 | Very Good |
| 0.7 – 0.8 | Good |
| Below 0.5 | Poor |

---

# 🌍 Real-World Applications

Random Forest Regression is widely used in many real-world applications, including:

- House Price Prediction
- Salary Prediction
- Insurance Cost Estimation
- Sales Forecasting
- Stock Price Analysis
- Demand Forecasting
- Energy Consumption Prediction
- Agricultural Yield Prediction
- Financial Forecasting
- Business Analytics

---

# ✅ Advantages of Random Forest Regression

- High prediction accuracy.
- Reduces overfitting through ensemble learning.
- Handles non-linear relationships effectively.
- Robust against noisy data and outliers.
- Provides feature importance scores.
- Works well with large datasets.
- Requires minimal data preprocessing.

---

# ❌ Limitations of Random Forest Regression

- Training can be computationally expensive.
- Uses more memory than a single Decision Tree.
- Less interpretable than individual Decision Trees.
- May be slower for real-time predictions with very large forests.
- Hyperparameter tuning may be required for optimal performance.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Random-Forest-Regression.git
```

Navigate to the project directory:

```bash
cd Random-Forest-Regression
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
Random Forest Regression.ipynb
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

- Understand Random Forest Regression and Ensemble Learning.
- Learn the concepts of Bagging and Bootstrap Sampling.
- Understand Random Feature Selection.
- Build a Random Forest Regression model using Scikit-learn.
- Evaluate regression models using MAE, MSE, RMSE, and R² Score.
- Interpret Feature Importance scores.
- Analyze model performance using Residual Error Plots.
- Apply Random Forest Regression to real-world prediction problems.

---

# 📝 Conclusion

Random Forest Regression is a powerful ensemble learning algorithm that improves prediction accuracy by combining the outputs of multiple Decision Trees. Through Bagging, Bootstrap Sampling, and Random Feature Selection, it reduces overfitting while effectively modeling complex non-linear relationships.

In this project, the Position Salaries dataset was explored, preprocessed, used to train a Random Forest Regression model, evaluated using multiple regression metrics, and analyzed through feature importance and residual error visualization.

This project provides a strong foundation for understanding ensemble regression techniques and prepares learners for advanced algorithms such as **Gradient Boosting Regression**, **AdaBoost Regression**, and **XGBoost Regression**.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
