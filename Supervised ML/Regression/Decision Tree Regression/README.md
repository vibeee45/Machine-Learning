# 🌳 Decision Tree Regression

A comprehensive Machine Learning project demonstrating **Decision Tree Regression** using the **Position_Salaries** dataset. This project covers the complete machine learning workflow, including data preprocessing, model training, prediction, performance evaluation, regression metrics, feature importance analysis, tree visualization, and high-resolution regression curve visualization.

The notebook is designed for beginners as well as intermediate learners who want to understand how **Decision Tree Regression** predicts continuous values by recursively splitting the dataset into smaller subsets based on feature values.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Regression?
- What is Decision Tree Regression?
- Why Use Decision Tree Regression?
- Decision Tree Structure
- Root Node
- Internal Nodes
- Leaf Nodes
- Recursive Binary Splitting
- Variance Reduction
- Tree Depth
- Advantages of Decision Tree Regression
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually writing rules, Machine Learning algorithms automatically discover relationships within the data and use those learned patterns to make predictions on unseen observations.

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

Decision Tree Regression belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 📈 What is Regression?

Regression is a supervised learning technique used to predict **continuous numerical values**.

Unlike classification, where the output belongs to predefined categories, regression predicts numeric quantities.

Examples include:

- House Price Prediction
- Salary Prediction
- Stock Price Prediction
- Sales Forecasting
- Temperature Prediction
- Insurance Cost Prediction

In this project, the model predicts the **salary corresponding to a given position level**.

---

# 🌳 What is Decision Tree Regression?

Decision Tree Regression is a tree-based supervised Machine Learning algorithm used to predict continuous numerical values.

It works by repeatedly dividing the dataset into smaller subsets using decision rules that minimize prediction error.

Instead of fitting a straight curve like Linear Regression, Decision Tree Regression creates multiple regions and predicts the average value of observations within each region.

This makes it highly effective for learning **non-linear relationships**.

---

# ⭐ Why Use Decision Tree Regression?

Decision Tree Regression is popular because it can model complex relationships without requiring feature scaling or mathematical assumptions.

Some important advantages include:

- Captures non-linear patterns.
- Easy to understand and interpret.
- Requires minimal data preprocessing.
- Works well with small datasets.
- Handles numerical and categorical features.
- Provides feature importance.
- Can model complex decision boundaries.

---

# 🌲 Decision Tree Structure

A Decision Tree consists of several components:

- Root Node
- Internal Nodes
- Branches
- Leaf Nodes

The tree begins from the root node and continues splitting until stopping criteria are met.

Each split attempts to reduce prediction error.

---

# 🌱 Root Node

The **Root Node** is the topmost node of the Decision Tree.

It represents the entire dataset before any splitting occurs.

The algorithm selects the best feature and threshold to divide the data into more homogeneous groups.

---

# 🌿 Internal Nodes

Internal Nodes represent intermediate decision points.

Each internal node applies a rule such as:

```text
Position Level ≤ 6.5
```

Depending on whether the condition is true or false, observations move to different branches of the tree.

Multiple internal nodes allow the model to capture increasingly complex patterns.

---

# 🍃 Leaf Nodes

Leaf Nodes are the final nodes of the Decision Tree.

Unlike Decision Tree Classification, where leaf nodes contain class labels, Decision Tree Regression leaf nodes contain **predicted numerical values**.

Every observation reaching the same leaf receives the same predicted value.

---

# 🔄 Recursive Binary Splitting

Decision Tree Regression uses a process called **Recursive Binary Splitting**.

The algorithm repeatedly:

1. Evaluates every possible split.
2. Chooses the split that minimizes prediction error.
3. Divides the dataset into two groups.
4. Repeats the process for each new group.

This recursive procedure continues until predefined stopping conditions are reached.

---

# 📉 Variance Reduction

Decision Tree Regression selects splits that maximize **Variance Reduction**.

Variance measures how spread out the target values are within a node.

The objective is to create child nodes whose target values are as similar as possible.

Lower variance within a node results in more accurate predictions.

---

# 🌲 Tree Depth

Tree Depth refers to the maximum number of levels from the root node to the deepest leaf node.

- Small tree depth:
  - Simpler model
  - Lower risk of overfitting
  - May underfit

- Large tree depth:
  - More complex model
  - Captures detailed patterns
  - Higher risk of overfitting

Choosing an appropriate tree depth is essential for achieving good performance.

---

# ✅ Advantages of Decision Tree Regression

- Easy to visualize and interpret.
- Handles non-linear relationships.
- Requires little data preprocessing.
- No feature scaling required.
- Handles multiple input variables.
- Provides feature importance.
- Fast prediction after training.

---

# 📌 Assumptions of Decision Tree Regression

Decision Tree Regression makes relatively few assumptions compared to linear models.

Better performance is achieved when:

- Training data is representative.
- Features contain useful information.
- Outliers are limited.
- Tree depth is properly controlled.
- Sufficient training data is available.

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
Train Decision Tree Model
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
Visualize Decision Tree
        │
        ▼
Feature Importance
        │
        ▼
Display Hyperparameters
```
# 📂 Dataset Information

**Dataset Name:** Position_Salaries.csv

The dataset contains different job positions, their corresponding position levels, and salaries.

The objective of this project is to build a **Decision Tree Regression** model that predicts the salary of an employee based on their position level.

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

The Decision Tree Regression model is trained using:

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
Decision-Tree-Regression/
│
├── Decision Tree Regression.ipynb
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
from sklearn.tree import DecisionTreeRegressor
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
| DecisionTreeRegressor | Build the Decision Tree Regression model |

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

- **Training Set (80%)** – Used to train the Decision Tree Regression model.
- **Testing Set (20%)** – Used to evaluate the model on unseen observations.

Using separate datasets helps measure how well the trained model generalizes to new data.

---

## 🔹 Cell 9 – Train the Decision Tree Regressor

```python
model = DecisionTreeRegressor(
    random_state=42
)

model.fit(X_train, y_train)
```

The Decision Tree Regression model is trained using the training dataset.

During training, the algorithm recursively splits the data into smaller subsets based on the feature values.

Each split aims to reduce the variance of the target variable, resulting in more accurate salary predictions.

---

## 🔹 Cell 10 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained Decision Tree Regression model predicts salaries for the testing dataset.

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

This comparison helps evaluate how closely the predicted salaries match the actual salaries.

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

Large prediction errors receive greater penalties because the differences are squared.

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

RMSE is expressed in the same unit as the target variable, making it easier to interpret than MSE.

Smaller RMSE values indicate better predictive performance.

---

## 🔹 Cell 15 – R² Score

```python
r2 = r2_score(y_test, y_pred)
```

The **R² Score (Coefficient of Determination)** measures how well the regression model explains the variation in the target variable.

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

The trained Decision Tree Regression model predicts the salary for a new employee whose position level is **6.5**.

This demonstrates how the trained model can be used for real-world salary prediction.

---

## 🔹 Cell 17 – High-Resolution Decision Tree Regression Curve

```python
X_grid = np.arange(
    min(X.values),
    max(X.values),
    0.01
)
```

A high-resolution regression curve is generated using a dense sequence of feature values.

Unlike Linear Regression, Decision Tree Regression produces a **step-like prediction curve** because each region of the feature space receives a constant prediction.

This visualization clearly illustrates how Decision Tree Regression partitions the data into multiple intervals.

---

## 🔹 Cell 18 – Visualize the Decision Tree

```python
plot_tree(model)
```

The trained Decision Tree is visualized using Scikit-learn's **plot_tree()** function.

The visualization displays:

- Root Node
- Internal Nodes
- Leaf Nodes
- Decision Rules
- Predicted Values

This provides a clear understanding of how the model makes salary predictions.

---

## 🔹 Cell 19 – Decision Tree Hyperparameters

```python
model.get_params()
```

The final notebook cell displays the hyperparameters used to train the Decision Tree Regression model.

Some important hyperparameters include:

- **criterion** – Function used to measure split quality.
- **max_depth** – Maximum depth of the tree.
- **min_samples_split** – Minimum samples required to split a node.
- **min_samples_leaf** – Minimum samples required at a leaf node.
- **random_state** – Ensures reproducible model training.

Understanding these hyperparameters helps optimize the model for different regression problems.

---

## 🔹 Cell 20 – Feature Importance

```python
model.feature_importances_
```

Decision Tree Regression calculates **Feature Importance Scores**, indicating how much each input feature contributes to salary prediction.

This notebook displays:

- A table of feature importance values.
- A bar chart comparing feature importance.

Higher importance scores indicate greater influence on the model's predictions.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Decision Tree Regression model.

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
- Easy to interpret because it uses the same units as the target variable.

---

## 📈 Mean Squared Error (MSE)

MSE measures the average squared prediction error.

### Formula

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

### Interpretation

- Larger errors receive greater penalties.
- Lower MSE indicates better model performance.

---

## 📈 Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

### Formula

```text
RMSE = √MSE
```

### Interpretation

- Expressed in the same unit as the target variable.
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

Decision Tree Regression is widely used in many real-world applications, including:

- Salary Prediction
- House Price Prediction
- Sales Forecasting
- Demand Forecasting
- Insurance Cost Prediction
- Energy Consumption Prediction
- Agricultural Yield Prediction
- Medical Cost Estimation
- Financial Forecasting
- Business Analytics

---

# ✅ Advantages of Decision Tree Regression

- Easy to understand and visualize.
- Captures non-linear relationships.
- Requires little data preprocessing.
- No feature scaling required.
- Handles numerical and categorical data.
- Provides feature importance scores.
- Fast prediction after training.

---

# ❌ Limitations of Decision Tree Regression

- Can easily overfit training data.
- Sensitive to small changes in the dataset.
- Produces step-like predictions instead of smooth curves.
- Performance depends on tree depth.
- Often less accurate than ensemble regression methods.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Decision-Tree-Regression.git
```

Navigate to the project directory:

```bash
cd Decision-Tree-Regression
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
Decision Tree Regression.ipynb
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

- Understand Decision Tree Regression.
- Learn Recursive Binary Splitting and Variance Reduction.
- Understand Root Nodes, Internal Nodes, and Leaf Nodes.
- Build a Decision Tree Regression model using Scikit-learn.
- Evaluate regression models using MAE, MSE, RMSE, and R² Score.
- Visualize regression trees and prediction curves.
- Interpret Feature Importance scores.
- Apply Decision Tree Regression to real-world prediction problems.

---

# 📝 Conclusion

Decision Tree Regression is a powerful supervised Machine Learning algorithm that predicts continuous values by recursively splitting the dataset into smaller regions. It effectively models non-linear relationships without requiring feature scaling or complex mathematical assumptions.

In this project, the Position Salaries dataset was explored, preprocessed, used to train a Decision Tree Regression model, evaluated using multiple regression metrics, and analyzed through feature importance and tree visualization.

This project provides a strong foundation for understanding tree-based regression techniques and prepares learners for advanced ensemble regression algorithms such as **Random Forest Regression**, **Gradient Boosting Regression**, and **XGBoost Regression**.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
