# 🚀 Support Vector Regression (SVR)

A comprehensive Machine Learning project demonstrating **Support Vector Regression (SVR)** using the **Position_Salaries** dataset. This project covers the complete machine learning workflow, including data preprocessing, feature scaling, model training, prediction, performance evaluation, regression metrics, support vector analysis, residual error analysis, and high-resolution regression curve visualization.

The notebook is designed for beginners as well as intermediate learners who want to understand how **Support Vector Regression (SVR)** predicts continuous values by finding an optimal hyperplane while keeping prediction errors within a predefined margin using kernel functions.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Regression?
- What is Support Vector Regression (SVR)?
- Why Use SVR?
- Support Vector Machines (SVM)
- Hyperplane
- Margin
- Support Vectors
- Epsilon (ε)-Insensitive Tube
- Kernel Trick
- Linear Kernel
- Polynomial Kernel
- RBF (Gaussian) Kernel
- Regularization Parameter (C)
- Gamma
- Advantages of SVR
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually defining rules, Machine Learning algorithms automatically identify relationships within the data and use those learned relationships to make predictions on unseen observations.

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

Support Vector Regression belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 📈 What is Regression?

Regression is a supervised learning technique used to predict **continuous numerical values**.

Unlike classification, where the output belongs to predefined categories, regression predicts numerical quantities.

Examples include:

- House Price Prediction
- Salary Prediction
- Stock Price Prediction
- Sales Forecasting
- Temperature Prediction
- Insurance Cost Prediction

In this project, the model predicts the **salary corresponding to a given position level**.

---

# 🚀 What is Support Vector Regression (SVR)?

Support Vector Regression (SVR) is a supervised Machine Learning algorithm derived from **Support Vector Machines (SVM)**.

Instead of separating classes like Support Vector Classification (SVC), SVR predicts continuous numerical values by finding the best-fitting function that keeps prediction errors within a predefined margin called the **ε (epsilon)-insensitive tube**.

Rather than minimizing the error for every training point, SVR focuses on maintaining predictions within this acceptable error margin while keeping the regression function as simple as possible.

This makes SVR particularly effective for complex non-linear regression problems.

---

# ⭐ Why Use SVR?

Support Vector Regression is widely used because of its ability to model complex relationships while maintaining strong generalization performance.

Some important advantages include:

- Handles non-linear relationships effectively.
- Produces highly accurate predictions.
- Uses only important observations (Support Vectors).
- Resistant to overfitting.
- Supports multiple kernel functions.
- Performs well on small and medium-sized datasets.

---

# 📚 Support Vector Machines (SVM)

Support Vector Machines are supervised learning algorithms designed to find an optimal decision boundary.

For regression problems, this concept is extended into **Support Vector Regression (SVR)**.

Instead of maximizing class separation, SVR finds the optimal regression function while allowing small prediction errors inside a predefined margin.

---

# 📏 Hyperplane

A **Hyperplane** is the mathematical function used by SVR to represent the relationship between input features and the target variable.

The objective is to identify the hyperplane that best fits the data while maintaining the maximum possible margin.

Kernel functions allow this hyperplane to model both linear and non-linear relationships.

---

# 📐 Margin

The **Margin** represents the distance around the regression function where prediction errors are considered acceptable.

Instead of forcing every observation to lie exactly on the regression curve, SVR allows small deviations inside this margin.

A larger margin generally improves the model's ability to generalize to unseen data.

---

# 🎯 Support Vectors

Support Vectors are the observations that lie closest to the boundary of the epsilon-insensitive tube.

These observations are the most influential points during model training.

Unlike many regression algorithms, SVR relies primarily on these support vectors rather than the entire dataset.

---

# 📦 Epsilon (ε)-Insensitive Tube

One of the defining features of SVR is the **ε (epsilon)-insensitive tube**.

Predictions falling within this tube are considered acceptable and do not contribute to the loss function.

Only observations lying outside the tube influence the optimization process.

This makes SVR less sensitive to small prediction errors.

---

# 🧠 Kernel Trick

The **Kernel Trick** allows SVR to solve complex non-linear regression problems without explicitly transforming the data into higher-dimensional space.

Instead, kernel functions compute similarities between observations efficiently.

Common kernels include:

- Linear Kernel
- Polynomial Kernel
- Radial Basis Function (RBF) Kernel

---

# 📈 Linear Kernel

The **Linear Kernel** is used when the relationship between the independent and dependent variables is approximately linear.

It is computationally efficient and performs well on simple datasets.

---

# 📉 Polynomial Kernel

The **Polynomial Kernel** models curved relationships by introducing polynomial transformations of the input features.

It is suitable for moderately complex regression problems.

---

# 🌐 RBF (Gaussian) Kernel

The **Radial Basis Function (RBF) Kernel** is the most commonly used kernel for SVR.

It effectively models highly non-linear relationships by measuring the similarity between observations.

In this project, the **RBF Kernel** is used because the Position Salaries dataset exhibits a non-linear relationship between position level and salary.

---

# ⚙️ Regularization Parameter (C)

The **Regularization Parameter (C)** controls the balance between model complexity and prediction accuracy.

- Small **C**
  - Larger margin
  - Better generalization
  - Simpler model

- Large **C**
  - Smaller margin
  - Fits training data more closely
  - Higher risk of overfitting

Selecting an appropriate value of **C** is important for optimal model performance.

---

# 🎛️ Gamma

**Gamma** controls how much influence each training observation has on the regression function.

- Small Gamma
  - Smooth regression curve
  - Better generalization

- Large Gamma
  - More complex regression curve
  - Higher risk of overfitting

Proper tuning of Gamma significantly affects SVR performance.

---

# ✅ Advantages of Support Vector Regression

- High prediction accuracy.
- Handles complex non-linear relationships.
- Uses only support vectors for learning.
- Robust against overfitting.
- Flexible through multiple kernel functions.
- Strong generalization capability.
- Effective for small and medium-sized datasets.

---

# 📌 Assumptions of Support Vector Regression

Although SVR is highly flexible, better performance is achieved when:

- Training data is representative.
- Features are properly scaled.
- Kernel function is selected appropriately.
- Hyperparameters are tuned carefully.
- Noise is reasonably limited.

Feature Scaling is **mandatory** for SVR because the algorithm is highly sensitive to feature magnitudes.

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
Feature Scaling
        │
        ▼
Train-Test Split
        │
        ▼
Train SVR Model
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
Visualize SVR Curve
        │
        ▼
Display Support Vectors
        │
        ▼
Residual Error Plot
```
# 📂 Dataset Information

**Dataset Name:** Position_Salaries.csv

The dataset contains different job positions, their corresponding position levels, and salaries.

The objective of this project is to build a **Support Vector Regression (SVR)** model that predicts the salary of an employee based on their position level.

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

The Support Vector Regression model is trained using:

- **Feature (X):** Level
- **Target (y):** Salary

Since SVR is sensitive to the scale of the data, **both the feature and the target variable are standardized using StandardScaler before training**.

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
Support-Vector-Regression/
│
├── Support Vector Regression.ipynb
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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
| StandardScaler | Standardize features and target variable |
| SVR | Build the Support Vector Regression model |

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

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

The **Position** column is removed because it is a categorical identifier and is not directly used for regression.

The remaining data is divided into:

**Feature (X):**

- Level

**Target (y):**

- Salary

These variables are then prepared for feature scaling and model training.

---

## 🔹 Cell 8 – Feature Scaling

```python
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)

y = y.reshape(-1,1)
y = sc_y.fit_transform(y)
y = y.ravel()
```

Support Vector Regression is highly sensitive to the scale of the data.

Therefore, both the independent variable and the target variable are standardized using **StandardScaler**.

Standardization transforms the data so that:

- Mean = 0
- Standard Deviation = 1

This improves model convergence and prediction performance.

---

## 🔹 Cell 9 – Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The dataset is divided into:

- **Training Set (80%)** – Used to train the SVR model.
- **Testing Set (20%)** – Used to evaluate the model on unseen observations.

Using separate datasets helps measure how well the trained model generalizes to new data.

---

## 🔹 Cell 10 – Train the Support Vector Regression Model

```python
model = SVR(kernel="rbf")

model.fit(X_train, y_train)
```

The Support Vector Regression model is trained using the **Radial Basis Function (RBF) Kernel**.

The RBF kernel enables the model to learn complex non-linear relationships between position level and salary.

Unlike Linear Regression, SVR attempts to fit the best regression function while keeping prediction errors within the predefined epsilon-insensitive margin.
---

## 🔹 Cell 11 – Predict the Test Data

```python
y_pred_scaled = model.predict(X_test)
```

After training the Support Vector Regression model, predictions are generated for the testing dataset.

Since both the feature and target variables were standardized before training, the predicted values are initially obtained in their scaled form.

These scaled predictions are later converted back to the original salary scale using the inverse transformation of the target scaler.

---

## 🔹 Cell 12 – Compare Actual and Predicted Values

```python
comparison = pd.DataFrame({
    "Actual": y_actual.flatten(),
    "Predicted": y_pred.flatten()
})

comparison.head()
```

A comparison table is created containing:

- Actual Salary
- Predicted Salary

This comparison provides a quick overview of how closely the SVR model predicts salary values for unseen observations.

---

## 🔹 Cell 13 – Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_actual, y_pred)
```

Mean Absolute Error (MAE) measures the average absolute difference between the actual and predicted salary values.

### Formula

```text
MAE = (1/n) Σ |Actual − Predicted|
```

Lower MAE values indicate better prediction accuracy.

---

## 🔹 Cell 14 – Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_actual, y_pred)
```

Mean Squared Error (MSE) measures the average squared difference between the actual and predicted salaries.

### Formula

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

Larger prediction errors receive greater penalties because the errors are squared.

Lower MSE values indicate better model performance.

---

## 🔹 Cell 15 – Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mse)
```

Root Mean Squared Error (RMSE) is the square root of the Mean Squared Error.

### Formula

```text
RMSE = √MSE
```

RMSE is expressed in the same unit as the target variable (salary), making it easier to interpret than MSE.

Smaller RMSE values indicate better predictive performance.

---

## 🔹 Cell 16 – R² Score

```python
r2 = r2_score(y_actual, y_pred)
```

The **R² Score (Coefficient of Determination)** measures how well the SVR model explains the variation in salary values.

### Interpretation

- **1.0** → Perfect Prediction
- **0.0** → No Predictive Ability
- **Negative** → Model performs worse than predicting the mean

Higher R² values indicate better model performance.

---

## 🔹 Cell 17 – Predict a New Position Salary

```python
new_level = np.array([[6.5]])

predicted_salary = model.predict(...)
```

The trained Support Vector Regression model predicts the salary for a new employee whose **Position Level is 6.5**.

Before prediction:

- The new input is standardized using the previously fitted **StandardScaler**.
- The predicted value is converted back to the original salary scale using the inverse transformation.

This demonstrates how the trained model can be applied to real-world salary prediction.

---

## 🔹 Cell 18 – High-Resolution SVR Regression Curve

```python
X_grid = np.arange(...)
```

A high-resolution regression curve is generated using closely spaced position levels.

Unlike Decision Tree or Random Forest Regression, Support Vector Regression produces a **smooth continuous curve** because it learns a continuous mathematical function.

This visualization clearly demonstrates how the RBF kernel captures the non-linear relationship between position level and salary.

---

## 🔹 Cell 19 – Model Hyperparameters and Support Vectors

```python
model.get_params()

model.support_vectors_
```

This notebook displays the hyperparameters used to train the Support Vector Regression model.

Some important hyperparameters include:

- **kernel** – Specifies the kernel function.
- **C** – Controls regularization.
- **gamma** – Determines the influence of individual observations.
- **epsilon** – Defines the width of the epsilon-insensitive tube.

The notebook also displays the **Support Vectors**, which are the most influential observations used by the SVR model during training.

Unlike many regression algorithms, SVR relies primarily on these support vectors instead of the entire dataset.

---

## 🔹 Cell 20 – Residual Error Plot

```python
residuals = y_actual.flatten() - y_pred.flatten()
```

A Residual Error Plot is created to evaluate the prediction errors produced by the SVR model.

Residuals represent the difference between the actual salary and the predicted salary.

### Formula

```text
Residual = Actual Salary − Predicted Salary
```

A well-performing regression model produces residuals that are randomly distributed around zero without forming any systematic pattern.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the Support Vector Regression model.

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
- Easy to understand because it uses the same unit as the target variable.

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

Support Vector Regression is widely used in many real-world applications, including:

- House Price Prediction
- Salary Prediction
- Stock Market Forecasting
- Sales Forecasting
- Energy Consumption Prediction
- Weather Forecasting
- Financial Forecasting
- Medical Cost Estimation
- Demand Forecasting
- Engineering Performance Prediction

---

# ✅ Advantages of Support Vector Regression

- High prediction accuracy.
- Handles complex non-linear relationships.
- Uses support vectors, making the model robust.
- Strong generalization capability.
- Supports multiple kernel functions.
- Effective on small and medium-sized datasets.
- Resistant to overfitting with proper parameter tuning.

---

# ❌ Limitations of Support Vector Regression

- Requires feature scaling.
- Choosing the appropriate kernel can be challenging.
- Hyperparameter tuning is computationally expensive.
- Training becomes slower on very large datasets.
- Less interpretable than simple linear regression models.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Support-Vector-Regression.git
```

Navigate to the project directory:

```bash
cd Support-Vector-Regression
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
Support Vector Regression.ipynb
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

- Understand the working principles of Support Vector Regression.
- Learn the concepts of Hyperplanes, Margins, and Support Vectors.
- Understand the Epsilon-Insensitive Tube.
- Apply Feature Scaling for SVR.
- Build a Support Vector Regression model using Scikit-learn.
- Evaluate regression models using MAE, MSE, RMSE, and R² Score.
- Interpret Support Vectors and model hyperparameters.
- Apply SVR to real-world regression problems.

---

# 📝 Conclusion

Support Vector Regression (SVR) is a powerful supervised Machine Learning algorithm capable of modeling both linear and non-linear relationships through kernel functions. By focusing on support vectors and maintaining prediction errors within an epsilon-insensitive margin, SVR achieves strong generalization and high predictive accuracy.

In this project, the Position Salaries dataset was explored, standardized, used to train an SVR model with the RBF kernel, evaluated using multiple regression metrics, and analyzed through support vectors and residual error visualization.

This project provides a strong foundation for understanding kernel-based regression techniques and prepares learners for advanced regression algorithms such as **K-Nearest Neighbors Regression**, **Gradient Boosting Regression**, and **XGBoost Regression**.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
