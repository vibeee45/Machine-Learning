# 🤝 K-Nearest Neighbors (KNN) Regression

A comprehensive Machine Learning project demonstrating **K-Nearest Neighbors (KNN) Regression** using the **Medical Cost Personal (Insurance)** dataset. This project covers the complete machine learning workflow, including data preprocessing, categorical data encoding, feature scaling, model training, prediction, performance evaluation, regression metrics, and residual error analysis.

The notebook is designed for beginners as well as intermediate learners who want to understand how **KNN Regression** predicts continuous values by identifying the nearest neighboring observations and averaging their target values.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Regression?
- What is K-Nearest Neighbors (KNN) Regression?
- Why Use KNN Regression?
- Instance-Based Learning
- Lazy Learning
- Distance Metrics
- Euclidean Distance
- Manhattan Distance
- Minkowski Distance
- Choosing the Value of K
- Distance Weighting
- Feature Scaling
- Advantages of KNN Regression
- Assumptions
- Project Workflow

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually writing rules, Machine Learning algorithms automatically identify relationships within data and use those learned relationships to make predictions for unseen observations.

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

K-Nearest Neighbors Regression belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 📈 What is Regression?

Regression is a supervised learning technique used to predict **continuous numerical values**.

Unlike classification, where outputs belong to predefined categories, regression predicts numerical quantities.

Examples include:

- House Price Prediction
- Medical Insurance Cost Prediction
- Salary Prediction
- Sales Forecasting
- Stock Price Prediction
- Temperature Prediction

In this project, the model predicts the **medical insurance charges** based on an individual's demographic and health-related information.

---

# 🤝 What is K-Nearest Neighbors (KNN) Regression?

K-Nearest Neighbors (KNN) Regression is a supervised Machine Learning algorithm that predicts continuous values by analyzing the **K most similar data points** in the training dataset.

Instead of learning a mathematical equation during training, KNN stores the entire training dataset.

When a new observation is provided, the algorithm:

1. Calculates the distance between the new observation and all training observations.
2. Finds the **K nearest neighbors**.
3. Calculates the average of their target values.
4. Returns this average as the predicted value.

Because of this behavior, KNN is known as an **instance-based learning algorithm**.

---

# ⭐ Why Use KNN Regression?

KNN Regression is widely used because of its simplicity and ability to capture complex relationships without assuming a mathematical model.

Some important advantages include:

- Easy to understand.
- No assumptions about data distribution.
- Learns non-linear relationships.
- Effective for small and medium-sized datasets.
- Naturally adapts to local patterns.
- Simple implementation.

---

# 📚 Instance-Based Learning

KNN Regression belongs to a family of algorithms known as **Instance-Based Learning**.

Instead of creating a mathematical model during training, the algorithm stores all training examples.

Predictions are generated only when new observations are received.

The model relies directly on previously observed instances rather than learning explicit equations.

---

# 💤 Lazy Learning

KNN is called a **Lazy Learning Algorithm** because it performs almost no computation during training.

Instead of building a predictive model, it simply stores the training dataset.

Most of the computational work happens during prediction, where distances between observations are calculated.

---

# 📏 Distance Metrics

Distance metrics determine how similar two observations are.

The most commonly used distance metrics include:

- Euclidean Distance
- Manhattan Distance
- Minkowski Distance

The default distance metric used by Scikit-learn's KNN implementation is the **Minkowski Distance**, which becomes Euclidean Distance when **p = 2**.

---

# 📐 Euclidean Distance

Euclidean Distance measures the straight-line distance between two observations.

### Formula

```text
Distance = √[(x₁ − x₂)² + (y₁ − y₂)²]
```

It is the most commonly used distance metric for KNN Regression.

---

# 📊 Manhattan Distance

Manhattan Distance measures the total distance traveled along horizontal and vertical directions.

### Formula

```text
Distance = |x₁ − x₂| + |y₁ − y₂|
```

It is useful when movement is restricted to grid-like paths.

---

# 📈 Minkowski Distance

Minkowski Distance is a generalized distance metric.

### Formula

```text
Distance = (Σ |xᵢ − yᵢ|ᵖ)^(1/p)
```

Special cases include:

- **p = 1** → Manhattan Distance
- **p = 2** → Euclidean Distance

Scikit-learn uses Minkowski Distance by default.

---

# 🎯 Choosing the Value of K

The value of **K** determines how many neighboring observations are used during prediction.

- Small K
  - Captures local patterns.
  - More sensitive to noise.
  - Higher variance.

- Large K
  - Produces smoother predictions.
  - Less sensitive to noise.
  - Higher bias.

Selecting an appropriate value of **K** is important for achieving good prediction accuracy.

---

# ⚖️ Distance Weighting

In standard KNN Regression, each neighbor contributes equally to the prediction.

Distance-weighted KNN assigns greater importance to neighbors that are closer to the query point.

This often improves prediction accuracy because nearby observations usually have greater similarity.

---

# ⚙️ Feature Scaling

Feature Scaling is essential for KNN Regression because the algorithm relies entirely on distance calculations.

If one feature has much larger values than others, it can dominate the distance computation and reduce model accuracy.

In this project, **StandardScaler** is used to standardize all input features before training.

---

# ✅ Advantages of KNN Regression

- Simple and intuitive algorithm.
- No assumptions about data distribution.
- Learns complex non-linear relationships.
- Easy to implement.
- Naturally adapts to local data patterns.
- Effective for small and medium-sized datasets.
- No explicit model training required.

---

# 📌 Assumptions of KNN Regression

Although KNN is a non-parametric algorithm, better performance is achieved when:

- Features are properly scaled.
- Relevant features are selected.
- The value of K is chosen appropriately.
- Training data is representative.
- Noise and outliers are limited.

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
Encode Categorical Variables
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Train KNN Regression Model
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
Predict Insurance Charges
        │
        ▼
Display Hyperparameters
        │
        ▼
Residual Error Plot
```
# 📂 Dataset Information

**Dataset Name:** insurance.csv

The dataset contains demographic and health-related information about individuals along with their medical insurance charges.

The objective of this project is to build a **K-Nearest Neighbors (KNN) Regression** model that predicts the medical insurance cost based on an individual's personal and health characteristics.

## Dataset Summary

| Property | Value |
|----------|-------|
| Number of Rows | 1338 |
| Number of Columns | 7 |
| Missing Values | No |
| Numerical Features | 4 |
| Categorical Features | 3 |
| Target Variable | charges |

---

## Features

| Feature | Description |
|----------|-------------|
| age | Age of the individual |
| sex | Gender (Male/Female) |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status (Yes/No) |
| region | Residential region |
| charges | Annual medical insurance charges (Target Variable) |

The KNN Regression model is trained using:

**Features (X):**

- age
- sex
- bmi
- children
- smoker
- region

**Target (y):**

- charges

Since KNN is a distance-based algorithm, all categorical variables are encoded and numerical features are standardized before model training.

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
KNN-Regression/
│
├── KNN Regression.ipynb
├── insurance.csv
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
from sklearn.neighbors import KNeighborsRegressor
```

### Purpose

The required Python libraries are imported before beginning data preprocessing and model development.

| Library | Purpose |
|----------|----------|
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| LabelEncoder | Encode categorical variables |
| StandardScaler | Standardize numerical features |
| train_test_split | Split the dataset into training and testing sets |
| KNeighborsRegressor | Build the KNN Regression model |

---

## 🔹 Cell 2 – Load the Dataset

```python
df = pd.read_csv("insurance.csv")
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

The Insurance dataset contains **no missing values**, making it suitable for training without additional preprocessing.

---

## 🔹 Cell 6 – Dataset Shape

```python
df.shape
```

The shape function returns the dimensions of the dataset.

For this dataset:

- Number of Rows: **1338**
- Number of Columns: **7**

This provides a quick overview of the available training data.

---

## 🔹 Cell 7 – Feature and Target Selection

```python
X = df.drop("charges", axis=1)
y = df["charges"]
```

The dataset is divided into:

**Features (X):**

- age
- sex
- bmi
- children
- smoker
- region

**Target (y):**

- charges

Separating the independent variables from the target variable prepares the dataset for model training.

---

## 🔹 Cell 8 – Encode Categorical Variables

```python
encoder = LabelEncoder()

for column in ["sex", "smoker", "region"]:
    X[column] = encoder.fit_transform(X[column])
```

Machine Learning algorithms require numerical input.

The categorical variables are converted into numerical values using **Label Encoding**.

The encoded columns include:

- sex
- smoker
- region

This transformation enables the KNN Regression model to compute distances between observations effectively.

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

- **Training Set (80%)** – Used to train the KNN Regression model.
- **Testing Set (20%)** – Used to evaluate the model on unseen observations.

Using separate datasets helps measure how well the trained model generalizes to new data.

---

## 🔹 Cell 10 – Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Feature Scaling is an essential preprocessing step for KNN Regression because the algorithm relies entirely on distance calculations.

Standardization transforms the numerical features so that:

- Mean = 0
- Standard Deviation = 1

This prevents features with larger numerical values from dominating the distance computation and improves prediction accuracy.
---

## 🔹 Cell 11 – Train the KNN Regression Model

```python
model = KNeighborsRegressor(n_neighbors=5)

model.fit(X_train, y_train)
```

The K-Nearest Neighbors Regression model is trained using the training dataset.

Unlike most regression algorithms, KNN does not learn a mathematical equation during training.

Instead, it stores the training observations and predicts new values based on the nearest neighboring data points.

In this project, **K = 5**, meaning the model uses the five nearest neighbors to estimate medical insurance charges.

---

## 🔹 Cell 12 – Predict the Test Data

```python
y_pred = model.predict(X_test)
```

The trained KNN Regression model predicts the medical insurance charges for the testing dataset.

The predicted values are stored in **`y_pred`**, which are later compared with the actual insurance charges to evaluate model performance.

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

- Actual Insurance Charges
- Predicted Insurance Charges

This comparison provides a quick overview of how accurately the KNN Regression model predicts unseen observations.

---

## 🔹 Cell 14 – Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
```

Mean Absolute Error (MAE) measures the average absolute difference between the actual and predicted insurance charges.

### Formula

```text
MAE = (1/n) Σ |Actual − Predicted|
```

Lower MAE values indicate better prediction accuracy.

---

## 🔹 Cell 15 – Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

Mean Squared Error (MSE) measures the average squared difference between the actual and predicted insurance charges.

### Formula

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

Large prediction errors receive greater penalties because the differences are squared.

Lower MSE values indicate better model performance.

---

## 🔹 Cell 16 – Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mse)
```

Root Mean Squared Error (RMSE) is the square root of the Mean Squared Error.

### Formula

```text
RMSE = √MSE
```

RMSE is expressed in the same unit as the target variable (insurance charges), making it easier to interpret than MSE.

Smaller RMSE values indicate better predictive performance.

---

## 🔹 Cell 17 – R² Score

```python
r2 = r2_score(y_test, y_pred)
```

The **R² Score (Coefficient of Determination)** measures how well the KNN Regression model explains the variation in medical insurance charges.

### Interpretation

- **1.0** → Perfect Prediction
- **0.0** → No Predictive Ability
- **Negative** → Model performs worse than predicting the mean

Higher R² values indicate better model performance.

---

## 🔹 Cell 18 – Predict Insurance Cost for a New Customer

```python
new_customer = pd.DataFrame({
    ...
})

predicted_charge = model.predict(new_customer_scaled)
```

The trained KNN Regression model predicts the medical insurance charges for a new customer.

Before prediction:

- The categorical values are encoded.
- The input features are standardized using the previously fitted **StandardScaler**.
- The scaled data is passed to the trained KNN model.

This demonstrates how the trained model can be applied to estimate insurance costs for new individuals.

---

## 🔹 Cell 19 – Model Hyperparameters

```python
model.get_params()
```

The notebook displays the hyperparameters used to train the KNN Regression model.

Some important hyperparameters include:

- **n_neighbors** – Number of nearest neighbors used for prediction.
- **weights** – Specifies whether all neighbors contribute equally or are weighted by distance.
- **algorithm** – Algorithm used to compute nearest neighbors.
- **metric** – Distance metric used to measure similarity.
- **p** – Parameter controlling the Minkowski distance calculation.

Understanding these hyperparameters helps optimize the model for different regression problems.

---

## 🔹 Cell 20 – Residual Error Plot

```python
residuals = y_test - y_pred
```

A Residual Error Plot is created to analyze the prediction errors made by the KNN Regression model.

Residuals represent the difference between the actual insurance charges and the predicted charges.

### Formula

```text
Residual = Actual Charges − Predicted Charges
```

A well-performing regression model produces residuals that are randomly distributed around zero without any systematic pattern.

---

# 📊 Model Evaluation Metrics

Several evaluation metrics are used to assess the performance of the KNN Regression model.

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
- Easy to interpret because it uses the same unit as the target variable.

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

- Expressed in the same unit as insurance charges.
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

K-Nearest Neighbors Regression is widely used in many real-world applications, including:

- Medical Insurance Cost Prediction
- House Price Prediction
- Sales Forecasting
- Product Price Estimation
- Customer Spending Prediction
- Energy Consumption Forecasting
- Agricultural Yield Prediction
- Healthcare Cost Analysis
- Financial Forecasting
- Business Analytics

---

# ✅ Advantages of KNN Regression

- Simple and easy to understand.
- No assumptions about data distribution.
- Learns complex non-linear relationships.
- Effective for small and medium-sized datasets.
- Naturally adapts to local data patterns.
- No explicit model-building phase.
- Flexible and easy to implement.

---

# ❌ Limitations of KNN Regression

- Prediction becomes slower as the dataset grows.
- Requires feature scaling for accurate distance calculations.
- Sensitive to irrelevant features and noisy data.
- Choosing the optimal value of **K** can be challenging.
- Performance decreases in high-dimensional datasets (Curse of Dimensionality).

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/KNN-Regression.git
```

Navigate to the project directory:

```bash
cd KNN-Regression
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
KNN Regression.ipynb
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

- Understand the working principles of K-Nearest Neighbors Regression.
- Learn Instance-Based Learning and Lazy Learning.
- Understand Euclidean, Manhattan, and Minkowski distance metrics.
- Learn the importance of Feature Scaling in KNN.
- Build a KNN Regression model using Scikit-learn.
- Evaluate regression models using MAE, MSE, RMSE, and R² Score.
- Interpret model hyperparameters.
- Apply KNN Regression to real-world prediction problems.

---

# 📝 Conclusion

K-Nearest Neighbors (KNN) Regression is a simple yet effective supervised Machine Learning algorithm that predicts continuous values by analyzing the nearest neighboring observations. Unlike traditional regression models, KNN does not build an explicit mathematical model; instead, it relies on similarity between data points to make predictions.

In this project, the Medical Insurance dataset was explored, categorical features were encoded, numerical features were standardized, a KNN Regression model was trained, and its performance was evaluated using multiple regression metrics and residual error analysis.

This project provides a strong foundation for understanding instance-based learning and prepares learners for advanced ensemble regression algorithms such as **Gradient Boosting Regression**, **AdaBoost Regression**, and **XGBoost Regression**.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
