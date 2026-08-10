# 🤝 AdaBoost Regression

A comprehensive Machine Learning project demonstrating **AdaBoost Regression** using the **Medical Cost Personal (Insurance)** dataset.

This project covers the complete Machine Learning workflow, including data preprocessing, categorical data encoding, train-test splitting, model training, prediction, performance evaluation, regression metrics, feature importance, and residual error analysis.

The notebook is designed for beginners as well as intermediate learners who want to understand how **AdaBoost Regression** predicts continuous values by combining multiple weak learners sequentially and focusing more attention on observations that are difficult to predict.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Regression?
- What is AdaBoost Regression?
- Why Use AdaBoost Regression?
- Ensemble Learning
- Boosting
- Weak Learners
- Sequential Learning
- Adaptive Boosting
- Base Estimator
- Number of Estimators
- Learning Rate
- Regression Loss
- Advantages of AdaBoost Regression
- Limitations of AdaBoost Regression
- Dataset Information
- Features
- Technologies Used
- Project Structure
- Project Workflow
- Step-by-Step Notebook Explanation

---

# 🤖 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from historical data and make predictions without being explicitly programmed.

Instead of manually defining rules for every possible situation, Machine Learning algorithms analyze existing data, identify useful patterns, and use those learned patterns to make predictions on new or unseen data.

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

AdaBoost Regression belongs to the **Supervised Learning** category because it learns from labeled training data.

---

# 📈 What is Regression?

Regression is a supervised Machine Learning technique used to predict **continuous numerical values**.

Unlike classification, where outputs belong to predefined categories, regression predicts numerical quantities.

Examples include:

- House Price Prediction
- Medical Insurance Cost Prediction
- Salary Prediction
- Sales Forecasting
- Stock Price Prediction
- Temperature Prediction
- Electricity Consumption Prediction

In this project, the model predicts **medical insurance charges** based on an individual's demographic and health-related information.

The target variable is:

```text
charges
```

---

# 🤝 What is AdaBoost Regression?

**AdaBoost**, short for **Adaptive Boosting**, is an ensemble Machine Learning algorithm that combines multiple weak learners to create a stronger predictive model.

For regression problems, AdaBoost uses a sequence of regression learners.

Instead of depending on a single Decision Tree, AdaBoost builds multiple weak learners sequentially.

After each learner makes predictions, the algorithm evaluates its errors and gives greater attention to observations that are difficult to predict.

The next learner then focuses more strongly on those difficult observations.

The process continues through multiple boosting stages, and the individual learners are combined to produce the final prediction.

The basic concept can be represented as:

```text
Training Data
      ↓
Weak Learner 1
      ↓
Calculate Errors
      ↓
Focus on Difficult Observations
      ↓
Weak Learner 2
      ↓
Calculate Errors
      ↓
Focus on Difficult Observations
      ↓
      ...
      ↓
Final Strong Model
```

---

# ⭐ Why Use AdaBoost Regression?

AdaBoost Regression is useful because it combines several relatively simple learners into a stronger ensemble model.

Instead of expecting one Decision Tree to learn the complete relationship in the dataset, AdaBoost builds multiple weak learners and combines their contributions.

Important characteristics include:

- Combines multiple weak learners.
- Learns sequentially.
- Focuses on difficult observations.
- Can capture non-linear relationships.
- Can improve prediction performance.
- Provides feature importance.
- Supports hyperparameter tuning.
- Works well with Decision Tree base estimators.

---

# 🌐 Ensemble Learning

**Ensemble Learning** is a Machine Learning technique where multiple models are combined to produce a stronger predictive model.

Instead of depending on a single model, ensemble methods combine the predictions or contributions of several models.

Common ensemble learning techniques include:

- Bagging
- Boosting
- Stacking

AdaBoost Regression belongs to the **Boosting** family of ensemble algorithms.

---

# 📈 Boosting

Boosting is an ensemble learning technique where multiple models are trained **sequentially**.

Unlike Bagging, where models are generally trained independently, Boosting builds models one after another.

Each new learner attempts to improve the predictions of the existing ensemble.

The general process is:

```text
Weak Learner 1
      ↓
Evaluate Predictions
      ↓
Identify Difficult Observations
      ↓
Weak Learner 2
      ↓
Improve Previous Predictions
      ↓
Weak Learner 3
      ↓
      ...
      ↓
Final Strong Model
```

AdaBoost is one of the important algorithms used to implement the Boosting approach.

---

# 🌱 Weak Learners

A **Weak Learner** is a relatively simple Machine Learning model that may not provide highly accurate predictions by itself.

The main idea behind Boosting is to combine multiple weak learners so that their combined prediction becomes stronger.

In this project, a:

```python
DecisionTreeRegressor
```

is used as the base weak learner.

The base model used in the notebook is:

```python
base_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)
```

The `max_depth=3` parameter limits the complexity of the Decision Tree.

Instead of creating one extremely complex tree, AdaBoost combines multiple relatively simple learners.

---

# 🔄 Sequential Learning

AdaBoost follows a **sequential learning process**.

The learners are not trained independently.

The general process is:

### Step 1

Train the first weak learner.

### Step 2

Generate predictions using the learner.

### Step 3

Identify observations where predictions are less accurate.

### Step 4

Give greater attention to difficult observations.

### Step 5

Train another weak learner.

### Step 6

Repeat the process for the specified number of estimators.

### Step 7

Combine the learners to generate the final prediction.

This allows the model to progressively improve its predictions.

---

# ⚖️ Adaptive Boosting

The word **Adaptive** in AdaBoost refers to the model's ability to adapt to errors made during previous boosting stages.

Observations that are difficult to predict receive greater attention during subsequent learning stages.

Conceptually:

```text
Original Training Data
          ↓
   Train Weak Learner
          ↓
   Evaluate Prediction
          ↓
Identify Difficult Cases
          ↓
Increase Their Importance
          ↓
   Train Next Learner
          ↓
       Repeat
          ↓
 Final AdaBoost Model
```

This adaptive process is the central idea behind AdaBoost.

---

# 🌳 Base Estimator

The **Base Estimator** is the individual Machine Learning model used by AdaBoost.

In this project, the base estimator is a:

```python
DecisionTreeRegressor
```

The exact base estimator used in the notebook is:

```python
base_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)
```

### Why use a Decision Tree?

Decision Trees are useful as base learners because:

- They can model non-linear relationships.
- They can capture feature interactions.
- They are relatively simple to train.
- They work effectively with boosting techniques.

The individual tree does not need to be extremely powerful because AdaBoost combines multiple learners.

---

# 🔢 Number of Estimators

The **`n_estimators`** parameter determines the number of boosting stages.

The model in this project uses:

```python
n_estimators=100
```

This means the AdaBoost model uses 100 boosting stages.

Each stage contributes another weak learner to the ensemble.

Increasing the number of estimators can allow the model to learn more complex patterns.

However, increasing the number of estimators can also:

- Increase training time.
- Increase model complexity.
- Potentially increase overfitting depending on the dataset and configuration.

---

# ⚙️ Learning Rate

The **Learning Rate** controls the contribution of each weak learner to the final model.

The model in this project uses:

```python
learning_rate=0.1
```

A smaller learning rate means that each individual learner contributes less to the final model.

A larger learning rate allows each learner to have a greater influence.

There is generally a trade-off between:

```text
Learning Rate
      ↕
Number of Estimators
```

A smaller learning rate may require more estimators to achieve strong performance.

---

# 📉 Regression Loss

AdaBoost Regression uses a loss function to determine how prediction errors affect the boosting process.

Common loss functions available for AdaBoost Regression include:

- Linear
- Square
- Exponential

The Scikit-learn `AdaBoostRegressor` uses **linear loss by default**.

The loss function determines how the algorithm responds to prediction errors during the boosting process.

---

# ⚙️ AdaBoost Model Configuration

The AdaBoost model used in this project is configured as:

```python
model = AdaBoostRegressor(
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
```

The configuration is:

| Parameter | Value | Description |
|----------|-------|-------------|
| `estimator` | Decision Tree | Base weak learner |
| `n_estimators` | 100 | Number of boosting stages |
| `learning_rate` | 0.1 | Contribution of each learner |
| `random_state` | 42 | Ensures reproducible results |

The model is trained using:

```python
model.fit(X_train, y_train)
```

During training, AdaBoost sequentially builds the ensemble of weak learners.

---

# ✅ Advantages of AdaBoost Regression

AdaBoost Regression provides several advantages:

- Combines multiple weak learners into a stronger model.
- Can capture non-linear relationships.
- Learns sequentially from previous errors.
- Can improve prediction performance.
- Provides feature importance.
- Relatively simple to implement.
- Supports hyperparameter tuning.
- Works effectively with Decision Tree base learners.

---

# ❌ Limitations of AdaBoost Regression

Despite its advantages, AdaBoost also has some limitations:

- Can be sensitive to noisy observations.
- Can be sensitive to outliers.
- Poor base learners can negatively affect performance.
- Sequential training can be slower than independently trained ensemble methods.
- Hyperparameter selection can significantly affect performance.
- Increasing the number of estimators unnecessarily can increase model complexity.

---

# 🔄 Project Workflow

The complete AdaBoost Regression project follows this workflow:

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
Check Dataset Information
        │
        ▼
Check Missing Values
        │
        ▼
Display Dataset Dimensions
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
Create Base Decision Tree
        │
        ▼
Create AdaBoost Regression Model
        │
        ▼
Train AdaBoost Model
        │
        ▼
Predict Test Data
        │
        ▼
Compare Actual & Predicted Values
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
Predict Insurance Cost
        │
        ▼
Display Hyperparameters
        │
        ▼
Feature Importance
        │
        ▼
Residual Error Analysis
```

---

# 📂 Dataset Information

**Dataset Name:** `insurance.csv`

The dataset contains demographic and health-related information about individuals along with their medical insurance charges.

The objective of this project is to build an **AdaBoost Regression** model that predicts medical insurance charges based on an individual's characteristics.

## Dataset Summary

| Property | Value |
|----------|-------|
| Number of Rows | 1338 |
| Number of Columns | 7 |
| Missing Values | No |
| Numerical Features | 4 |
| Categorical Features | 3 |
| Target Variable | charges |

The notebook confirms that the dataset contains **1338 entries and 7 columns**. All columns contain 1338 non-null values. :contentReference[oaicite:2]{index=2}

---

# 🧩 Dataset Features

| Feature | Description |
|----------|-------------|
| `age` | Age of the individual |
| `sex` | Gender of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of children |
| `smoker` | Smoking status |
| `region` | Residential region |
| `charges` | Medical insurance charges |

The first five rows of the notebook show these seven columns and their corresponding values. :contentReference[oaicite:3]{index=3}

---

# 🎯 Target Variable

The target variable is:

```text
charges
```

The model attempts to predict the medical insurance charges based on the other six columns.

The features and target are separated using:

```python
X = df.drop("charges", axis=1)
y = df["charges"]
```

After separation:

```text
Features Shape: (1338, 6)
Target Shape: (1338,)
```

The notebook therefore uses six input features and one target variable. :contentReference[oaicite:4]{index=4}

---

# 🛠 Technologies Used

The following technologies and libraries are used in this project:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

The notebook specifically imports NumPy, Pandas, Matplotlib, Seaborn, `train_test_split`, `LabelEncoder`, `AdaBoostRegressor`, and `DecisionTreeRegressor`. :contentReference[oaicite:5]{index=5}

---

# 📁 Project Structure

```text
AdaBoost-Regression/
│
├── AdaBoost Regression(1).ipynb
├── insurance.csv
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 📖 Step-by-Step Notebook Explanation

The following sections explain the actual cells used in the AdaBoost Regression notebook.

The code below follows the same order as the notebook.

---

## 🔹 Cell 1 – Importing Required Libraries

```python
# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

### Purpose

This is the first cell of the notebook.

It imports all the libraries required for:

- Data manipulation.
- Numerical calculations.
- Visualization.
- Data preprocessing.
- Train-test splitting.
- AdaBoost Regression.
- Decision Tree Regression.

### NumPy

```python
import numpy as np
```

NumPy is used for numerical operations.

Later in the notebook it is used to calculate the Root Mean Squared Error:

```python
rmse = np.sqrt(mse)
```

### Pandas

```python
import pandas as pd
```

Pandas is used for:

- Loading the CSV dataset.
- Creating DataFrames.
- Selecting features.
- Manipulating data.
- Creating comparison tables.
- Creating feature importance tables.

### Matplotlib

```python
import matplotlib.pyplot as plt
```

Matplotlib is used for creating visualizations later in the notebook.

### Seaborn

```python
import seaborn as sns
```

Seaborn is used for statistical and graphical visualization.

### Scikit-learn

The notebook imports:

```python
from sklearn.model_selection import train_test_split
```

This is used to divide the dataset into training and testing data.

```python
from sklearn.preprocessing import LabelEncoder
```

This is used to convert categorical values into numerical values.

```python
from sklearn.ensemble import AdaBoostRegressor
```

This is the main regression algorithm used in the project.

```python
from sklearn.tree import DecisionTreeRegressor
```

This is used as the base estimator for AdaBoost.

### Warning Handling

```python
import warnings
warnings.filterwarnings("ignore")
```

This prevents unnecessary warning messages from appearing throughout the notebook.

---

## 🔹 Cell 2 – Loading the Dataset

```python
# Loading the dataset

df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\insurance.csv"
)

# Displaying the first five rows

df.head()
```

### Purpose

This cell loads the Insurance dataset into a Pandas DataFrame.

The DataFrame is stored in the variable:

```python
df
```

The dataset is loaded using:

```python
pd.read_csv()
```

The notebook uses the local path:

```text
C:\Users\91958\Desktop\Machine Learning\datasets\insurance.csv
```

After loading the dataset, the first five rows are displayed using:

```python
df.head()
```

This provides an initial view of the dataset.

The first five records contain:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges`

For a GitHub repository where `insurance.csv` is placed beside the notebook, the path can be changed to:

```python
df = pd.read_csv("insurance.csv")
```

---

## 🔹 Cell 3 – Displaying Dataset Information

```python
# Displaying dataset information

df.info()
```

### Purpose

The `info()` function provides structural information about the dataset.

The notebook output shows:

```text
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns)
```

The columns are:

```text
age
sex
bmi
children
smoker
region
charges
```

The data types are:

```text
age        int64
sex        object
bmi        float64
children   int64
smoker     object
region     object
charges    float64
```

The output also confirms that all 1338 records are non-null. :contentReference[oaicite:6]{index=6}

This information is important because it identifies:

- Numerical columns.
- Categorical columns.
- Missing values.
- Dataset size.

---

## 🔹 Cell 4 – Statistical Summary

```python
# Statistical summary of numerical columns

df.describe()
```

### Purpose

The `describe()` function generates descriptive statistics for numerical columns.

The notebook calculates statistics for:

```text
age
bmi
children
charges
```

The output includes:

- Count
- Mean
- Standard deviation
- Minimum
- 25th percentile
- 50th percentile
- 75th percentile
- Maximum

For example, the dataset has:

```text
Age Mean       = 39.207025
BMI Mean       = 30.663397
Children Mean  = 1.094918
Charges Mean   = 13270.422265
```

The maximum insurance charge in the dataset is:

```text
63770.428010
```

These statistics provide an overview of the numerical distribution of the dataset. :contentReference[oaicite:7]{index=7}

---

## 🔹 Cell 5 – Checking Missing Values

```python
# Checking for missing values

df.isnull().sum()
```

### Purpose

Before training a Machine Learning model, it is important to check whether the dataset contains missing values.

The expression:

```python
df.isnull()
```

identifies missing values.

The `sum()` function counts them for every column.

The notebook output is:

```text
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
```

Therefore, the dataset contains **no missing values**. :contentReference[oaicite:8]{index=8}

No missing-value imputation is required before training the model.

---

## 🔹 Cell 6 – Displaying Dataset Dimensions

```python
# Displaying dataset dimensions

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
```

### Purpose

This cell displays the number of rows and columns in the dataset.

The notebook output is:

```text
Rows : 1338
Columns : 7
```

Therefore, the dataset contains:

- **1338 observations**
- **7 columns**

This provides a quick overview of the dataset size before preprocessing. :contentReference[oaicite:9]{index=9}

---

## 🔹 Cell 7 – Selecting Features and Target

```python
# Selecting independent and dependent variables

X = df.drop("charges", axis=1)
y = df["charges"]

print("Features:")
display(X.head())

print("\nTarget:")
display(y.head())

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
```

### Purpose

This cell separates the dataset into:

- Independent variables/features (`X`)
- Dependent variable/target (`y`)

The target column is:

```text
charges
```

Therefore, it is removed from `X`:

```python
X = df.drop("charges", axis=1)
```

The target is stored separately:

```python
y = df["charges"]
```

### Features

The six input features are:

```text
age
sex
bmi
children
smoker
region
```

### Target

The target variable is:

```text
charges
```

The notebook output confirms:

```text
Features Shape: (1338, 6)
Target Shape: (1338,)
```

This means there are 1338 observations with six input features and one target variable. :contentReference[oaicite:10]{index=10}

---

## 🔹 Cell 8 – Encoding Categorical Columns

```python
# Encoding categorical columns

encoder = LabelEncoder()

categorical_columns = [
    "sex",
    "smoker",
    "region"
]

for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])

print("Encoded Features:")
display(X.head())
```

### Purpose

The dataset contains categorical columns that contain text values.

The categorical columns selected in the notebook are:

```text
sex
smoker
region
```

These columns need to be converted into numerical representations before being passed to the regression model.

A `LabelEncoder` object is created:

```python
encoder = LabelEncoder()
```

The categorical columns are stored in:

```python
categorical_columns = [
    "sex",
    "smoker",
    "region"
]
```

A loop is then used to encode each column:

```python
for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])
```

After encoding, the first five rows contain numerical values for these categorical columns.

For example, the notebook output shows:

```text
sex
0
1

smoker
0
1

region
0
1
2
3
```

The encoded dataset can now be used by the AdaBoost Regression model. :contentReference[oaicite:11]{index=11}

---

## 🔹 Cell 9 – Splitting the Dataset

```python
# Splitting the dataset into Training and Testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Features :", X_train.shape)
print("Testing Features :", X_test.shape)

print("Training Labels :", y_train.shape)
print("Testing Labels :", y_test.shape)
```

### Purpose

The dataset is divided into training and testing subsets using:

```python
train_test_split()
```

The important parameter is:

```python
test_size=0.20
```

This means:

- 80% of observations are used for training.
- 20% of observations are used for testing.

The notebook output is:

```text
Training Features : (1070, 6)
Testing Features : (268, 6)

Training Labels : (1070,)
Testing Labels : (268,)
```

Therefore:

```text
Training Data = 1070 observations
Testing Data  = 268 observations
```

The parameter:

```python
random_state=42
```

ensures that the same random split can be reproduced.

The training dataset is used to train the AdaBoost model, while the testing dataset is kept separate for evaluating the model on unseen observations. :contentReference[oaicite:12]{index=12}

---

## 🔹 Cell 10 – Creating and Training the AdaBoost Model

```python
# Creating the base Decision Tree estimator

base_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

# Creating the AdaBoost Regression model

model = AdaBoostRegressor(
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Training the model

model.fit(X_train, y_train)

print("AdaBoost Regression model trained successfully!")
```

### Purpose

This is the main Machine Learning cell of the notebook.

First, a Decision Tree Regressor is created:

```python
base_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)
```

This Decision Tree acts as the **base weak learner**.

The tree has:

```python
max_depth=3
```

which limits the complexity of each base learner.

---

### Creating AdaBoost

The AdaBoost Regression model is created using:

```python
model = AdaBoostRegressor(
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
```

The important parameters are:

| Parameter | Value | Meaning |
|----------|-------|---------|
| `estimator` | `base_model` | Decision Tree used as the base learner |
| `n_estimators` | `100` | Number of boosting stages |
| `learning_rate` | `0.1` | Contribution of each learner |
| `random_state` | `42` | Reproducibility |

---

### Training the Model

The model is trained using:

```python
model.fit(X_train, y_train)
```

The AdaBoost model uses the training data to build its sequence of weak learners.

The notebook successfully outputs:

```text
AdaBoost Regression model trained successfully!
```

This completes the model training stage. :contentReference[oaicite:13]{index=13}

---

## 🔹 Cell 11 – Predicting the Test Data

```python
# Predicting the test data

y_pred = model.predict(X_test)

print("Predicted Insurance Charges:")
print(y_pred[:10])
```

### Purpose

After training, the model is used to predict insurance charges for the testing dataset.

The predictions are generated using:

```python
y_pred = model.predict(X_test)
```

The predictions are stored in:

```text
y_pred
```

The notebook displays the first ten predicted values:

```text
12628.34722126
 6950.53483301
25914.05244861
12628.34722126
36562.88559407
 7877.76490675
 5281.67115178
15738.48674580
 6724.74037692
13601.40260483
```

These predicted values will later be compared with the actual insurance charges from `y_test`. :contentReference[oaicite:14]{index=14}

---

---

## 🔹 Cell 12 – Comparing Actual and Predicted Values

```python
# Comparing Actual and Predicted Values

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.head(10)
```

### Purpose

After generating predictions using:

```python
y_pred = model.predict(X_test)
```

it is useful to compare the model's predictions with the actual values.

A new DataFrame is created:

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
```

The DataFrame contains two columns:

| Column | Meaning |
|--------|---------|
| `Actual` | Actual insurance charges |
| `Predicted` | Charges predicted by AdaBoost |

The first ten records from the notebook are:

| Actual | Predicted |
|--------:|----------:|
| 9095.06825 | 12628.347221 |
| 5272.17580 | 6950.534833 |
| 29330.98315 | 25914.052449 |
| 9301.89355 | 12628.347221 |
| 33750.29180 | 36562.885594 |
| 4536.25900 | 7877.764907 |
| 2117.33885 | 5281.671152 |
| 14210.53595 | 15738.486746 |
| 3732.62510 | 6724.740377 |
| 10264.44210 | 13601.402605 |

This gives a quick visual understanding of the difference between actual and predicted insurance charges. :contentReference[oaicite:1]{index=1}

---

# 📏 Model Evaluation

Simply generating predictions is not enough.

We need to measure how well the AdaBoost Regression model performs.

The notebook uses four common regression metrics:

```text
MAE
MSE
RMSE
R² Score
```

These metrics measure prediction errors from different perspectives.

---

## 🔹 Cell 13 – Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error

# Calculating Mean Absolute Error

mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
```

### Output

```text
Mean Absolute Error (MAE): 4141.76
```

The notebook calculates an MAE of:

```text
4141.76
```

:contentReference[oaicite:2]{index=2}

### What is MAE?

**Mean Absolute Error** measures the average absolute difference between actual and predicted values.

Formula:

```text
MAE = (1/n) Σ |Actual − Predicted|
```

In simple words:

> MAE tells us, on average, how far the predictions are from the actual values.

For example, an MAE of approximately:

```text
4141.76
```

means that the model's predictions differ from the actual insurance charges by roughly **4141.76 units on average**.

### Interpretation

```text
Lower MAE → Better
Higher MAE → Worse
```

MAE is easy to interpret because it is expressed in the same unit as the target variable.

---

## 🔹 Cell 14 – Mean Squared Error (MSE)

```python
from sklearn.metrics import mean_squared_error

# Calculating Mean Squared Error

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
```

### Output

```text
Mean Squared Error (MSE): 25889199.17
```

The model produces an MSE of:

```text
25,889,199.17
```

:contentReference[oaicite:3]{index=3}

### What is MSE?

**Mean Squared Error** calculates the average squared difference between actual and predicted values.

Formula:

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

The important difference between MSE and MAE is that MSE **squares the errors**.

Therefore, large errors receive a much greater penalty.

For example:

```text
Error = 10
Squared Error = 100
```

while:

```text
Error = 100
Squared Error = 10,000
```

### Interpretation

```text
Lower MSE → Better
Higher MSE → Worse
```

MSE is particularly useful when we want larger prediction errors to have a stronger impact on the evaluation.

---

## 🔹 Cell 15 – Root Mean Squared Error (RMSE)

```python
# Calculating Root Mean Squared Error

rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```

### Output

```text
Root Mean Squared Error (RMSE): 5088.14
```

The notebook obtains an RMSE of:

```text
5088.14
```

:contentReference[oaicite:4]{index=4}

### What is RMSE?

**Root Mean Squared Error** is the square root of MSE.

Formula:

```text
RMSE = √MSE
```

Since:

```text
MSE = 25,889,199.17
```

the square root gives approximately:

```text
RMSE = 5088.14
```

### Why use RMSE?

MSE is expressed in squared units, which makes it difficult to interpret directly.

RMSE converts the value back into the same unit as the target variable.

Therefore, RMSE is easier to understand.

### Interpretation

```text
Lower RMSE → Better
Higher RMSE → Worse
```

RMSE also gives greater importance to large prediction errors because it is calculated from squared errors.

---

## 🔹 Cell 16 – R² Score

```python
from sklearn.metrics import r2_score

# Calculating R² Score

r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
```

### Output

```text
R² Score: 0.8332
```

The model achieves an:

```text
R² Score = 0.8332
```

:contentReference[oaicite:5]{index=5}

### What is R² Score?

**R² Score**, or the **Coefficient of Determination**, measures how well the regression model explains the variation in the target variable.

The formula is:

```text
R² = 1 − (Residual Sum of Squares / Total Sum of Squares)
```

### Interpretation

Generally:

| R² Score | Interpretation |
|----------|----------------|
| 1.0 | Perfect prediction |
| Close to 1 | Very strong model |
| 0.0 | No improvement over predicting the mean |
| Negative | Worse than the mean prediction |

The notebook's result:

```text
0.8332
```

indicates that the model explains a substantial portion of the variation in the insurance charges.

---

# 📊 Final Model Performance

The four metrics obtained from the notebook are:

| Metric | Result |
|--------|-------:|
| MAE | 4141.76 |
| MSE | 25,889,199.17 |
| RMSE | 5088.14 |
| R² Score | 0.8332 |

These metrics provide a more complete picture of model performance than relying on a single metric.

---

## 🔹 Cell 17 – Predicting Insurance Cost for a New Customer

```python
# Predicting insurance cost for a new customer

new_customer = pd.DataFrame({
    "age": [35],
    "sex": [1],         # Male
    "bmi": [28.5],
    "children": [2],
    "smoker": [0],      # Non-Smoker
    "region": [2]
})

# Predicting insurance charges

predicted_charge = model.predict(new_customer)

print(f"Predicted Insurance Charge: ₹{predicted_charge[0]:,.2f}")
```

### Output

```text
Predicted Insurance Charge: ₹8,873.78
```

:contentReference[oaicite:6]{index=6}

### Purpose

This cell demonstrates how the trained AdaBoost model can be used to make a prediction for a completely new customer.

The new customer's information is:

| Feature | Value |
|---------|------:|
| Age | 35 |
| Sex | 1 |
| BMI | 28.5 |
| Children | 2 |
| Smoker | 0 |
| Region | 2 |

The values for the categorical variables are already encoded because the model was trained using encoded data.

The new customer is stored in:

```python
new_customer
```

The model then generates a prediction:

```python
predicted_charge = model.predict(new_customer)
```

The resulting predicted insurance charge is:

```text
₹8,873.78
```

This demonstrates the practical use of the trained regression model for predicting insurance costs for new observations.

---

## 🔹 Cell 18 – Displaying AdaBoost Hyperparameters

```python
# Displaying AdaBoost Hyperparameters

parameters = pd.DataFrame({
    "Hyperparameter": model.get_params().keys(),
    "Value": model.get_params().values()
})

display(parameters)
```

### Purpose

Machine Learning models contain **hyperparameters** that control how the model behaves.

The notebook uses:

```python
model.get_params()
```

to retrieve the parameters of the trained AdaBoost model.

These parameters are then converted into a Pandas DataFrame:

```python
parameters = pd.DataFrame({
    "Hyperparameter": model.get_params().keys(),
    "Value": model.get_params().values()
})
```

This makes the model configuration easier to inspect.

The important parameters shown in the notebook include:

| Hyperparameter | Value |
|----------------|-------|
| `estimator__max_depth` | 3 |
| `estimator__criterion` | squared_error |
| `estimator__random_state` | 42 |
| `learning_rate` | 0.1 |
| `loss` | linear |
| `n_estimators` | 100 |
| `random_state` | 42 |

The complete notebook output contains the AdaBoost parameters as well as the parameters of its underlying Decision Tree estimator. :contentReference[oaicite:7]{index=7}

---

# ⚙️ Important AdaBoost Hyperparameters

## `n_estimators`

```python
n_estimators = 100
```

Controls the number of boosting stages.

More estimators allow the ensemble to contain more weak learners.

---

## `learning_rate`

```python
learning_rate = 0.1
```

Controls how strongly each weak learner contributes to the final model.

---

## `estimator`

```python
DecisionTreeRegressor(max_depth=3)
```

Defines the base learner used by AdaBoost.

---

## `max_depth`

```python
max_depth = 3
```

Controls the maximum depth of each Decision Tree.

A smaller depth keeps the base learner relatively simple.

---

## `loss`

The notebook reports:

```text
linear
```

as the AdaBoost regression loss.

---

## `random_state`

```python
random_state = 42
```

Helps make the results reproducible.

---

# 🔹 Cell 19 – Displaying Feature Importance

```python
# Displaying Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

display(importance)

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("AdaBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.grid(axis="x")

plt.show()
```

### Purpose

Feature importance tells us how useful each feature was to the trained AdaBoost model.

The model provides feature importance through:

```python
model.feature_importances_
```

A DataFrame is created containing:

```text
Feature
Importance
```

The values are then sorted from highest to lowest:

```python
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
```

This makes it easier to identify the most influential features.

---

# 📊 Feature Importance Results

The notebook produces the following feature importance values:

| Rank | Feature | Importance |
|-----:|---------|-----------:|
| 1 | smoker | 0.661712 |
| 2 | bmi | 0.188042 |
| 3 | age | 0.132923 |
| 4 | children | 0.015614 |
| 5 | region | 0.001710 |
| 6 | sex | 0.000000 |

:contentReference[oaicite:8]{index=8}

The results show that:

```text
smoker
```

has the highest feature importance in this trained model.

Its importance score is approximately:

```text
0.661712
```

The next important features are:

```text
bmi  → 0.188042
age  → 0.132923
```

while:

```text
children → 0.015614
region   → 0.001710
sex      → 0.000000
```

have much smaller importance values in this particular trained model.

---

# 📊 Feature Importance Visualization

The notebook also creates a bar chart using Seaborn:

```python
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)
```

The chart title is:

```text
AdaBoost Feature Importance
```

The X-axis represents:

```text
Importance Score
```

and the Y-axis represents:

```text
Feature
```

The chart provides a visual representation of which features contribute most strongly to the trained AdaBoost model. :contentReference[oaicite:9]{index=9}

---

# 📌 Understanding the Feature Importance

Based on this particular trained model:

### 🥇 Smoker

```text
Importance = 0.661712
```

`smoker` has the highest importance score among the input features.

### 🥈 BMI

```text
Importance = 0.188042
```

BMI is the second most important feature.

### 🥉 Age

```text
Importance = 0.132923
```

Age is the third most important feature.

### Children

```text
Importance = 0.015614
```

Children has a relatively small importance score.

### Region

```text
Importance = 0.001710
```

Region contributes very little to the model according to the calculated importance.

### Sex

```text
Importance = 0.000000
```

Sex has zero feature importance in this trained model.

These values describe the importance calculated by **this specific trained AdaBoost model**; they should not automatically be interpreted as causal relationships.

---

# 🎯 Complete Model Summary

The AdaBoost Regression model in this project uses:

```text
Dataset:
Medical Insurance Dataset

Rows:
1338

Features:
6

Target:
charges

Training Data:
1070 rows

Testing Data:
268 rows

Base Estimator:
DecisionTreeRegressor

Maximum Tree Depth:
3

Number of Estimators:
100

Learning Rate:
0.1

Loss:
linear

Random State:
42
```

---

# 📈 Model Evaluation Summary

The trained model produces:

```text
MAE  = 4141.76
MSE  = 25889199.17
RMSE = 5088.14
R²   = 0.8332
```

These values show the performance of the AdaBoost Regression model on the test dataset used in the notebook. :contentReference[oaicite:10]{index=10}

---

# 🌍 Real-World Applications

AdaBoost Regression can be used for many regression problems, including:

- Medical Cost Prediction
- House Price Prediction
- Salary Prediction
- Sales Forecasting
- Customer Spending Prediction
- Demand Forecasting
- Financial Prediction
- Energy Consumption Prediction
- Product Price Prediction
- Business Analytics

In this project, the algorithm is specifically applied to:

```text
Medical Insurance Cost Prediction
```

---

# ✅ Advantages of AdaBoost Regression

- Combines multiple weak learners.
- Can model non-linear relationships.
- Sequentially improves predictions.
- Can provide strong predictive performance.
- Provides feature importance.
- Works effectively with Decision Tree base learners.
- Relatively easy to implement using Scikit-learn.
- Supports configurable learning rate and number of estimators.

---

# ❌ Limitations of AdaBoost Regression

- Sensitive to noisy data.
- Sensitive to extreme outliers.
- Sequential training can take more time.
- Performance depends on the quality of the base estimator.
- Hyperparameters need to be selected carefully.
- Too many estimators can increase model complexity.

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand the concept of AdaBoost Regression.
- Understand Ensemble Learning.
- Understand Boosting.
- Understand Weak Learners.
- Understand Sequential Learning.
- Use Decision Trees as base estimators.
- Build an AdaBoost Regression model using Scikit-learn.
- Understand `n_estimators`.
- Understand `learning_rate`.
- Evaluate regression models using MAE, MSE, RMSE, and R².
- Predict values for new observations.
- Inspect model hyperparameters.
- Analyze feature importance.
- Visualize feature importance using Seaborn.

---

# 📝 Conclusion

AdaBoost Regression is an ensemble Machine Learning algorithm that combines multiple weak regression learners to create a stronger predictive model.

In this project, a `DecisionTreeRegressor` with a maximum depth of 3 is used as the base estimator. AdaBoost combines 100 such boosting stages with a learning rate of 0.1.

The Medical Insurance dataset is first explored and preprocessed. Categorical variables are encoded using `LabelEncoder`, and the data is divided into training and testing sets.

The trained model is then used to predict insurance charges and is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The model achieves:

```text
MAE  = 4141.76
MSE  = 25889199.17
RMSE = 5088.14
R²   = 0.8332
```

The notebook also demonstrates prediction for a new customer, inspection of AdaBoost hyperparameters, and feature importance analysis.

The feature importance results show that `smoker` has the highest importance in this particular trained model, followed by `bmi` and `age`.

Overall, this project provides a practical introduction to **AdaBoost Regression and ensemble learning** using a real-world medical insurance cost prediction problem.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Engineer | Data Science Enthusiast | Python Developer
