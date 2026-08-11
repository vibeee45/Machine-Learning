# 🚀 XGBoost Regression

A comprehensive Machine Learning project demonstrating **XGBoost Regression** using the **Medical Cost Personal (Insurance)** dataset.

This project follows a complete regression workflow starting from dataset loading and exploration, followed by preprocessing, categorical encoding, train-test splitting, XGBoost model creation, model training, prediction, evaluation, new-customer prediction, hyperparameter inspection, and feature importance analysis.

The objective of this project is to understand how **XGBoost Regression** can be used to predict a continuous target variable such as medical insurance charges.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- What is Supervised Learning?
- What is Regression?
- What is Ensemble Learning?
- What is Boosting?
- What is XGBoost?
- What is XGBoost Regression?
- How XGBoost Works
- Decision Trees in XGBoost
- Sequential Learning
- Learning Rate
- Number of Estimators
- Maximum Depth
- Regularization
- Advantages of XGBoost
- Limitations of XGBoost
- Dataset Information
- Dataset Features
- Target Variable
- Technologies Used
- Project Structure
- Project Workflow
- Step-by-Step Notebook Explanation
- Model Evaluation
- Feature Importance
- New Customer Prediction
- Conclusion
- Learning Outcomes

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without requiring every rule to be manually programmed.

A Machine Learning model learns relationships from historical examples.

For example, in this project we have information about:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

along with medical insurance charges.

The Machine Learning model learns the relationship between these input features and the insurance charges.

After training, the model can use the learned patterns to predict insurance charges for new observations.

Machine Learning is commonly used in:

- Healthcare
- Finance
- Banking
- Marketing
- Fraud Detection
- Recommendation Systems
- Business Analytics
- Forecasting
- Risk Analysis
- Customer Analytics

---

# 🧠 What is Supervised Learning?

Supervised Learning is a Machine Learning approach where a model learns from labeled data.

The dataset contains:

```text
Input Features → Known Target
```

The model attempts to learn a relationship between the input variables and the known target.

Supervised Learning is mainly divided into:

```text
Supervised Learning
       │
       ├── Classification
       │
       └── Regression
```

XGBoost Regression belongs to the **Supervised Learning** category.

---

# 📈 What is Regression?

Regression is a supervised Machine Learning technique used to predict continuous numerical values.

Unlike classification, where the output represents a class or category, regression produces a numerical prediction.

Examples of regression problems include:

- House Price Prediction
- Salary Prediction
- Medical Insurance Cost Prediction
- Sales Forecasting
- Revenue Prediction
- Temperature Prediction
- Electricity Consumption Prediction
- Demand Forecasting

In this project, the target variable is:

```text
charges
```

The XGBoost model learns to predict medical insurance charges from the available customer information.

---

# 🌐 What is Ensemble Learning?

Ensemble Learning combines multiple Machine Learning models to create a stronger predictive system.

Instead of depending entirely on one model, an ensemble uses multiple learners.

Common ensemble approaches include:

- Bagging
- Boosting
- Stacking

XGBoost is a highly optimized implementation of **gradient boosting**.

---

# 📈 What is Boosting?

Boosting is an ensemble learning technique in which multiple weak learners are trained sequentially.

The general idea is:

```text
Training Data
      ↓
Learner 1
      ↓
Evaluate Errors
      ↓
Learner 2
      ↓
Improve Previous Model
      ↓
Learner 3
      ↓
Continue Improving
      ↓
Final Strong Model
```

Each new learner attempts to improve the predictions produced by the existing ensemble.

XGBoost uses this boosting idea with decision trees as its base learners.

---

# ⚡ What is XGBoost?

**XGBoost** stands for **Extreme Gradient Boosting**.

It is a powerful and optimized implementation of gradient boosting designed for high performance, scalability, and predictive accuracy.

XGBoost builds an ensemble of decision trees sequentially.

Each new tree attempts to reduce the errors made by the existing ensemble.

The basic idea can be represented as:

```text
Initial Prediction
       ↓
Calculate Errors
       ↓
Build Tree to Correct Errors
       ↓
Update Prediction
       ↓
Build Another Tree
       ↓
Update Prediction Again
       ↓
      ...
       ↓
Final XGBoost Model
```

XGBoost is widely used for:

- Regression
- Classification
- Ranking
- Structured/tabular data problems
- Business prediction
- Financial prediction
- Risk modeling

---

# 📊 What is XGBoost Regression?

XGBoost Regression is the regression implementation of XGBoost.

It is used when the target variable is continuous.

In this project, XGBoost Regression predicts:

```text
Medical Insurance Charges
```

using:

```text
age
sex
bmi
children
smoker
region
```

The target is:

```text
charges
```

---

# 🔄 How XGBoost Works

XGBoost builds decision trees one after another.

The first tree makes an initial prediction.

The errors from the current model are then used to guide the next tree.

The next tree attempts to reduce the remaining error.

This process continues for the specified number of estimators.

Conceptually:

```text
Tree 1
  ↓
Prediction
  ↓
Error
  ↓
Tree 2
  ↓
Correct Previous Error
  ↓
Tree 3
  ↓
Correct Remaining Error
  ↓
...
  ↓
Final Prediction
```

This is why XGBoost is considered a boosting algorithm.

---

# 🌳 Decision Trees in XGBoost

XGBoost uses decision trees as its base learners.

The model in this project is configured with:

```python
max_depth=3
```

This limits the depth of each tree.

A smaller depth produces relatively simple trees.

The ensemble then combines many such trees to learn more complex patterns.

---

# 🔢 Number of Estimators

The `n_estimators` parameter controls the number of boosting rounds or trees.

This project uses:

```python
n_estimators=100
```

Therefore, the model is configured to build 100 boosting stages.

Increasing the number of estimators can increase the model's ability to learn complex patterns.

However, a very large number can:

- Increase training time.
- Increase model complexity.
- Potentially increase overfitting.

---

# ⚙️ Learning Rate

The `learning_rate` parameter controls how strongly each new tree contributes to the final prediction.

This project uses:

```python
learning_rate=0.1
```

A smaller learning rate means each tree makes a smaller contribution.

A larger learning rate makes each tree contribute more strongly.

There is commonly a trade-off between:

```text
Learning Rate
      ↕
Number of Estimators
```

For example, a smaller learning rate may require more trees.

---

# 🌲 Maximum Depth

The `max_depth` parameter controls the maximum depth of each decision tree.

This project uses:

```python
max_depth=3
```

A smaller depth:

- Keeps trees simpler.
- Reduces model complexity.
- Can help control overfitting.

A larger depth:

- Allows more complex relationships.
- Can capture more interactions.
- Can increase the risk of overfitting.

---

# 🛡️ Regularization

XGBoost includes regularization mechanisms that help control model complexity.

Regularization can help prevent the model from becoming unnecessarily complex.

Important XGBoost parameters related to regularization and tree complexity can include:

- `reg_alpha`
- `reg_lambda`
- `gamma`
- `max_depth`
- `min_child_weight`

The current notebook uses a simple configuration and does not explicitly set all of these parameters.

The primary parameters explicitly configured in this project are:

```python
n_estimators=100
learning_rate=0.1
max_depth=3
random_state=42
```

---

# ⚙️ XGBoost Model Configuration

The model used in this project is:

```python
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

The parameters are:

| Parameter | Value | Purpose |
|----------|-------|---------|
| `n_estimators` | 100 | Number of boosting trees |
| `learning_rate` | 0.1 | Contribution of each tree |
| `max_depth` | 3 | Maximum tree depth |
| `random_state` | 42 | Reproducible results |

---

# ✅ Advantages of XGBoost Regression

XGBoost provides several advantages:

- Strong performance on structured data.
- Captures non-linear relationships.
- Captures feature interactions.
- Uses boosting to improve predictions sequentially.
- Supports regularization.
- Provides feature importance.
- Supports many useful hyperparameters.
- Highly optimized implementation of gradient boosting.
- Works well for many regression problems.

---

# ❌ Limitations of XGBoost Regression

XGBoost also has limitations:

- More complex than simple linear models.
- Requires careful hyperparameter selection.
- Can overfit if model complexity is not controlled.
- Training can become computationally expensive for very large configurations.
- Predictions are less interpretable than a simple linear equation.
- Categorical variables generally need suitable preprocessing depending on the implementation and configuration.

---

# 📂 Dataset Information

## Dataset Name

```text
insurance.csv
```

The dataset contains information about individuals and their medical insurance costs.

The objective is to predict:

```text
charges
```

using demographic and health-related features.

---

# 📊 Dataset Summary

The dataset contains:

```text
Rows    : 1338
Columns : 7
```

The seven columns are:

```text
age
sex
bmi
children
smoker
region
charges
```

The notebook checks the dataset structure using:

```python
df.info()
```

and confirms the presence of numerical and categorical variables.

---

# 🧩 Dataset Features

| Feature | Description |
|---------|-------------|
| `age` | Age of the individual |
| `sex` | Sex of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependents |
| `smoker` | Smoking status |
| `region` | Residential region |
| `charges` | Medical insurance charges |

---

# 🎯 Target Variable

The target variable is:

```text
charges
```

It represents the medical insurance charges that the model attempts to predict.

The target is separated from the feature dataset using:

```python
X = df.drop("charges", axis=1)
y = df["charges"]
```

Therefore:

```text
X → Input Features
y → Target
```

The feature matrix contains six input variables.

---

# 🔤 Categorical Features

Three columns are categorical:

```text
sex
smoker
region
```

The notebook uses `LabelEncoder` to convert these values into numerical representations.

The encoder is applied using:

```python
encoder = LabelEncoder()

categorical_columns = [
    "sex",
    "smoker",
    "region"
]

for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])
```

After encoding, the Machine Learning model receives numerical feature values.

---

# 🔢 Train-Test Split

The dataset is divided into training and testing subsets using:

```python
train_test_split()
```

The notebook uses:

```python
test_size=0.20
```

Therefore:

```text
80% → Training
20% → Testing
```

The split also uses:

```python
random_state=42
```

which makes the split reproducible.

The resulting shapes are:

```text
Training Features : (1070, 6)
Testing Features  : (268, 6)

Training Labels : (1070,)
Testing Labels  : (268,)
```

---

# 🛠️ Technologies Used

The project uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

# 📁 Project Structure

```text
XGBoost-Regression/
│
├── XGBoost Regression(1).ipynb
├── insurance.csv
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 🔄 Complete Project Workflow

```text
Import Libraries
       ↓
Load Dataset
       ↓
Explore Dataset
       ↓
Display Dataset Information
       ↓
Statistical Summary
       ↓
Check Missing Values
       ↓
Display Dataset Dimensions
       ↓
Separate Features and Target
       ↓
Encode Categorical Variables
       ↓
Train-Test Split
       ↓
Create XGBoost Regressor
       ↓
Train XGBoost Model
       ↓
Predict Test Data
       ↓
Compare Actual vs Predicted
       ↓
Calculate MAE
       ↓
Calculate MSE
       ↓
Calculate RMSE
       ↓
Calculate R²
       ↓
Predict New Customer
       ↓
Inspect Hyperparameters
       ↓
Analyze Feature Importance
```

---

# 📖 Step-by-Step Notebook Explanation

The following sections explain the cells in the XGBoost Regression notebook in the same order in which they appear.

The notebook contains 20 completed code cells.

---

## 🔹 Cell 1 – Importing the Required Libraries

```python
# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# XGBoost
from xgboost import XGBRegressor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

### Purpose

This cell imports the libraries required for the entire project.

The notebook does not contain an installation cell because XGBoost is already installed in the environment.

---

### NumPy

```python
import numpy as np
```

NumPy is used for numerical calculations.

It is later used to calculate RMSE:

```python
rmse = np.sqrt(mse)
```

---

### Pandas

```python
import pandas as pd
```

Pandas is used for:

- Reading the CSV file.
- Creating DataFrames.
- Selecting columns.
- Manipulating features.
- Creating comparison tables.
- Creating feature importance tables.

---

### Matplotlib

```python
import matplotlib.pyplot as plt
```

Matplotlib is used for plotting the feature importance visualization.

---

### Seaborn

```python
import seaborn as sns
```

Seaborn is used to create the feature importance bar chart.

---

### Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

This function divides the dataset into training and testing subsets.

---

### Label Encoder

```python
from sklearn.preprocessing import LabelEncoder
```

`LabelEncoder` converts categorical values into numerical values.

---

### XGBRegressor

```python
from xgboost import XGBRegressor
```

`XGBRegressor` is the main regression model used in this project.

---

### Warning Handling

```python
import warnings
warnings.filterwarnings("ignore")
```

This prevents warning messages from unnecessarily cluttering the notebook output.

---

# 🔹 Cell 2 – Loading the Dataset

```python
# Loading the dataset

df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\insurance.csv"
)

df.head()
```

### Purpose

This cell loads the Insurance dataset using Pandas.

The DataFrame is stored in:

```python
df
```

The dataset is loaded with:

```python
pd.read_csv()
```

The notebook uses a local Windows path.

For a GitHub repository where the CSV is stored beside the notebook, it can be changed to:

```python
df = pd.read_csv("insurance.csv")
```

The `head()` method displays the first five records.

The first five rows contain:

```text
age
sex
bmi
children
smoker
region
charges
```

This confirms that the dataset has loaded correctly.

---

# 🔹 Cell 3 – Displaying Dataset Information

```python
# Displaying dataset information

df.info()
```

### Purpose

This cell provides structural information about the DataFrame.

It displays:

- Number of rows.
- Number of columns.
- Column names.
- Non-null counts.
- Data types.
- Memory usage.

The dataset contains:

```text
1338 entries
7 columns
```

The data types include:

```text
int64
float64
object
```

The categorical columns are stored as object values before preprocessing.

This information is important before building the Machine Learning model.

---

# 🔹 Cell 4 – Statistical Summary

```python
# Statistical summary of numerical columns

df.describe()
```

### Purpose

The `describe()` method generates descriptive statistics for numerical columns.

The numerical columns include:

```text
age
bmi
children
charges
```

The output provides:

- Count
- Mean
- Standard deviation
- Minimum
- 25th percentile
- Median
- 75th percentile
- Maximum

For example, the dataset contains 1338 records for the numerical columns.

The mean values include approximately:

```text
Age       → 39.207025
BMI       → 30.663397
Children  → 1.094918
Charges   → 13270.422265
```

The maximum `charges` value is approximately:

```text
63770.428010
```

This statistical summary helps understand the distribution and scale of the numerical variables.

---

# 🔹 Cell 5 – Checking Missing Values

```python
# Checking for missing values

df.isnull().sum()
```

### Purpose

Before training a Machine Learning model, it is important to determine whether the dataset contains missing values.

The expression:

```python
df.isnull()
```

identifies missing values.

The `sum()` method counts them for every column.

The dataset contains:

```text
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
```

Therefore, no missing-value treatment is required for this dataset.

---

# 🔹 Cell 6 – Displaying Dataset Dimensions

```python
# Displaying dataset dimensions

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
```

### Purpose

This cell displays the dimensions of the dataset.

The output is:

```text
Rows : 1338
Columns : 7
```

Therefore, the dataset contains:

```text
1338 observations
7 columns
```

This is a quick way to verify the dataset size.

---

# 🔹 Cell 7 – Selecting Independent and Dependent Variables

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

This cell separates the dataset into input features and the target variable.

The target is:

```text
charges
```

Therefore:

```python
X = df.drop("charges", axis=1)
```

removes `charges` from the feature dataset.

The target is stored separately:

```python
y = df["charges"]
```

---

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

The feature matrix has:

```text
1338 rows
6 columns
```

Therefore:

```text
Features Shape: (1338, 6)
```

---

### Target

The target contains:

```text
1338 values
```

Therefore:

```text
Target Shape: (1338,)
```

The model will learn the relationship:

```text
Features → charges
```

---

# 🔹 Cell 8 – Encoding Categorical Columns

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

The dataset contains categorical variables.

The selected categorical columns are:

```text
sex
smoker
region
```

These columns contain categorical values rather than purely numerical values.

The notebook creates a `LabelEncoder`:

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

The notebook then loops through these columns:

```python
for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])
```

This converts the categorical values into numerical representations.

After encoding, the feature dataset contains numerical values for all six input columns.

---

# 🔹 Cell 9 – Splitting the Dataset

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

The dataset is divided into training and testing data.

The split is performed using:

```python
train_test_split()
```

The notebook specifies:

```python
test_size=0.20
```

Therefore:

```text
80% → Training
20% → Testing
```

The resulting dimensions are:

```text
Training Features : (1070, 6)
Testing Features : (268, 6)

Training Labels : (1070,)
Testing Labels : (268,)
```

The model learns from:

```text
X_train
y_train
```

and is later evaluated using:

```text
X_test
y_test
```

The `random_state=42` parameter makes the split reproducible.

---

# 🔹 Cell 10 – Creating the XGBoost Regression Model

```python
# Creating the XGBoost Regression model

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

print("XGBoost Regression model created successfully!")
```

### Purpose

This cell creates the XGBoost Regression model.

The class used is:

```python
XGBRegressor
```

The model is configured with four explicit parameters.

---

### `n_estimators`

```python
n_estimators=100
```

This specifies the number of boosting trees.

The model is configured to use 100 trees.

---

### `learning_rate`

```python
learning_rate=0.1
```

This controls the contribution of each tree to the overall model.

A smaller value generally makes each boosting step more gradual.

---

### `max_depth`

```python
max_depth=3
```

This limits the maximum depth of each decision tree.

It controls the complexity of the individual trees.

---

### `random_state`

```python
random_state=42
```

This is used to make the model's results reproducible where randomness is involved.

---

### Model Configuration

The final model configuration is:

```text
XGBRegressor
    │
    ├── n_estimators = 100
    ├── learning_rate = 0.1
    ├── max_depth = 3
    └── random_state = 42
```

The notebook then prints:

```text
XGBoost Regression model created successfully!
```

---

# 🔹 Cell 11 – Training the XGBoost Regression Model

```python
# Training the XGBoost Regression model

model.fit(X_train, y_train)

print("XGBoost Regression model trained successfully!")
```

### Purpose

This cell trains the XGBoost model using the training dataset.

The training data consists of:

```text
X_train
y_train
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

During training, XGBoost builds its sequence of decision trees.

Each boosting stage contributes toward improving the overall predictions.

The trained model is then ready to predict insurance charges for unseen data.

---

# 🔹 Cell 12 – Predicting the Test Data

```python
# Predicting the test data

y_pred = model.predict(X_test)

print("Predicted Insurance Charges:")
print(y_pred[:10])
```

### Purpose

After training, the model is used to predict insurance charges for the testing dataset.

The prediction is generated using:

```python
y_pred = model.predict(X_test)
```

The predictions are stored in:

```text
y_pred
```

Only the first ten predictions are displayed:

```python
y_pred[:10]
```

The predictions can later be compared with:

```text
y_test
```

which contains the actual insurance charges.

---

# 🔹 Cell 13 – Comparing Actual and Predicted Values

```python
# Comparing Actual and Predicted Values

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.head(10)
```

### Purpose

This cell creates a comparison table containing:

- Actual insurance charges.
- Predicted insurance charges.

A DataFrame is created:

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
```

The two columns are:

```text
Actual
Predicted
```

The first ten rows are displayed using:

```python
comparison.head(10)
```

This allows us to visually inspect how closely the model's predictions match the actual target values.

---

# 🔹 Cell 14 – Mean Absolute Error

```python
from sklearn.metrics import mean_absolute_error

# Calculating Mean Absolute Error

mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
```

### Purpose

This cell calculates the **Mean Absolute Error**, commonly called MAE.

MAE measures the average absolute difference between actual and predicted values.

The formula is:

```text
MAE = (1/n) Σ |Actual − Predicted|
```

The metric is calculated using:

```python
mean_absolute_error(y_test, y_pred)
```

The result is stored in:

```python
mae
```

A lower MAE generally indicates that predictions are closer to actual values.

Because MAE uses absolute errors, all errors are treated as positive distances from the actual values.

---

# 🔹 Cell 15 – Mean Squared Error

```python
from sklearn.metrics import mean_squared_error

# Calculating Mean Squared Error

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
```

### Purpose

This cell calculates the **Mean Squared Error**, or MSE.

The formula is:

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

The errors are squared before being averaged.

This means larger errors receive a larger penalty.

The calculation is performed using:

```python
mean_squared_error(y_test, y_pred)
```

The result is stored in:

```python
mse
```

A lower MSE indicates smaller prediction errors.

---

# 🔹 Cell 16 – Root Mean Squared Error

```python
# Calculating Root Mean Squared Error

rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```

### Purpose

RMSE is the square root of Mean Squared Error.

The formula is:

```text
RMSE = √MSE
```

The notebook uses NumPy:

```python
np.sqrt(mse)
```

to calculate RMSE.

The result is stored in:

```python
rmse
```

RMSE is useful because it is expressed in the same unit as the target variable.

Like MSE, RMSE gives greater influence to larger errors.

A lower RMSE generally indicates better predictive performance.

---

# 🔹 Cell 17 – R² Score

```python
from sklearn.metrics import r2_score

# Calculating R² Score

r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
```

### Purpose

This cell calculates the **R² Score**, also known as the Coefficient of Determination.

The R² score measures how well the model explains variation in the target variable.

The calculation is:

```python
r2_score(y_test, y_pred)
```

The result is stored in:

```python
r2
```

General interpretation:

```text
R² close to 1 → Strong explanatory performance
R² close to 0 → Weak explanatory performance
Negative R²   → Worse than the mean baseline
```

The R² score should be interpreted together with other metrics rather than in isolation.

---

# 🔹 Cell 18 – Predicting Insurance Cost for a New Customer

```python
# Predicting insurance cost for a new customer

new_customer = pd.DataFrame({
    "age": [35],
    "sex": [1],
    "bmi": [28.5],
    "children": [2],
    "smoker": [0],
    "region": [2]
})

# Predicting insurance charges

predicted_charge = model.predict(new_customer)

print(f"Predicted Insurance Charge: ₹{predicted_charge[0]:,.2f}")
```

### Purpose

This cell demonstrates how the trained XGBoost model can be used to predict the insurance cost of a new customer.

The new customer has:

```text
Age       = 35
Sex       = 1
BMI       = 28.5
Children  = 2
Smoker    = 0
Region    = 2
```

The data is stored in a DataFrame:

```python
new_customer
```

The model then predicts the insurance charge:

```python
predicted_charge = model.predict(new_customer)
```

The result is displayed using:

```python
print(f"Predicted Insurance Charge: ₹{predicted_charge[0]:,.2f}")
```

This demonstrates how the trained model can be used for a real prediction scenario.

---

# 🔹 Cell 19 – Displaying XGBoost Hyperparameters

```python
# Displaying XGBoost Hyperparameters

parameters = pd.DataFrame({
    "Hyperparameter": model.get_params().keys(),
    "Value": model.get_params().values()
})

display(parameters)
```

### Purpose

This cell retrieves the model's hyperparameters using:

```python
model.get_params()
```

The parameters are converted into a DataFrame containing:

```text
Hyperparameter
Value
```

This makes the configuration of the XGBoost model easier to inspect.

Important parameters configured explicitly in the notebook are:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 3
random_state = 42
```

XGBoost also has many additional parameters with default values.

Displaying the complete parameter dictionary helps understand the model configuration being used.

---

# 🔹 Cell 20 – Displaying Feature Importance

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

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.grid(axis="x")

plt.show()
```

### Purpose

This cell analyzes the importance of the input features in the trained XGBoost model.

XGBoost provides feature importance through:

```python
model.feature_importances_
```

A DataFrame is created:

```python
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
```

The feature importance values are then sorted in descending order:

```python
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
```

This places the most important features at the top.

---

# 📊 Feature Importance Visualization

The notebook creates a bar chart using:

```python
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)
```

The plot uses:

```text
X-axis → Importance Score
Y-axis → Feature
```

The title is:

```text
XGBoost Feature Importance
```

The chart makes it easier to compare the contribution of the six input features.

---

# 📌 Features Used by the Model

The XGBoost model uses:

```text
age
sex
bmi
children
smoker
region
```

The target variable is:

```text
charges
```

The model therefore learns:

```text
age
sex
bmi
children
smoker
region
       ↓
XGBoost Regression
       ↓
charges
```

---

# 📏 Model Evaluation Metrics

The notebook evaluates the XGBoost model using four standard regression metrics:

```text
MAE
MSE
RMSE
R² Score
```

Each metric provides a different perspective on prediction performance.

---

# 📉 Mean Absolute Error

MAE measures the average absolute prediction error.

```text
MAE = (1/n) Σ |Actual − Predicted|
```

Advantages:

- Easy to interpret.
- Same units as the target.
- Treats each absolute error equally.

Lower values are generally better.

---

# 📉 Mean Squared Error

MSE calculates the average squared error.

```text
MSE = (1/n) Σ (Actual − Predicted)²
```

Because errors are squared, larger errors have a stronger effect.

Lower values are generally better.

---

# 📉 Root Mean Squared Error

RMSE is:

```text
RMSE = √MSE
```

RMSE is expressed in the same units as the target.

It gives more weight to larger errors than MAE.

Lower RMSE generally indicates better prediction performance.

---

# 📈 R² Score

R² measures the amount of variation in the target that is explained by the model.

A value closer to 1 generally indicates stronger explanatory performance.

However, R² should be interpreted alongside MAE, MSE, and RMSE.

---

# 🎯 New Customer Prediction

The project demonstrates a practical prediction using:

```python
new_customer = pd.DataFrame({
    "age": [35],
    "sex": [1],
    "bmi": [28.5],
    "children": [2],
    "smoker": [0],
    "region": [2]
})
```

The model predicts the insurance charge using:

```python
model.predict(new_customer)
```

This demonstrates how a trained XGBoost model can be applied to an unseen customer record.

---

# 📊 Hyperparameter Summary

The main XGBoost parameters selected in this project are:

| Parameter | Selected Value |
|----------|----------------|
| `n_estimators` | 100 |
| `learning_rate` | 0.1 |
| `max_depth` | 3 |
| `random_state` | 42 |

These values define the basic complexity and training behavior of the model.

---

# 🔍 Feature Importance

Feature importance provides information about which input variables contributed most to the trained model.

The six features analyzed are:

```text
age
sex
bmi
children
smoker
region
```

The importance scores are extracted using:

```python
model.feature_importances_
```

The values are sorted so that the most important features appear first.

---

# 📊 Why Feature Importance Matters

Feature importance can help us understand the trained model.

It can answer questions such as:

- Which feature contributes most to predictions?
- Which features have relatively low importance?
- Which variables should receive more attention during analysis?
- Which variables may be useful for future feature engineering?

Feature importance describes the behavior of the trained model and should not automatically be interpreted as causation.

---

# 🗂️ Complete Notebook Structure

The notebook follows this sequence:

```text
Cell 1
Import Required Libraries

Cell 2
Load Dataset

Cell 3
Display Dataset Information

Cell 4
Statistical Summary

Cell 5
Check Missing Values

Cell 6
Display Dataset Dimensions

Cell 7
Select Features and Target

Cell 8
Encode Categorical Variables

Cell 9
Train-Test Split

Cell 10
Create XGBoost Regression Model

Cell 11
Train XGBoost Model

Cell 12
Predict Test Data

Cell 13
Compare Actual and Predicted Values

Cell 14
Calculate MAE

Cell 15
Calculate MSE

Cell 16
Calculate RMSE

Cell 17
Calculate R² Score

Cell 18
Predict New Customer

Cell 19
Display Hyperparameters

Cell 20
Display Feature Importance
```

---

# 🔄 End-to-End XGBoost Workflow

```text
Raw Insurance Dataset
          ↓
Data Loading
          ↓
Data Exploration
          ↓
Missing Value Check
          ↓
Feature/Target Separation
          ↓
Categorical Encoding
          ↓
Train-Test Split
          ↓
XGBoost Regressor
          ↓
Model Training
          ↓
Test Prediction
          ↓
Actual vs Predicted
          ↓
Regression Metrics
          ↓
New Customer Prediction
          ↓
Hyperparameter Inspection
          ↓
Feature Importance
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand Machine Learning.
- Understand supervised learning.
- Understand regression.
- Understand ensemble learning.
- Understand boosting.
- Understand XGBoost.
- Understand XGBoost Regression.
- Understand the role of decision trees in boosting.
- Understand `n_estimators`.
- Understand `learning_rate`.
- Understand `max_depth`.
- Prepare categorical data for XGBoost.
- Split data into training and testing sets.
- Train an XGBoost Regression model.
- Generate regression predictions.
- Evaluate regression performance.
- Calculate MAE.
- Calculate MSE.
- Calculate RMSE.
- Calculate R².
- Predict values for new customers.
- Inspect model hyperparameters.
- Analyze feature importance.
- Visualize feature importance.

---

# 🚀 Installation

If XGBoost is already installed, no installation step is required.

Otherwise, XGBoost can be installed with:

```bash
pip install xgboost
```

The remaining project dependencies can be installed with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

# 📦 Requirements

The project requires:

```text
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
Jupyter Notebook
```

---

# ▶️ Running the Project

Clone or download the project repository.

Place:

```text
insurance.csv
```

in the appropriate dataset directory.

Open:

```text
XGBoost Regression(1).ipynb
```

using Jupyter Notebook or another compatible notebook environment.

Run the cells sequentially from Cell 1 through Cell 20.

---

# ⚠️ Dataset Path

The notebook currently loads the dataset using a local Windows path:

```python
df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\insurance.csv"
)
```

This path is specific to the development environment.

If another user downloads the project, they should change the path to match their own dataset location.

For example, if the dataset is in the same directory as the notebook:

```python
df = pd.read_csv("insurance.csv")
```

---

# 🧪 Reproducibility

The project uses:

```python
random_state=42
```

for:

- Train-test splitting.
- XGBoost model configuration.

Using a fixed random state helps make results reproducible when the same environment and data are used.

---

# 📝 Important Note About Categorical Encoding

The notebook uses `LabelEncoder` for:

```text
sex
smoker
region
```

The encoded values are used for model training.

When creating a new customer prediction, the values are therefore provided in their encoded form:

```python
"sex": [1],
"smoker": [0],
"region": [2]
```

The new input must use the same encoding convention as the training data.

---

# 🧠 Understanding the Complete Model

The complete model can be summarized as:

```text
Insurance Dataset
       ↓
Data Preprocessing
       ↓
Categorical Encoding
       ↓
Train-Test Split
       ↓
XGBoost Regression
       ↓
100 Boosting Trees
       ↓
Learning Rate = 0.1
       ↓
Maximum Tree Depth = 3
       ↓
Predictions
       ↓
Evaluation
```

---

# 🌟 Why XGBoost is Important

XGBoost is an important algorithm to learn because it is a powerful tree-based boosting technique.

It provides a bridge between traditional Machine Learning algorithms and modern gradient-boosting systems.

After understanding XGBoost, it becomes easier to explore other advanced boosting libraries and algorithms.

Important concepts learned through XGBoost include:

- Boosting
- Decision Trees
- Sequential learning
- Learning rate
- Tree depth
- Model complexity
- Feature importance
- Regression evaluation

---

# 📌 Project Objective

The primary objective of this project is to build an XGBoost Regression model that predicts medical insurance charges.

The model uses:

```text
age
sex
bmi
children
smoker
region
```

to predict:

```text
charges
```

The project demonstrates the complete process from raw data to trained Machine Learning model.

---

# 📊 Final Project Summary

```text
Dataset:
Medical Insurance Dataset

Rows:
1338

Columns:
7

Input Features:
6

Target:
charges

Training Samples:
1070

Testing Samples:
268

Model:
XGBRegressor

Number of Estimators:
100

Learning Rate:
0.1

Maximum Depth:
3

Random State:
42
```

---

# 🏁 Conclusion

XGBoost Regression is a powerful ensemble learning algorithm based on gradient boosting.

In this project, XGBoost is used to predict medical insurance charges using demographic and health-related features.

The project begins by loading and exploring the Insurance dataset.

The categorical variables are converted into numerical representations using `LabelEncoder`.

The dataset is then divided into training and testing subsets.

An `XGBRegressor` model is created with:

```python
n_estimators=100
learning_rate=0.1
max_depth=3
random_state=42
```

The model is trained using the training data and then used to predict insurance charges for the test dataset.

The predictions are evaluated using:

- MAE
- MSE
- RMSE
- R² Score

The project also demonstrates how to:

- Compare actual and predicted values.
- Predict the insurance charge of a new customer.
- Inspect model hyperparameters.
- Analyze feature importance.
- Visualize feature importance.

Overall, this project provides a practical introduction to **XGBoost Regression**, one of the most important gradient-boosting techniques for structured Machine Learning problems.



# ⭐ Project Workflow Summary

```text
                 XGBoost Regression
                         │
                         ▼
                Load Insurance Data
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
                XGBRegressor Model
                         │
                         ▼
                  Train the Model
                         │
                         ▼
                  Make Predictions
                         │
                         ▼
              Actual vs Predicted
                         │
                         ▼
              ┌──────────┴──────────┐
              ▼                     ▼
         Error Metrics       Feature Importance
              │                     │
              ▼                     ▼
       MAE / MSE / RMSE       Importance Plot
              │
              ▼
             R²
              │
              ▼
       New Customer Prediction
```

---

# 🎯 Final Takeaway

The most important concepts demonstrated by this project are:

```text
Machine Learning
      ↓
Supervised Learning
      ↓
Regression
      ↓
Ensemble Learning
      ↓
Boosting
      ↓
Gradient Boosting
      ↓
XGBoost Regression
```

XGBoost builds a strong regression model by combining multiple decision trees in a boosting framework.

The final model can then be used to predict continuous values such as medical insurance charges.

This completes the XGBoost Regression project.
---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning | Data Science | Python

---
