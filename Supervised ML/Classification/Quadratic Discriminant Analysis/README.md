# 🟣 Quadratic Discriminant Analysis (QDA) — Classification

A comprehensive Machine Learning project demonstrating **Quadratic Discriminant Analysis (QDA)** as a **Supervised Learning classification algorithm** using the **Wine Recognition Dataset** from Scikit-learn.

This project contains a compact **20-cell implementation** covering dataset loading, exploration, feature-target separation, train-test splitting, feature scaling, QDA training, prediction, accuracy, confusion matrix, classification report, probability analysis, actual-vs-predicted comparison, and final evaluation.

---

# 📑 Table of Contents

- Project Overview
- Machine Learning
- Supervised Learning
- Classification
- What is QDA?
- QDA as Classification
- Why QDA?
- How QDA Works
- Gaussian Distribution
- Class Mean
- Class-Specific Covariance
- Class Priors
- Bayes Decision Rule
- Quadratic Decision Boundary
- QDA vs LDA
- QDA vs Logistic Regression
- Dataset
- Dataset Features
- Target Variable
- Train-Test Split
- Stratification
- Feature Scaling
- QDA Model
- Predictions
- Accuracy
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score
- Prediction Probabilities
- Actual vs Predicted
- Complete Workflow
- 20 Cell Explanation
- Advantages
- Limitations
- Applications
- Installation
- Requirements
- Learning Outcomes
- Conclusion
- Final Takeaway

---

# 📌 Project Overview

The purpose of this project is to understand how **Quadratic Discriminant Analysis** can be used to classify observations into known categories.

The project uses the Wine Recognition Dataset provided by Scikit-learn.

The dataset contains:

```text
178 observations
13 numerical features
3 target classes
```

The task is supervised classification because target labels are available during training.

The feature matrix is separated from the target vector.

The dataset is divided into training and testing subsets.

The split uses stratification so that class proportions are preserved as much as possible.

The features are standardized using `StandardScaler`.

A `QuadraticDiscriminantAnalysis` model is created.

The model is fitted on scaled training data.

Predictions are generated for scaled test data.

The model is evaluated using accuracy, confusion matrix, and classification report.

Class probabilities are also calculated.

Actual and predicted classes are displayed together.

The implementation contains exactly **20 cells**.

The documentation below explains the complete project and all 20 cells.

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that enables computers to learn patterns from data and use those patterns to make predictions or decisions.

A simplified Machine Learning structure is:

```text
Machine Learning
       │
       ├── Supervised Learning
       │       ├── Regression
       │       └── Classification
       │
       ├── Unsupervised Learning
       │       ├── Clustering
       │       └── Dimensionality Reduction
       │
       └── Reinforcement Learning
```

QDA belongs to:

```text
Supervised Learning
        ↓
Classification
        ↓
Quadratic Discriminant Analysis
```

---

# 🧠 What is Supervised Learning?

Supervised Learning uses labeled data.

The model receives:

```text
X → Input Features
y → Target / Label
```

The algorithm learns the relationship between the features and known target classes.

For this project:

```text
X → Wine chemical measurements
y → Wine class
```

After training, the model predicts classes for observations it has not seen during training.

---

# 🏷️ What is Classification?

Classification is a supervised learning task where the target represents categories.

Examples include:

```text
Email
→ Spam / Not Spam

Transaction
→ Fraud / Not Fraud

Patient
→ Disease / No Disease

Wine
→ Class 0 / Class 1 / Class 2
```

The goal of this project is:

```text
Wine Features
      ↓
QDA Classifier
      ↓
Predicted Wine Class
```

The Wine dataset contains three classes, so this is a multiclass classification problem.

---

# 🟣 What is QDA?

QDA stands for:

> **Quadratic Discriminant Analysis**

QDA is a supervised statistical classification method.

It is closely related to Linear Discriminant Analysis.

The major difference is that QDA allows every class to have its own covariance matrix.

Conceptually:

```text
Class 0 → Mean + Covariance 0
Class 1 → Mean + Covariance 1
Class 2 → Mean + Covariance 2
```

Because covariance can differ between classes, QDA can produce quadratic decision boundaries.

---

# 🎯 QDA as a Classification Algorithm

The notebook treats QDA as:

```text
Supervised Machine Learning
          ↓
      Classification
          ↓
 Quadratic Discriminant Analysis
```

Training:

```text
X_train + y_train
        ↓
       QDA
        ↓
   Learned Model
```

Prediction:

```text
X_test
   ↓
QDA Model
   ↓
y_pred
```

QDA can also generate class probabilities.

---

# ⭐ Why Use QDA?

QDA is useful when different classes have different covariance structures.

Classical LDA assumes a common covariance structure.

QDA relaxes that assumption.

This gives QDA more flexibility.

The trade-off is that QDA estimates more parameters.

Therefore, QDA can require more data for stable covariance estimation.

---

# 📈 Gaussian Distribution

QDA models the distribution of observations within each class using multivariate Gaussian distributions.

A Gaussian distribution is characterized by:

```text
Mean
+
Covariance
```

For every class, QDA estimates the parameters needed to describe that class distribution.

The model then evaluates which class distribution best explains a new observation.

---

# 📍 Class Mean

Each class has its own mean vector.

The Wine dataset has 13 features.

Therefore, each class mean contains 13 values.

Conceptually:

```text
Class 0 → Mean Vector 0
Class 1 → Mean Vector 1
Class 2 → Mean Vector 2
```

The mean represents the center of each class distribution.

---

# 📊 Class-Specific Covariance

The key QDA idea is that each class can have its own covariance matrix.

```text
Class 0 → Σ₀
Class 1 → Σ₁
Class 2 → Σ₂
```

This allows the classes to have different:

- Spread
- Orientation
- Feature relationships

This flexibility is what allows QDA to produce quadratic decision boundaries.

---

# ⚖️ Class Prior Probability

QDA also considers class priors.

A class prior represents the probability of a class before considering the features of a particular observation.

Conceptually:

```text
Class Prior
+
Feature Likelihood
        ↓
Posterior Probability
```

The class with the strongest posterior probability is selected as the prediction.

---

# 🧮 Bayes Decision Rule

QDA can be understood through Bayes' theorem.

Conceptually:

```text
P(Class | Features)
        ∝
P(Features | Class) × P(Class)
```

The model evaluates how likely the observed features are under every class distribution.

It also considers the prior probability of each class.

The most probable class becomes the prediction.

---

# 📐 Quadratic Decision Boundary

The name QDA comes from the quadratic form of its decision function.

Because every class has its own covariance matrix, the boundary between two classes can be curved.

Conceptually:

```text
LDA
→ Linear boundary
```

while:

```text
QDA
→ Quadratic boundary
```

The actual decision boundary depends on the learned class distributions.

---

# 🆚 QDA vs LDA

| LDA | QDA |
|---|---|
| Linear Discriminant Analysis | Quadratic Discriminant Analysis |
| Supervised classification | Supervised classification |
| Common covariance assumption | Separate covariance per class |
| Linear decision boundaries | Quadratic decision boundaries |
| Fewer parameters | More parameters |
| More restrictive | More flexible |
| Often more stable with limited data | Can need more data |

The LDA project similarly treats LDA as supervised classification and uses the Wine dataset. fileciteturn15file1L217-L223

The important distinction is:

```text
LDA → Common covariance
QDA → Class-specific covariance
```

---

# 🆚 QDA vs Logistic Regression

| QDA | Logistic Regression |
|---|---|
| Models class distributions | Models class probabilities directly |
| Uses means and covariance | Uses learned coefficients |
| Can create quadratic boundaries | Standard form creates linear boundaries |
| Generative approach | Discriminative approach |
| Flexible covariance structure | Simpler decision structure |

Both can be used for multiclass classification.

---

# 🍷 Dataset

The project uses the **Wine Recognition Dataset** from Scikit-learn.

It is loaded with:

```python
from sklearn.datasets import load_wine

wine = load_wine()
```

The dataset contains:

```text
178 observations
13 numerical features
3 classes
```

The original target labels are used because QDA is a supervised classification algorithm.

---

# 📋 Dataset Structure

The DataFrame is created using:

```python
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df["target"] = wine.target
```

The DataFrame contains:

```text
13 feature columns
+
1 target column
=
14 columns
```

The target is the dependent variable.

---

# 🧪 Dataset Features

The 13 features are:

```text
alcohol
malic_acid
ash
alcalinity_of_ash
magnesium
total_phenols
flavanoids
nonflavanoid_phenols
proanthocyanins
color_intensity
hue
od280/od315_of_diluted_wines
proline
```

All features are numerical.

No categorical encoding is required.

---

# 🎯 Target Variable

The target is:

```python
y = df["target"]
```

The target contains three classes:

```text
0
1
2
```

Therefore:

```text
13 Features
     ↓
    QDA
     ↓
3 Possible Classes
```

---

# 🔀 Train-Test Split

The notebook uses:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The test size is 20%.

Approximately 80% is used for training.

With 178 observations:

```text
Training → 142
Testing  → 36
```

The random state is 42.

This makes the split reproducible.

---

# 🧩 Stratification

The split uses:

```python
stratify=y
```

Stratification attempts to preserve class proportions in the training and testing subsets.

This is useful for classification problems.

The concept is:

```text
Original Dataset
       ↓
Preserve Class Proportions
       ↓
Training + Testing
```

---

# 📏 Feature Scaling

The notebook uses:

```python
scaler = StandardScaler()
```

Training data is scaled with:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

Testing data is transformed with:

```python
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on training data.

This avoids fitting preprocessing parameters on the test set.

The scaled shapes remain:

```text
Training → 142 × 13
Testing  → 36 × 13
```

---

# 🛡️ Correct Scaling Workflow

The correct workflow is:

```text
X_train
   ↓
fit_transform
   ↓
X_train_scaled

X_test
   ↓
transform using training scaler
   ↓
X_test_scaled
```

This keeps the test set independent for evaluation.

---

# 🧠 QDA Model

The model is created with:

```python
qda = QuadraticDiscriminantAnalysis()
```

This creates the classifier object.

The model is not trained yet.

Training happens with:

```python
qda.fit(
    X_train_scaled,
    y_train
)
```

---

# 🏋️ Training

QDA is trained using:

```python
qda.fit(X_train_scaled, y_train)
```

The model receives:

```text
Scaled Training Features
+
Training Labels
```

It estimates the statistical parameters of the three class distributions.

---

# 🔮 Prediction

Predictions are generated using:

```python
y_pred = qda.predict(X_test_scaled)
```

The resulting array contains one predicted class for every test observation.

The possible predictions are:

```text
0
1
2
```

---

# 🎯 Accuracy

Accuracy is calculated using:

```python
accuracy_score(y_test, y_pred)
```

The formula is:

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

The notebook prints both the decimal value and percentage.

The README does not invent a numerical accuracy value when the executed output is not supplied.

The value printed by Cell 13 is the authoritative result.

---

# 📊 Confusion Matrix

The confusion matrix is calculated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

For three classes, it has a:

```text
3 × 3
```

structure.

Rows represent actual classes.

Columns represent predicted classes.

The diagonal represents correct predictions.

Off-diagonal values represent misclassifications.

---

# 📉 Confusion Matrix Visualization

Cell 15 uses a Seaborn heatmap to visualize the confusion matrix.

The plot contains:

```text
Actual Class
vs
Predicted Class
```

The numeric annotations make individual classification counts easy to inspect.

The diagonal should ideally contain most of the observations.

---

# 📑 Classification Report

Cell 16 generates:

```python
classification_report(
    y_test,
    y_pred
)
```

The report includes:

```text
Precision
Recall
F1-score
Support
```

for each class.

It also provides averaged results.

This gives a more detailed view of classification performance than accuracy alone.

---

# 🎯 Precision

Precision answers:

> When the model predicts a class, how often is that prediction correct?

Formula:

```text
Precision =
TP
------------
TP + FP
```

High precision means fewer false-positive predictions for that class.

---

# 🔍 Recall

Recall answers:

> Of the observations that actually belong to a class, how many did the model find?

Formula:

```text
Recall =
TP
------------
TP + FN
```

High recall means fewer false negatives.

---

# ⚖️ F1-Score

F1-score combines precision and recall.

Formula:

```text
F1 =
2 × Precision × Recall
-----------------------
Precision + Recall
```

A high F1-score generally requires both precision and recall to be strong.

---

# 👥 Support

Support is the number of actual observations belonging to each class in the evaluated dataset.

For the classification report, support values correspond to the test set.

Together, the support values equal the number of test observations.

---

# 🎲 Prediction Probabilities

QDA can return class probabilities using:

```python
y_proba = qda.predict_proba(
    X_test_scaled
)
```

For every test observation, the model returns three probabilities.

Conceptually:

```text
Class 0 → 0.10
Class 1 → 0.85
Class 2 → 0.05
```

The predicted class would be Class 1.

---

# 🔐 Probability Interpretation

Probability output provides more information than a class label alone.

For example:

```text
Prediction = Class 1
```

does not tell us how strongly the model prefers Class 1.

But:

```text
Class 0 → 0.02
Class 1 → 0.96
Class 2 → 0.02
```

shows a strong Class 1 preference.

---

# 🧾 Actual vs Predicted

Cell 18 creates:

```python
comparison_df = pd.DataFrame({
    "Actual Class": y_test.values,
    "Predicted Class": y_pred
})
```

This creates an observation-level comparison.

A prediction is correct when:

```text
Actual Class == Predicted Class
```

It is incorrect when:

```text
Actual Class != Predicted Class
```

This table is useful for manually inspecting model predictions.

---

# 📊 Final Evaluation Summary

Cell 19 prints:

```text
Training Samples
Testing Samples
Number of Features
Number of Classes
Accuracy
Accuracy Percentage
```

For this notebook:

```text
Training Samples → 142
Testing Samples  → 36
Features         → 13
Classes          → 3
```

The accuracy is taken from Cell 13.

---

# 📖 20-Cell Notebook Explanation

The implementation contains exactly 20 cells.

The first six cells focus on dataset loading and exploration.

Cells 7–10 prepare the supervised-learning data and create the QDA model.

Cells 11–18 perform training, prediction, and evaluation.

Cells 19–20 provide the final summary and conclusion.

---

# 🔹 Cell 1 — Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
```

### Explanation

This cell imports NumPy, Pandas, Matplotlib, Seaborn, and the required Scikit-learn tools.

`load_wine` loads the dataset.

`train_test_split` creates training and testing sets.

`StandardScaler` performs feature standardization.

`QuadraticDiscriminantAnalysis` provides the QDA classifier.

---

# 🔹 Cell 2 — Load Wine Dataset

```python
wine = load_wine()

print("Wine dataset loaded successfully!")
```

### Explanation

This cell loads the Wine dataset directly from Scikit-learn.

The `wine` object contains the features, feature names, target values, and target names.

---

# 🔹 Cell 3 — Create DataFrame

```python
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df["target"] = wine.target

print("Dataset Shape:", df.shape)

display(df.head())
```

### Explanation

The Wine feature matrix is converted into a Pandas DataFrame.

The target is added as the `target` column.

The resulting DataFrame contains:

```text
178 rows
14 columns
```

The first five observations are displayed.

---

# 🔹 Cell 4 — Dataset Information

```python
print("Dataset Information:
")

df.info()
```

### Explanation

`df.info()` displays the structure of the DataFrame.

It provides column names, data types, non-null counts, and other structural information.

---

# 🔹 Cell 5 — Statistical Summary

```python
display(df.describe())
```

### Explanation

This cell displays descriptive statistics.

The summary includes count, mean, standard deviation, minimum, quartiles, and maximum.

It provides an overview of feature distributions and numerical scales.

---

# 🔹 Cell 6 — Check Missing Values

```python
print("Missing Values:
")

display(df.isnull().sum())
```

### Explanation

This checks every column for missing values.

The Wine dataset contains no missing observations requiring imputation.

---

# 🔹 Cell 7 — Define Independent and Dependent Variables

```python
X = df.drop("target", axis=1)
y = df["target"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("
Features:")
display(X.head())

print("
Target:")
display(y.head())
```

### Explanation

The target is removed from the feature matrix.

Therefore:

```text
X → 178 × 13
y → 178
```

This creates the standard supervised-learning structure.

---

# 🔹 Cell 8 — Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
```

### Explanation

The data is split into training and testing subsets.

The test size is 20%.

The split is reproducible with `random_state=42`.

Stratification preserves class proportions as much as possible.

Expected shapes:

```text
X_train → 142 × 13
X_test  → 36 × 13
```

---

# 🔹 Cell 9 — Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training data scaled successfully!")
print("Testing data scaled successfully!")

print("
Scaled Training Shape:", X_train_scaled.shape)
print("Scaled Testing Shape:", X_test_scaled.shape)
```

### Explanation

Training data is used to fit the scaler.

The test data is transformed using the same scaler.

This avoids fitting preprocessing information independently on the test set.

---

# 🔹 Cell 10 — Create QDA Model

```python
qda = QuadraticDiscriminantAnalysis()

print("QDA Model Created Successfully!")
print(qda)
```

### Explanation

This creates the QDA classifier object.

The model is not trained until Cell 11.

---

# 🔹 Cell 11 — Train QDA Model

```python
qda.fit(X_train_scaled, y_train)

print("QDA model trained successfully!")
```

### Explanation

The model learns from the scaled training features and their labels.

It estimates the class-specific parameters required for classification.

---

# 🔹 Cell 12 — Make Predictions

```python
y_pred = qda.predict(X_test_scaled)

print("Predictions generated successfully!")

print("
Predicted Classes:")
print(y_pred)
```

### Explanation

The trained QDA model predicts the classes of the test observations.

The resulting predictions are stored in `y_pred`.

---

# 🔹 Cell 13 — Calculate Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("QDA Accuracy:", accuracy)
print("QDA Accuracy (%):", accuracy * 100)
```

### Explanation

This calculates the percentage of test observations classified correctly.

The result is stored in `accuracy`.

---

# 🔹 Cell 14 — Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")

display(pd.DataFrame(
    cm,
    index=[
        "Actual Class 0",
        "Actual Class 1",
        "Actual Class 2"
    ],
    columns=[
        "Predicted Class 0",
        "Predicted Class 1",
        "Predicted Class 2"
    ]
))
```

### Explanation

This creates a 3 × 3 confusion matrix.

It shows how actual classes correspond to predicted classes.

---

# 🔹 Cell 15 — Visualize Confusion Matrix

```python
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Class 0",
        "Class 1",
        "Class 2"
    ],
    yticklabels=[
        "Class 0",
        "Class 1",
        "Class 2"
    ]
)

plt.title("QDA Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.show()
```

### Explanation

The confusion matrix is displayed as a heatmap.

This makes correct and incorrect predictions visually easier to identify.

---

# 🔹 Cell 16 — Classification Report

```python
from sklearn.metrics import classification_report

print("QDA Classification Report:
")

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

### Explanation

This generates precision, recall, F1-score, and support for each class.

It provides class-level performance information.

---

# 🔹 Cell 17 — Prediction Probabilities

```python
y_proba = qda.predict_proba(X_test_scaled)

probability_df = pd.DataFrame(
    y_proba,
    columns=[
        "Class 0 Probability",
        "Class 1 Probability",
        "Class 2 Probability"
    ]
)

display(probability_df.head(10))
```

### Explanation

This calculates class probabilities for the test observations.

Each observation receives three probability values because the dataset has three classes.

---

# 🔹 Cell 18 — Compare Actual vs Predicted

```python
comparison_df = pd.DataFrame({
    "Actual Class": y_test.values,
    "Predicted Class": y_pred
})

display(comparison_df.head(20))
```

### Explanation

This creates an observation-level table comparing actual and predicted labels.

It makes individual correct and incorrect predictions easy to inspect.

---

# 🔹 Cell 19 — Final Evaluation Summary

```python
print("========== QDA MODEL SUMMARY ==========")

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("Number of Features:", X.shape[1])
print("Number of Classes :", len(np.unique(y)))

print("
Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
```

### Explanation

This cell provides a compact summary of the experiment.

The project uses 142 training observations and 36 testing observations.

There are 13 features and 3 target classes.

The accuracy is taken from the trained QDA model.

---

# 🔹 Cell 20 — Final Conclusion

```python
print("========== FINAL CONCLUSION ==========")

print(
    f"QDA achieved an accuracy of {accuracy * 100:.2f}% "
    "on the test dataset."
)

print(
    "Quadratic Discriminant Analysis was successfully "
    "trained and evaluated on the Wine dataset."
)
```

### Explanation

This is the final notebook cell.

It dynamically prints the actual accuracy calculated by the model.

It then confirms successful QDA training and evaluation.

---

# 📋 Complete 20-Cell Structure

```text
Cell 1  → Import Libraries
Cell 2  → Load Wine Dataset
Cell 3  → Create DataFrame
Cell 4  → Dataset Information
Cell 5  → Statistical Summary
Cell 6  → Check Missing Values
Cell 7  → Define X and y
Cell 8  → Train-Test Split
Cell 9  → Feature Scaling
Cell 10 → Create QDA Model

Cell 11 → Train QDA Model
Cell 12 → Make Predictions
Cell 13 → Calculate Accuracy
Cell 14 → Confusion Matrix
Cell 15 → Confusion Matrix Visualization
Cell 16 → Classification Report
Cell 17 → Prediction Probabilities
Cell 18 → Actual vs Predicted
Cell 19 → Final Evaluation Summary
Cell 20 → Final Conclusion
```

---

# 🔄 Complete QDA Classification Pipeline

```text
Wine Dataset
      ↓
Data Exploration
      ↓
Feature / Target Separation
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
QDA Model
      ↓
Training
      ↓
Prediction
      ↓
Accuracy
      ↓
Confusion Matrix
      ↓
Classification Report
      ↓
Probability Analysis
      ↓
Actual vs Predicted
      ↓
Final Summary
```

---

# 🔥 Advantages of QDA

- Allows separate covariance matrices for classes.
- Can model quadratic decision boundaries.
- Supports multiclass classification.
- Provides probabilistic class predictions.
- More flexible than classical LDA.
- Has a clear statistical interpretation.
- Can work well when class distributions have different covariance structures.

---

# ❌ Limitations of QDA

- Estimates more parameters than LDA.
- Can require more observations for stable covariance estimates.
- Can overfit when data is limited.
- Sensitive to covariance estimation.
- Assumes class distributions are approximately Gaussian.
- May not perform well when the Gaussian assumption is inappropriate.
- More complex than a simple linear classifier.

---

# 🌍 Applications

QDA can be applied to:

- Medical classification
- Chemical classification
- Biological classification
- Pattern recognition
- Customer classification
- Quality control
- Financial classification
- Sensor classification
- Scientific classification
- Multiclass classification

---

# 🛠️ Technologies Used

```text
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

The LDA project documentation uses the same general Python and Scikit-learn ecosystem. fileciteturn15file7L1165-L1173

---

# 📁 Project Structure

```text
QDA-Classification/
│
├── QDA Classification.ipynb
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

The Wine dataset is loaded directly from Scikit-learn.

No separate CSV dataset is required.

---

# 📦 Requirements

```text
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

Install them with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

---

# ▶️ How to Run

Open the project folder.

Install the required packages.

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
QDA Classification.ipynb
```

Run Cell 1 through Cell 20 sequentially.

Running the notebook in order ensures that all variables are created before they are used.

---

# 🧠 Important Variables

| Variable | Meaning |
|---|---|
| `wine` | Wine dataset object |
| `df` | Complete DataFrame |
| `X` | Feature matrix |
| `y` | Target vector |
| `X_train` | Training features |
| `X_test` | Testing features |
| `y_train` | Training labels |
| `y_test` | Testing labels |
| `scaler` | StandardScaler object |
| `X_train_scaled` | Scaled training data |
| `X_test_scaled` | Scaled testing data |
| `qda` | QDA model |
| `y_pred` | Predicted test classes |
| `accuracy` | Classification accuracy |
| `cm` | Confusion matrix |
| `y_proba` | Prediction probabilities |
| `probability_df` | Probability DataFrame |
| `comparison_df` | Actual vs predicted DataFrame |

---

# 🎓 Learning Outcomes

After completing this project, you should be able to:

- Understand Machine Learning.
- Understand supervised learning.
- Understand classification.
- Explain QDA.
- Understand class-specific covariance.
- Understand Gaussian class distributions.
- Understand class means.
- Understand class priors.
- Understand Bayes-based classification.
- Explain quadratic decision boundaries.
- Compare QDA and LDA.
- Load the Wine dataset.
- Separate features and target.
- Perform train-test splitting.
- Use stratification.
- Standardize data correctly.
- Train a QDA model.
- Generate predictions.
- Calculate accuracy.
- Create a confusion matrix.
- Interpret precision.
- Interpret recall.
- Interpret F1-score.
- Generate prediction probabilities.
- Compare actual and predicted classes.
- Evaluate a multiclass classifier.
- Understand QDA advantages and limitations.

---

# 🧭 QDA in the Machine Learning Roadmap

QDA belongs to:

```text
Machine Learning
      ↓
Supervised Learning
      ↓
Classification
      ↓
Discriminant Analysis
      ├── LDA
      └── QDA
```

The key relationship is:

```text
LDA
→ More restrictive covariance assumption

QDA
→ More flexible covariance assumption
```

The LDA documentation identifies LDA as a supervised classification algorithm and describes its common-covariance assumption. fileciteturn15file3L564-L583

---

# 🧪 Possible Future Improvements

The current implementation intentionally contains only 20 cells.

Future versions could add:

```text
Cross-validation
ROC/AUC
Class-wise accuracy
LDA vs QDA comparison
Logistic Regression comparison
KNN comparison
Decision Tree comparison
Probability confidence analysis
Hyperparameter experiments
```

These additions are outside the current 20-cell implementation.

---

# 🏁 Conclusion

Quadratic Discriminant Analysis is an important supervised classification algorithm.

It models each class using its own Gaussian distribution and covariance matrix.

The ability to estimate separate covariance matrices gives QDA more flexibility than classical LDA.

The project uses the Wine Recognition Dataset from Scikit-learn.

The dataset contains:

```text
178 observations
13 numerical features
3 classes
```

The workflow begins with dataset loading and exploration.

The features and target are separated.

The dataset is divided into training and testing subsets using an 80/20 split.

The split uses stratification and `random_state=42`.

The training features are standardized using `StandardScaler`.

The test features are transformed using the same fitted scaler.

A QDA classifier is created.

The classifier is trained using the scaled training data.

Predictions are generated for the test observations.

The model is evaluated using accuracy, confusion matrix, and classification report.

Prediction probabilities provide additional information about class membership.

The actual-versus-predicted table allows observation-level inspection.

The final summary reports the overall model configuration and accuracy.

The entire implementation is contained in exactly 20 cells.

---

# 🔄 Final QDA Pipeline

```text
Wine Dataset
       ↓
13 Features + 3 Classes
       ↓
Feature / Target Separation
       ↓
80/20 Train-Test Split
       ↓
Stratification
       ↓
StandardScaler
       ↓
QDA
       ↓
Training
       ↓
Prediction
       ↓
Accuracy
       ↓
Confusion Matrix
       ↓
Classification Report
       ↓
Prediction Probabilities
       ↓
Actual vs Predicted
       ↓
Final Evaluation
```

---

# ⭐ Final Takeaway

The most important difference to remember is:

```text
LDA
→ Common covariance structure
→ Linear decision boundary

QDA
→ Class-specific covariance structures
→ Quadratic decision boundary
```

QDA therefore provides additional flexibility when different classes have different covariance structures.

For this project:

```text
Wine Features
       ↓
Standardization
       ↓
QDA
       ↓
3-Class Classification
       ↓
Model Evaluation
```

The project demonstrates the complete QDA classification workflow in exactly 20 cells.



# 📎 Project Summary

```text
Algorithm
Quadratic Discriminant Analysis

Abbreviation
QDA

Learning Type
Supervised Learning

Task
Multiclass Classification

Dataset
Scikit-learn Wine Recognition Dataset

Observations
178

Features
13

Classes
3

Train-Test Split
80% / 20%

Random State
42

Stratification
Yes

Feature Scaling
StandardScaler

Model
QuadraticDiscriminantAnalysis

Evaluation
Accuracy
Confusion Matrix
Classification Report
Prediction Probabilities

Notebook Cells
20
```

---
---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project is part of a practical Machine Learning learning journey covering supervised learning, classification, discriminant analysis, clustering, and other Machine Learning algorithms.

---

