# 🟠 Linear Discriminant Analysis (LDA) — Classification

A comprehensive Machine Learning project demonstrating **Linear Discriminant Analysis (LDA)** as a **Supervised Learning classification algorithm** using the **Wine Recognition Dataset** from Scikit-learn.

In this project, LDA is treated primarily as a **classification model**. The model is trained using labeled wine samples and evaluated using standard classification metrics.

LDA can also perform supervised dimensionality reduction, so the final cells additionally demonstrate the LDA discriminant space.

---

# 📑 Table of Contents

- Machine Learning
- Supervised Learning
- Classification
- What is LDA?
- LDA as Classification
- How LDA Works
- Dataset
- Features and Target
- Train-Test Split
- Feature Scaling
- LDA Model
- Predictions
- Accuracy
- Classification Report
- Confusion Matrix
- Cross-Validation
- Class Probabilities
- Decision Scores
- LDA Transformation
- PCA vs LDA
- Advantages
- Limitations
- Applications
- 30 Cell Explanation
- Learning Outcomes
- Conclusion

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that allows computers to learn patterns from data and use those patterns to make predictions or decisions.

A simplified structure is:

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

This project places **LDA under Supervised Learning → Classification**.

---

# 🧠 What is Supervised Learning?

Supervised Learning uses labeled data.

```text
X → Input Features
y → Target / Label
```

The model learns the relationship between the input features and known target classes.

For this project:

```text
X → Wine chemical features
y → Wine class
```

---

# 🏷️ What is Classification?

Classification is a supervised learning task where the target represents categories.

Examples:

```text
Email → Spam / Not Spam
Patient → Disease / No Disease
Transaction → Fraud / Not Fraud
Wine → Class 0 / Class 1 / Class 2
```

The goal of this project is:

```text
Wine Features
      ↓
LDA Classifier
      ↓
Predicted Wine Class
```

---

# 🟠 What is LDA?

**LDA** stands for:

> **Linear Discriminant Analysis**

LDA is a supervised statistical and Machine Learning technique that can be used for classification.

It finds linear combinations of features that help distinguish known classes.

For classification, LDA learns class-specific distributions and uses them to determine the most likely class for a new observation.

---

# 🎯 LDA as a Classification Algorithm

The notebook treats LDA as:

```text
Supervised Learning
        ↓
Classification
        ↓
Linear Discriminant Analysis
```

Training:

```text
X_train + y_train
        ↓
      LDA
        ↓
   Learned Model
```

Prediction:

```text
X_test
   ↓
LDA Model
   ↓
y_pred
```

---

# 🔄 How LDA Works

LDA considers class distributions and tries to find directions that separate classes effectively.

The general idea is to:

1. Estimate class means.
2. Consider within-class variation.
3. Consider between-class variation.
4. Find discriminant directions.
5. Use the learned structure for classification.

The objective can be summarized as:

```text
Maximize:
Between-Class Variation

Minimize:
Within-Class Variation
```

---

# 📊 Within-Class and Between-Class Variation

### Within-Class Variation

This represents how spread out observations are inside the same class.

LDA attempts to keep this relatively small.

### Between-Class Variation

This represents how separated different class groups are.

LDA attempts to make this large.

Conceptually:

```text
Good LDA Solution

Within-Class Spread
        ↓
      Small

Between-Class Separation
        ↓
       Large
```

---

# 🧪 Dataset Information

## Wine Recognition Dataset

The dataset is loaded directly from Scikit-learn:

```python
from sklearn.datasets import load_wine

wine = load_wine()
```

No external CSV file is required.

The standard dataset contains:

```text
178 observations
13 numerical features
3 classes
```

---

# 🍷 Wine Dataset Features

The 13 input features are:

```text
1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline
```

---

# 🏷️ Target Classes

The dataset contains three target classes:

```text
Class 0
Class 1
Class 2
```

The target is obtained from:

```python
wine.target
```

and class names from:

```python
wine.target_names
```

---

# 🎯 Features and Target

The notebook creates:

```python
X = df.drop("Target", axis=1)
y = df["Target"]
```

Therefore:

```text
X → 13 input features
y → Wine class
```

Unlike PCA, the target is required here because LDA is being used as a supervised classifier.

---

# ✂️ Train-Test Split

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

This creates approximately:

```text
80% → Training
20% → Testing
```

`stratify=y` helps preserve class proportions in both sets.

---

# ⚖️ Feature Scaling

The project uses:

```python
StandardScaler()
```

The scaler is fitted only on training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

and then applied to the test data:

```python
X_test_scaled = scaler.transform(X_test)
```

This prevents test-set information from being used when calculating the scaling parameters.

---

# 📐 Standardization

The standardization formula is:

```text
z = (x − mean) / standard deviation
```

Scaling places the features on comparable numerical scales.

---

# 🧠 LDA Model

The classifier is created using:

```python
lda = LinearDiscriminantAnalysis()
```

It is trained using:

```python
lda.fit(
    X_train_scaled,
    y_train
)
```

The model learns the relationship between the features and the wine classes.

---

# 🔮 Predictions

Predictions are generated using:

```python
y_pred = lda.predict(X_test_scaled)
```

`y_pred` contains the predicted class for every test observation.

---

# 🎯 Accuracy

Accuracy is calculated using:

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)
```

Conceptually:

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

---

# 📋 Classification Report

The report is generated using:

```python
classification_report(
    y_test,
    y_pred,
    target_names=wine.target_names
)
```

It provides:

```text
Precision
Recall
F1-Score
Support
```

---

# 🎯 Precision

Precision answers:

> Of the observations predicted as a class, how many actually belong to that class?

```text
Precision =
True Positives
-------------------------
True Positives + False Positives
```

---

# 🔍 Recall

Recall answers:

> Of all observations belonging to a class, how many were correctly identified?

```text
Recall =
True Positives
-------------------------
True Positives + False Negatives
```

---

# ⚖️ F1-Score

F1-score combines precision and recall:

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

---

# 🔢 Support

Support is the number of actual observations belonging to each class in the evaluation set.

---

# 🟦 Confusion Matrix

A confusion matrix summarizes classification results.

For three classes:

```text
             Predicted
           0    1    2

Actual 0   ✓    ?    ?
Actual 1   ?    ✓    ?
Actual 2   ?    ?    ✓
```

Correct predictions appear on the diagonal.

Incorrect predictions appear outside the diagonal.

---

# 🔄 Cross-Validation

The project uses five-fold cross-validation:

```python
cv_scores = cross_val_score(
    lda,
    X_train_scaled,
    y_train,
    cv=5
)
```

The mean validation score is calculated with:

```python
cv_scores.mean()
```

Cross-validation evaluates the classifier over multiple training-validation splits.

---

# 🔢 Class Probabilities

LDA can estimate class probabilities:

```python
y_proba = lda.predict_proba(
    X_test_scaled
)
```

For example, conceptually:

```text
Sample
 ├── Class 0 → 0.05
 ├── Class 1 → 0.90
 └── Class 2 → 0.05
```

The class with the highest probability becomes the predicted class.

---

# 🧮 Decision Scores

The notebook uses:

```python
decision_scores = lda.decision_function(
    X_test_scaled
)
```

Decision scores provide additional information about the model's class decisions.

---

# 📉 LDA as Supervised Dimensionality Reduction

Although the project treats LDA primarily as classification, LDA can also transform data into a discriminant space:

```python
X_test_lda = lda.transform(X_test_scaled)
```

Unlike PCA, LDA uses class labels to find directions that improve class separation.

---

# 📐 Maximum LDA Components

The number of LDA components is limited by:

```text
min(Number of Features, Number of Classes - 1)
```

For Wine:

```text
min(13, 3 - 1)
= 2
```

Therefore the data can be represented using:

```text
LD1
LD2
```

---

# 🎨 LDA Discriminant Visualization

The final notebook cell plots:

```text
Linear Discriminant 1
Linear Discriminant 2
```

and colors observations according to their actual class.

This provides a visual view of supervised class separation.

---

# 🆚 PCA vs LDA

| Feature | PCA | LDA |
|---|---|---|
| Learning Type | Unsupervised | Supervised |
| Uses Target | No | Yes |
| Main Goal | Preserve variance | Separate classes |
| Classification | No | Yes |
| Maximum Components | Up to feature count | Up to classes - 1 |
| Focus | Overall variation | Class separation |

In this roadmap:

```text
PCA
→ Unsupervised Dimensionality Reduction

LDA
→ Supervised Classification
```

---

# 🆚 LDA vs Other Classification Models

| Model | Main Idea |
|---|---|
| Logistic Regression | Linear probability-based classification |
| KNN | Neighbor-based classification |
| Decision Tree | Rule-based splitting |
| Random Forest | Ensemble of decision trees |
| SVM | Maximum-margin classification |
| LDA | Discriminant/class-distribution-based classification |

---

# 🔥 Advantages of LDA

- Supervised classification algorithm.
- Computationally efficient.
- Supports multiclass classification.
- Provides class probabilities.
- Produces linear discriminant directions.
- Can also perform supervised dimensionality reduction.
- Can work well on relatively small datasets.

---

# ❌ Limitations of LDA

- Relies on assumptions about class distributions.
- Classical LDA assumes similar covariance structures across classes.
- Linear boundaries may not work well for strongly nonlinear relationships.
- Performance can suffer when assumptions are strongly violated.
- Feature preprocessing may still be important.
- Complex nonlinear problems may require other classifiers.

---

# 📐 Important LDA Assumptions

Classical LDA generally assumes:

### 1. Continuous Features

The traditional formulation naturally handles continuous numerical predictors.

### 2. Approximately Normal Class Distributions

Feature distributions within classes are generally assumed to be reasonably close to multivariate normal.

### 3. Similar Covariance Matrices

Classes are generally assumed to share a common covariance structure.

### 4. Independent Observations

Observations are generally expected to be independent.

---

# 🌍 Real-World Applications

LDA classification can be applied to:

- Medical classification
- Pattern recognition
- Biological classification
- Quality control
- Customer classification
- Face recognition
- Classification of measured chemical properties
- Other multiclass classification tasks

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 📁 Project Structure

```text
LDA-Classification/
│
├── LDA Classification.ipynb
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

The Wine dataset is loaded directly from Scikit-learn.

---

# 🔄 Complete Project Workflow

```text
Wine Dataset
      ↓
Data Exploration
      ↓
Separate Features and Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Create LDA Classifier
      ↓
Train LDA
      ↓
Predict Test Data
      ↓
Accuracy
      ↓
Classification Report
      ↓
Confusion Matrix
      ↓
Cross-Validation
      ↓
Class Probabilities
      ↓
Class-Wise Accuracy
      ↓
Decision Scores
      ↓
LDA Transformation
      ↓
Discriminant Visualization
```

---

# 📖 Step-by-Step Notebook Explanation

The following sections explain all **30 cells** of the LDA classification notebook in the same order as the implementation.

---

## 🔹 Cell 1 — Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")
```

### Explanation

This imports the dataset loader, preprocessing tools, LDA classifier, evaluation metrics, and visualization libraries.

---

## 🔹 Cell 2 — Load Wine Dataset

```python
wine = load_wine()

print("Wine dataset loaded successfully!")
```

### Explanation

This loads the Wine Recognition Dataset directly from Scikit-learn.

---

## 🔹 Cell 3 — Create DataFrame

```python
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df["Target"] = wine.target

df.head()
```

### Explanation

The 13 numerical features are converted into a DataFrame and the target class is added as `Target`.

---

## 🔹 Cell 4 — Dataset Information

```python
df.info()
```

### Explanation

Displays the DataFrame structure, data types, and non-null counts.

---

## 🔹 Cell 5 — Statistical Summary

```python
df.describe()
```

### Explanation

Displays descriptive statistics for the dataset.

---

## 🔹 Cell 6 — Check Missing Values

```python
df.isnull().sum()
```

### Explanation

Checks whether any feature or target values are missing.

---

## 🔹 Cell 7 — Dataset Dimensions

```python
print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
```

### Explanation

The standard Wine dataset has:

```text
178 rows
14 columns
```

The 14 columns are:

```text
13 features + 1 target
```

---

## 🔹 Cell 8 — Display Target Classes

```python
print("Target Classes:")
print(wine.target_names)

print("
Unique Target Values:")
print(np.unique(wine.target))
```

### Explanation

Displays the available target classes and their numerical labels.

---

## 🔹 Cell 9 — Separate Features and Target

```python
X = df.drop("Target", axis=1)
y = df["Target"]

print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)
```

### Explanation

`X` contains the 13 input features and `y` contains the target class.

---

## 🔹 Cell 10 — Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)

print("Training Target   :", y_train.shape)
print("Testing Target    :", y_test.shape)
```

### Explanation

The dataset is divided into training and testing subsets using an 80/20 split.

---

## 🔹 Cell 11 — Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training data scaled successfully!")
print("Testing data scaled successfully!")
```

### Explanation

The scaler learns parameters only from the training data and then transforms both training and testing data.

---

## 🔹 Cell 12 — Create LDA Classifier

```python
lda = LinearDiscriminantAnalysis()

print("LDA classifier created successfully!")
```

### Explanation

Creates the LDA classification model.

---

## 🔹 Cell 13 — Train LDA Model

```python
lda.fit(X_train_scaled, y_train)

print("LDA classifier trained successfully!")
```

### Explanation

The model learns from the scaled training features and their known labels.

---

## 🔹 Cell 14 — Make Predictions

```python
y_pred = lda.predict(X_test_scaled)

print("Predictions generated successfully!")
```

### Explanation

Generates predicted classes for the unseen test data.

---

## 🔹 Cell 15 — Display Predictions

```python
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

display(results.head(20))
```

### Explanation

Creates a table comparing actual and predicted classes.

---

## 🔹 Cell 16 — Calculate Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("LDA Accuracy:", accuracy)
print("LDA Accuracy (%):", accuracy * 100)
```

### Explanation

Calculates the percentage of test observations classified correctly.

---

## 🔹 Cell 17 — Classification Report

```python
report = classification_report(
    y_test,
    y_pred,
    target_names=wine.target_names
)

print("Classification Report:
")
print(report)
```

### Explanation

Displays precision, recall, F1-score, and support for each class.

---

## 🔹 Cell 18 — Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)
```

### Explanation

Creates the confusion matrix containing correct and incorrect class predictions.

---

## 🔹 Cell 19 — Visualize Confusion Matrix

```python
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=wine.target_names,
    yticklabels=wine.target_names
)

plt.title("LDA Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()
```

### Explanation

Displays the confusion matrix as a heatmap.

---

## 🔹 Cell 20 — Display LDA Classifier Parameters

```python
print("LDA Model:")
print(lda)

print("
Number of Classes:", lda.classes_)
print("Number of Components:", lda.scalings_.shape[1])
```

### Explanation

Displays the model and useful learned dimensions/classes.

---

## 🔹 Cell 21 — Cross-Validation

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    lda,
    X_train_scaled,
    y_train,
    cv=5
)

print("Cross-Validation Scores:")
print(cv_scores)

print("
Mean Cross-Validation Accuracy:",
      cv_scores.mean())
```

### Explanation

Performs five-fold cross-validation on the training data.

---

## 🔹 Cell 22 — LDA Class Probabilities

```python
y_proba = lda.predict_proba(X_test_scaled)

print("Class Probabilities:")
display(pd.DataFrame(
    y_proba,
    columns=wine.target_names
).head(10))
```

### Explanation

Returns estimated probabilities for each possible class.

---

## 🔹 Cell 23 — Probability-Based Predictions

```python
probability_results = pd.DataFrame(
    y_proba,
    columns=[
        f"Probability_{name}"
        for name in wine.target_names
    ]
)

probability_results["Actual"] = y_test.values
probability_results["Predicted"] = y_pred

display(probability_results.head(10))
```

### Explanation

Combines probabilities with actual and predicted classes.

---

## 🔹 Cell 24 — Check Correct and Incorrect Predictions

```python
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results["Correct"] = (
    results["Actual"] == results["Predicted"]
)

print("Correct Predictions:",
      results["Correct"].sum())

print("Incorrect Predictions:",
      (~results["Correct"]).sum())
```

### Explanation

Counts correctly and incorrectly classified test observations.

---

## 🔹 Cell 25 — Prediction Accuracy by Class

```python
class_accuracy = {}

for class_value in np.unique(y_test):
    class_mask = y_test.values == class_value

    class_accuracy[class_value] = (
        y_pred[class_mask] == class_value
    ).mean()

print("Class-wise Accuracy:
")

for class_value, score in class_accuracy.items():
    print(
        wine.target_names[class_value],
        ":",
        score
    )
```

### Explanation

Calculates classification accuracy separately for each class.

---

## 🔹 Cell 26 — Visualize LDA Predictions

```python
plt.figure(figsize=(10, 6))

plt.scatter(
    range(len(y_test)),
    y_test,
    label="Actual",
    marker="o"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    label="Predicted",
    marker="x"
)

plt.title("LDA Actual vs Predicted Classes")
plt.xlabel("Test Sample Index")
plt.ylabel("Wine Class")

plt.legend()
plt.grid()
plt.show()
```

### Explanation

Visualizes actual and predicted classes across test samples.

---

## 🔹 Cell 27 — LDA Decision Scores

```python
decision_scores = lda.decision_function(
    X_test_scaled
)

print("LDA Decision Scores:")

display(pd.DataFrame(
    decision_scores,
    columns=[
        f"Class_{class_value}"
        for class_value in lda.classes_
    ]
).head(10))
```

### Explanation

Provides the LDA decision scores associated with class decisions.

---

## 🔹 Cell 28 — LDA Explained Variance Ratio

```python
if hasattr(lda, "explained_variance_ratio_"):
    print("LDA Explained Variance Ratio:")
    print(lda.explained_variance_ratio_)

    print(
        "
Total Variance Represented:",
        lda.explained_variance_ratio_.sum()
    )
else:
    print(
        "Explained variance ratio is not available "
        "for this LDA configuration."
    )
```

### Explanation

Checks whether the LDA implementation exposes the explained variance ratio and displays it when available.

---

## 🔹 Cell 29 — LDA Transformation to 2 Dimensions

```python
X_test_lda = lda.transform(X_test_scaled)

print("Original Test Shape :", X_test_scaled.shape)
print("LDA Test Shape      :", X_test_lda.shape)
```

### Explanation

Transforms test observations into the LDA discriminant space.

Because there are three classes:

```text
Maximum components = 3 - 1 = 2
```

---

## 🔹 Cell 30 — Visualize LDA Discriminant Space

```python
plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=X_test_lda[:, 0],
    y=X_test_lda[:, 1],
    hue=y_test.values,
    palette="deep",
    s=80
)

plt.title("LDA Discriminant Space")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")

plt.grid()
plt.legend(title="Wine Class")

plt.show()
```

### Explanation

Visualizes the test observations in the two-dimensional LDA discriminant space.

This shows how the supervised transformation separates the known wine classes.

---

# 📊 Complete LDA Classification Pipeline

```text
Wine Dataset
      ↓
13 Features + Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
LDA Classifier
      ↓
Training
      ↓
Predictions
      ↓
Accuracy
      ↓
Classification Report
      ↓
Confusion Matrix
      ↓
Cross-Validation
      ↓
Class Probabilities
      ↓
Class-Wise Analysis
      ↓
Decision Scores
      ↓
LDA Transformation
      ↓
Visualization
```

---

# 🆚 PCA vs LDA

```text
PCA
→ Unsupervised
→ Does not use labels
→ Maximizes overall variance

LDA
→ Supervised
→ Uses labels
→ Maximizes class separation
→ Can perform classification
```

This project specifically treats LDA as:

```text
Supervised Machine Learning
        ↓
Classification
```

---

# 🔥 Advantages of LDA

- Supervised classification algorithm.
- Computationally efficient.
- Supports multiclass classification.
- Provides class probabilities.
- Produces linear discriminant directions.
- Can also perform supervised dimensionality reduction.
- Useful for relatively small and medium-sized datasets.

---

# ❌ Limitations of LDA

- Relies on assumptions about class distributions.
- Classical LDA assumes similar covariance structures across classes.
- Linear boundaries may not work well for strongly nonlinear relationships.
- Performance can suffer when assumptions are strongly violated.
- Feature preprocessing may still be important.
- Complex nonlinear problems may require other classifiers.

---

# 🌍 Applications

LDA classification can be used for:

- Medical classification
- Pattern recognition
- Biological classification
- Quality control
- Customer classification
- Face recognition
- Chemical classification
- Multiclass classification tasks

---

# 📁 Complete Notebook Structure

```text
Cell 1  → Import Libraries
Cell 2  → Load Wine Dataset
Cell 3  → Create DataFrame
Cell 4  → Dataset Information
Cell 5  → Statistical Summary
Cell 6  → Check Missing Values
Cell 7  → Dataset Dimensions
Cell 8  → Display Target Classes
Cell 9  → Separate Features and Target
Cell 10 → Train-Test Split

Cell 11 → Feature Scaling
Cell 12 → Create LDA Classifier
Cell 13 → Train LDA Model
Cell 14 → Make Predictions
Cell 15 → Display Predictions
Cell 16 → Calculate Accuracy
Cell 17 → Classification Report
Cell 18 → Confusion Matrix
Cell 19 → Visualize Confusion Matrix
Cell 20 → Display LDA Parameters

Cell 21 → Cross-Validation
Cell 22 → Class Probabilities
Cell 23 → Probability-Based Predictions
Cell 24 → Correct / Incorrect Predictions
Cell 25 → Class-Wise Accuracy
Cell 26 → Actual vs Predicted Visualization
Cell 27 → Decision Scores
Cell 28 → LDA Explained Variance Ratio
Cell 29 → LDA Transformation
Cell 30 → LDA Discriminant Visualization
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand Supervised Learning.
- Understand classification.
- Understand Linear Discriminant Analysis.
- Use LDA as a classification algorithm.
- Understand how LDA uses target labels.
- Perform train-test splitting.
- Use stratified splitting.
- Standardize training and testing data correctly.
- Train an LDA classifier.
- Generate class predictions.
- Calculate classification accuracy.
- Generate a classification report.
- Interpret precision, recall, and F1-score.
- Interpret a confusion matrix.
- Perform cross-validation.
- Generate class probabilities.
- Calculate class-wise accuracy.
- Inspect decision scores.
- Use LDA for supervised dimensionality reduction.
- Visualize the LDA discriminant space.
- Compare LDA with PCA.
- Understand important LDA assumptions.

---

# 🚀 Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the LDA classification notebook and run the cells sequentially.

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

---

# 🏁 Conclusion

Linear Discriminant Analysis is an important supervised technique that can be used for classification and supervised dimensionality reduction.

In this project, LDA is treated primarily as:

```text
Supervised Learning
       ↓
Classification
```

The Wine Recognition Dataset contains:

```text
178 observations
13 numerical features
3 classes
```

The complete workflow is:

```text
Load Dataset
     ↓
Explore Data
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Create LDA Classifier
     ↓
Train Model
     ↓
Predict Classes
     ↓
Evaluate Performance
```

The classifier is evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Cross-Validation
```

The project also examines:

```text
Class Probabilities
Class-Wise Accuracy
Decision Scores
```

Finally, LDA transforms the test data into a supervised discriminant space for visualization.

The key distinction is:

```text
PCA
→ Unsupervised
→ Preserve overall variance

LDA
→ Supervised
→ Separate known classes
→ Can classify observations
```

Therefore, LDA belongs in the **Supervised Machine Learning → Classification** section of the roadmap.



# ⭐ Final Takeaway

```text
Labeled Dataset
      ↓
Feature / Target Separation
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
LDA Classifier
      ↓
Training
      ↓
Prediction
      ↓
Accuracy + Classification Report
      ↓
Confusion Matrix
      ↓
Cross-Validation
      ↓
Probability Analysis
      ↓
Decision Analysis
      ↓
LDA Discriminant Space
```

For the Wine dataset:

```text
13 Features + 3 Classes
          ↓
         LDA
          ↓
   Class Prediction
          ↓
Class 0 / Class 1 / Class 2
```

LDA provides a practical supervised classification model while also offering a useful discriminant representation of the data.
---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project is part of a practical Machine Learning learning journey focused on implementing and understanding Machine Learning algorithms using Python and Scikit-learn.

---
