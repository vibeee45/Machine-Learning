# 🟣 Principal Component Analysis (PCA)

A comprehensive Machine Learning project demonstrating **Principal Component Analysis (PCA)** using the **Wine Recognition Dataset** from Scikit-learn.

This project focuses on **Dimensionality Reduction**, an important area of Unsupervised Learning.

The original Wine dataset contains **13 numerical features**. PCA transforms these features into a smaller set of principal components and the project uses **2 components** for visualization.

---

# 📑 Table of Contents

- Overview
- Machine Learning
- Unsupervised Learning
- Dimensionality Reduction
- PCA
- Why PCA?
- Standardization
- Principal Components
- Explained Variance
- Explained Variance Ratio
- Cumulative Explained Variance
- Dataset
- Technologies
- Workflow
- 30 Cell Notebook Explanation
- PCA Transformation
- Visualization
- Component Loadings
- PCA vs Clustering
- Advantages
- Limitations
- Applications
- Learning Outcomes
- Conclusion

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that allows computers to learn patterns from data and make decisions or discover useful structures without explicitly programming every rule.

Common applications include:

- Healthcare
- Finance
- Marketing
- Recommendation systems
- Fraud detection
- Customer analytics
- Pattern recognition

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

PCA is commonly used for **dimensionality reduction**.

---

# 🧠 What is Unsupervised Learning?

Unsupervised Learning works with data where a predefined target is not required to discover the underlying structure.

Examples:

- Clustering
- Dimensionality Reduction
- Association analysis

In supervised learning:

```text
X → Input Features
y → Target
```

For PCA:

```text
X → Original Features
```

PCA analyzes the structure and variation of the input features.

The original wine classes are kept separately in this project only so that the reduced representation can be visualized by class.

---

# 📉 What is Dimensionality Reduction?

Dimensionality Reduction means reducing the number of features while attempting to preserve important information.

For example:

```text
13 Features
     ↓
Dimensionality Reduction
     ↓
2 Features
```

Benefits include:

- Reduced complexity
- Easier visualization
- Less redundancy
- Lower computational cost
- Easier exploratory analysis

PCA is one of the most widely used dimensionality-reduction techniques.

---

# 🟣 What is PCA?

**PCA** stands for **Principal Component Analysis**.

PCA transforms the original features into new variables called **principal components**.

The components are ordered according to the amount of variance they explain.

Conceptually:

```text
Original Features
       ↓
   Standardize
       ↓
      PCA
       ↓
Principal Components
       ↓
Reduced Representation
```

---

# 🎯 Why Use PCA?

Suppose a dataset contains:

```text
13 Features
```

It is difficult to visualize all 13 dimensions at once.

PCA can transform them into:

```text
PC1
PC2
```

and allow the data to be displayed on a 2D graph.

---

# ⚖️ Why Standardization is Important

PCA is affected by feature scale.

If one feature has much larger numerical values than another, it can have a disproportionately large influence on the resulting components.

Therefore, the project uses:

```python
StandardScaler()
```

before applying PCA.

The standardization formula is:

```text
z = (x − mean) / standard deviation
```

---

# 🧭 Principal Components

PCA creates new variables called principal components.

The first component is:

```text
PC1
```

The second is:

```text
PC2
```

and so on.

PC1 captures the maximum possible variance along its direction.

PC2 captures the maximum remaining variance subject to being orthogonal to PC1.

---

# 📈 Explained Variance

Explained variance tells us how much variability is captured by each principal component.

It is obtained with:

```python
pca.explained_variance_
```

---

# 📊 Explained Variance Ratio

The explained variance ratio represents the proportion of total variance captured by each component.

It is obtained using:

```python
pca.explained_variance_ratio_
```

For example, conceptually:

```text
PC1 → variance contribution
PC2 → variance contribution
PC3 → variance contribution
...
```

The exact values are calculated by the notebook.

---

# ➕ Cumulative Explained Variance

Cumulative explained variance shows how much total variance is retained when multiple components are combined.

The project calculates it using:

```python
cumulative_variance = np.cumsum(
    explained_variance_ratio
)
```

The cumulative value increases as more components are included.

---

# 📉 Cumulative Variance Plot

The project plots cumulative explained variance against the number of principal components.

A reference line at:

```text
95%
```

is displayed to help inspect a common variance-retention threshold.

---

# 🍷 Dataset Information

## Dataset

**Wine Recognition Dataset**

The dataset is loaded directly from Scikit-learn:

```python
from sklearn.datasets import load_wine

wine = load_wine()
```

Therefore, no external CSV file is required.

The standard dataset contains:

```text
178 observations
13 numerical features
3 wine classes
```

---

# 📊 Wine Dataset Features

The original dataset contains:

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

# 🎯 Features and Target

The project separates:

```python
X = df.copy()
y = wine.target
```

Therefore:

```text
X → 13 original numerical features
y → Original wine class
```

PCA is applied to `X`.

The target is retained separately for visualization.

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
PCA/
│
├── PCA.ipynb
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

The dataset is loaded directly through Scikit-learn.

---

# 🔄 Complete PCA Workflow

```text
Load Wine Dataset
       ↓
Create DataFrame
       ↓
Explore Dataset
       ↓
Check Missing Values
       ↓
Separate Features and Target
       ↓
Standardize Features
       ↓
Create PCA
       ↓
Fit PCA
       ↓
Explained Variance
       ↓
Explained Variance Ratio
       ↓
Cumulative Variance
       ↓
Select 2 Components
       ↓
Transform Dataset
       ↓
Create PCA DataFrame
       ↓
Add Target for Visualization
       ↓
Visualize PC1 vs PC2
       ↓
Analyze Component Loadings
       ↓
Create Loading Heatmap
```

---

# 📖 Step-by-Step Notebook Explanation

The following sections explain all **30 cells** in the notebook.

---

## 🔹 Cell 1 — Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
```

### Explanation

This imports NumPy, Pandas, visualization libraries, the Wine dataset loader, `StandardScaler`, and PCA.

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

df.head()
```

### Explanation

The numerical feature matrix is converted into a Pandas DataFrame using the official feature names.

---

## 🔹 Cell 4 — Dataset Information

```python
df.info()
```

### Explanation

This displays the number of rows, columns, data types, and non-null counts.

---

## 🔹 Cell 5 — Statistical Summary

```python
df.describe()
```

### Explanation

This provides descriptive statistics for the numerical features.

---

## 🔹 Cell 6 — Check Missing Values

```python
df.isnull().sum()
```

### Explanation

This checks each feature for missing observations.

---

## 🔹 Cell 7 — Dataset Dimensions

```python
print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
```

### Explanation

This displays the number of observations and features.

For the standard Wine dataset:

```text
Rows = 178
Columns = 13
```

---

## 🔹 Cell 8 — Display Feature Names

```python
print("Features in the Wine dataset:")

for i, feature in enumerate(wine.feature_names, start=1):
    print(i, ":", feature)
```

### Explanation

This prints all 13 feature names with their corresponding numbers.

---

## 🔹 Cell 9 — Separate Features and Target

```python
X = df.copy()
y = wine.target

print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)
```

### Explanation

`X` contains the input features and `y` contains the original wine classes.

PCA uses `X`; it does not use `y`.

---

## 🔹 Cell 10 — Display Features and Target

```python
print("Features:")
display(X.head())

print("
Target:")
print(y[:10])
```

### Explanation

This displays sample feature values and the first target labels.

---

## 🔹 Cell 11 — Standardize Features

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaled Features:")
display(pd.DataFrame(
    X_scaled,
    columns=X.columns
).head())
```

### Explanation

All 13 features are standardized before PCA so that scale differences do not dominate the analysis.

---

## 🔹 Cell 12 — Check Scaled Feature Statistics

```python
scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("Mean of scaled features:")
print(scaled_df.mean())

print("
Standard deviation of scaled features:")
print(scaled_df.std())
```

### Explanation

This verifies the mean and standard deviation of the standardized features.

---

## 🔹 Cell 13 — Create PCA Model

```python
pca = PCA()

print("PCA model created successfully!")
```

### Explanation

A PCA model is created without limiting the number of components. This allows us to inspect the variance of all available components.

---

## 🔹 Cell 14 — Fit PCA Model

```python
pca.fit(X_scaled)

print("PCA model fitted successfully!")
```

### Explanation

PCA learns the principal directions from the standardized dataset.

---

## 🔹 Cell 15 — Explained Variance

```python
explained_variance = pca.explained_variance_

print("Explained Variance:")
print(explained_variance)
```

### Explanation

This returns the variance associated with each principal component.

---

## 🔹 Cell 16 — Explained Variance Ratio

```python
explained_variance_ratio = pca.explained_variance_ratio_

print("Explained Variance Ratio:")
print(explained_variance_ratio)
```

### Explanation

This shows the proportion of total variance represented by each component.

---

## 🔹 Cell 17 — Create Variance DataFrame

```python
variance_df = pd.DataFrame({
    "Principal Component": [
        f"PC{i+1}" for i in range(len(explained_variance_ratio))
    ],
    "Explained Variance Ratio": explained_variance_ratio
})

display(variance_df)
```

### Explanation

This organizes the explained variance ratio into a readable DataFrame.

---

## 🔹 Cell 18 — Calculate Cumulative Explained Variance

```python
cumulative_variance = np.cumsum(
    explained_variance_ratio
)

print("Cumulative Explained Variance:")
print(cumulative_variance)
```

### Explanation

This calculates the total variance retained as additional components are included.

---

## 🔹 Cell 19 — Plot Explained Variance

```python
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% Variance"
)

plt.grid()
plt.legend()
plt.show()
```

### Explanation

This graph shows how quickly variance accumulates as components are added.

The 95% line helps inspect how many components would be needed for a high-variance representation.

---

## 🔹 Cell 20 — Select Number of Components

```python
n_components = 2

print("Selected Number of Components:", n_components)
```

### Explanation

For the final visualization, the project selects two principal components:

```text
PC1
PC2
```

---

## 🔹 Cell 21 — Create PCA with 2 Components

```python
pca_2 = PCA(
    n_components=n_components
)

print("PCA with 2 components created successfully!")
```

### Explanation

This creates the final PCA model that will produce exactly two components.

---

## 🔹 Cell 22 — Transform the Dataset

```python
X_pca = pca_2.fit_transform(X_scaled)

print("Original Shape :", X_scaled.shape)
print("PCA Shape      :", X_pca.shape)
```

### Explanation

This is the main dimensionality-reduction step.

The standard Wine dataset changes from:

```text
178 × 13
```

to:

```text
178 × 2
```

---

## 🔹 Cell 23 — Display PCA Data

```python
print("First 10 PCA observations:")

display(pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
).head(10))
```

### Explanation

This displays the first ten observations in the new two-dimensional PCA space.

---

## 🔹 Cell 24 — Create PCA DataFrame

```python
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

display(pca_df.head())
```

### Explanation

This creates a DataFrame containing the two principal components.

---

## 🔹 Cell 25 — Add Target Class

```python
pca_df["Target"] = y

display(pca_df.head(10))
```

### Explanation

The original wine class is added for visualization.

The target does not participate in PCA calculation.

---

## 🔹 Cell 26 — Check PCA Component Variance

```python
pca_variance = pca_2.explained_variance_ratio_

print("PC1 Variance :", pca_variance[0])
print("PC2 Variance :", pca_variance[1])

print("
Total Variance Retained :",
      pca_variance.sum())
```

### Explanation

This reports the explained variance of PC1, PC2, and their combined variance.

---

## 🔹 Cell 27 — Create PCA Scatter Plot

```python
plt.figure(figsize=(10, 7))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"]
)

plt.title("PCA - Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid()
plt.show()
```

### Explanation

This displays the wine observations using PC1 and PC2.

---

## 🔹 Cell 28 — PCA Visualization by Target Class

```python
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Target",
    palette="deep"
)

plt.title("PCA Visualization by Wine Class")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid()
plt.show()
```

### Explanation

This colors the reduced observations according to the original wine classes so that class separation can be visually inspected.

---

## 🔹 Cell 29 — Display PCA Components

```python
components_df = pd.DataFrame(
    pca_2.components_,
    columns=X.columns,
    index=["PC1", "PC2"]
)

display(components_df)
```

### Explanation

This displays the component loadings.

Rows represent PC1 and PC2, while columns represent the original 13 features.

---

## 🔹 Cell 30 — Visualize Feature Contributions

```python
plt.figure(figsize=(14, 6))

sns.heatmap(
    components_df,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("PCA Component Loadings")
plt.xlabel("Original Features")
plt.ylabel("Principal Components")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()
```

### Explanation

This heatmap makes it easier to inspect the contribution of every original feature to PC1 and PC2.

---

# 📊 PCA Before and After

Before PCA:

```text
178 Observations
13 Features
```

After PCA:

```text
178 Observations
2 Principal Components
```

Therefore:

```text
13 Dimensions
      ↓
     PCA
      ↓
2 Dimensions
```

---

# 🔬 Understanding Component Loadings

Principal components are combinations of the original features.

Conceptually:

```text
PC1 =
w1(Alcohol)
+ w2(Malic acid)
+ ...
+ w13(Proline)
```

and:

```text
PC2 =
v1(Alcohol)
+ v2(Malic acid)
+ ...
+ v13(Proline)
```

The coefficients can be inspected through:

```python
pca_2.components_
```

Larger absolute loading values indicate stronger contributions to that component.

---

# 🆚 PCA vs Clustering

PCA and clustering solve different problems.

### PCA

```text
Goal
 ↓
Reduce Dimensions
```

### K-Means

```text
Goal
 ↓
Create Clusters
```

### Hierarchical Clustering

```text
Goal
 ↓
Create a Cluster Hierarchy
```

### DBSCAN

```text
Goal
 ↓
Find Dense Regions and Noise
```

PCA can also be used before clustering as a dimensionality-reduction step.

---

# 🔗 PCA Before Clustering

A common workflow is:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Scaling
     ↓
PCA
     ↓
Reduced Features
     ↓
Clustering
```

This can be useful for high-dimensional datasets.

---

# 🌍 Real-World Applications

PCA can be used for:

- Data visualization
- Feature reduction
- Image processing
- Genomics
- Finance
- Customer analytics
- Pattern recognition
- Machine Learning preprocessing
- High-dimensional exploratory analysis

---

# ✅ Advantages of PCA

- Reduces dimensionality.
- Makes high-dimensional data easier to visualize.
- Can reduce redundant information.
- Can improve computational efficiency.
- Produces ordered components.
- Helps identify major directions of variation.
- Can be used as preprocessing for other Machine Learning algorithms.

---

# ❌ Limitations of PCA

- Components can be difficult to interpret.
- PCA is sensitive to feature scaling.
- Transformed features may not have direct real-world meanings.
- Information can be lost when too few components are selected.
- PCA maximizes variance, which is not always the same as maximizing predictive usefulness.
- PCA is a linear dimensionality-reduction technique.

---

# 📌 PCA Core Concepts

```text
PCA
 │
 ├── Standardization
 │
 ├── Principal Components
 │
 ├── Explained Variance
 │
 ├── Explained Variance Ratio
 │
 ├── Cumulative Variance
 │
 ├── Dimensionality Reduction
 │
 └── Component Loadings
```

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
Cell 8  → Display Feature Names
Cell 9  → Separate Features and Target
Cell 10 → Display Features and Target

Cell 11 → Standardize Features
Cell 12 → Check Scaled Feature Statistics
Cell 13 → Create PCA Model
Cell 14 → Fit PCA Model
Cell 15 → Explained Variance
Cell 16 → Explained Variance Ratio
Cell 17 → Create Variance DataFrame
Cell 18 → Calculate Cumulative Variance
Cell 19 → Plot Cumulative Explained Variance
Cell 20 → Select Number of Components

Cell 21 → Create PCA with 2 Components
Cell 22 → Transform Dataset
Cell 23 → Display PCA Data
Cell 24 → Create PCA DataFrame
Cell 25 → Add Target Class
Cell 26 → Check PCA Component Variance
Cell 27 → Create PCA Scatter Plot
Cell 28 → PCA Visualization by Target Class
Cell 29 → Display PCA Components
Cell 30 → Visualize Feature Contributions
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand dimensionality reduction.
- Understand PCA.
- Understand principal components.
- Understand PC1 and PC2.
- Understand feature standardization.
- Understand explained variance.
- Understand explained variance ratio.
- Understand cumulative explained variance.
- Select an appropriate number of components.
- Transform high-dimensional data.
- Visualize high-dimensional data in 2D.
- Understand PCA component loadings.
- Interpret component-loading heatmaps.
- Use Scikit-learn's `PCA`.
- Use `StandardScaler`.
- Apply PCA to a real dataset.
- Understand how PCA differs from clustering.
- Use PCA before other Machine Learning algorithms.

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

Open the PCA notebook and run the cells sequentially.

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

# 🧪 Dataset Loading

The dataset is loaded directly from Scikit-learn:

```python
from sklearn.datasets import load_wine

wine = load_wine()
```

No separate CSV download is required.

---

# 🏁 Conclusion

PCA is an important dimensionality-reduction technique for simplifying high-dimensional datasets.

In this project, the Wine Recognition Dataset contains:

```text
13 Original Features
```

The features are standardized using:

```python
StandardScaler()
```

PCA is initially fitted using all available components so that explained variance can be analyzed.

The final visualization uses:

```text
2 Principal Components
```

The transformation is:

```text
13 Dimensions
      ↓
     PCA
      ↓
2 Dimensions
      ↓
PC1 + PC2
```

The original classes are retained separately and used only to visualize the reduced data by class.

Finally, component loadings are visualized with a heatmap to understand how the original features contribute to PC1 and PC2.

This project demonstrates the complete PCA workflow, from dataset exploration and scaling to dimensionality reduction, visualization, and component interpretation.



# ⭐ Final Takeaway

```text
High-Dimensional Dataset
          ↓
Feature Exploration
          ↓
Feature Standardization
          ↓
Principal Component Analysis
          ↓
Calculate Explained Variance
          ↓
Select Components
          ↓
Transform Data
          ↓
Reduced Representation
          ↓
Visualization
          ↓
Interpret Component Loadings
```

For this project:

```text
13 Original Features
        ↓
     StandardScaler
        ↓
        PCA
        ↓
2 Principal Components
        ↓
PC1 + PC2
        ↓
2D Visualization
```
---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project is part of a practical Machine Learning learning journey focused on implementing and understanding algorithms using Python and Scikit-learn.

---
PCA is therefore a powerful technique for reducing the complexity of high-dimensional data while preserving important variation in the dataset.
