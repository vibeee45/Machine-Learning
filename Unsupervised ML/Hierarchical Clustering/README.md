# 🌳 Hierarchical Clustering

A comprehensive Machine Learning project demonstrating **Hierarchical Clustering** using the **Mall Customers** dataset. The project uses **Agglomerative Hierarchical Clustering** with Euclidean distance and Ward linkage to segment customers according to Annual Income and Spending Score.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Machine Learning](#-what-is-machine-learning)
- [Unsupervised Learning](#-what-is-unsupervised-learning)
- [Clustering](#-what-is-clustering)
- [Hierarchical Clustering](#-what-is-hierarchical-clustering)
- [Agglomerative Clustering](#-agglomerative-clustering)
- [Dendrogram](#-what-is-a-dendrogram)
- [Linkage](#-linkage)
- [Ward Linkage](#-ward-linkage)
- [Feature Scaling](#-feature-scaling)
- [Dataset](#-dataset-information)
- [Workflow](#-project-workflow)
- [Notebook Cells](#-step-by-step-notebook-explanation)
- [Advantages](#-advantages)
- [Limitations](#-limitations)
- [Applications](#-real-world-applications)
- [Learning Outcomes](#-learning-outcomes)
- [Conclusion](#-conclusion)

---

# 📌 Overview

This project demonstrates how **Hierarchical Clustering** can discover natural groups in customer data without using a target variable.

The two clustering features are:

```text
Annual Income (k$)
Spending Score (1-100)
```

The workflow is:

```text
Load Data → Explore → Select Features → Scale → Linkage → Dendrogram → Choose K → Agglomerative Clustering → Cluster Labels
```

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that allows computers to learn patterns from data and use those patterns to make predictions or discover useful structures.

Common Machine Learning applications include:

- Healthcare
- Finance
- Marketing
- Fraud detection
- Recommendation systems
- Customer analytics
- Forecasting
- Business intelligence

Machine Learning can be broadly organized as:

```text
Machine Learning
├── Supervised Learning
│   ├── Regression
│   └── Classification
├── Unsupervised Learning
│   ├── Clustering
│   └── Dimensionality Reduction
└── Reinforcement Learning
```

Hierarchical Clustering belongs to **Unsupervised Learning**.

---

# 🧠 What is Unsupervised Learning?

Unsupervised Learning works with data where a predefined target variable is not supplied.

The algorithm attempts to discover hidden structures, patterns, similarities, or groups.

In supervised learning we commonly have:

```text
X → Features
y → Target
```

In this project there is no `y` target. The algorithm works only with the selected customer features.

---

# 🔍 What is Clustering?

Clustering is an Unsupervised Learning technique that divides observations into groups based on similarity.

Each group is called a **cluster**.

```text
Customer Data
      ↓
Measure Similarity
      ↓
Group Similar Customers
      ↓
Create Clusters
      ↓
Interpret Segments
```

Customers in the same cluster should be relatively similar according to the features being analyzed.

---

# 🌳 What is Hierarchical Clustering?

Hierarchical Clustering creates a hierarchy of clusters rather than only producing one final grouping.

The hierarchy can be visualized using a **dendrogram**.

The dendrogram shows how observations or groups are merged at different distance levels.

This makes hierarchical clustering useful when we want to understand not only the final groups, but also the relationships between those groups.

---

# 🔗 Agglomerative Clustering

This project uses **Agglomerative Hierarchical Clustering**.

Agglomerative clustering follows a bottom-up approach.

Initially every observation is treated as its own cluster:

```text
{A} {B} {C} {D} {E}
```

The algorithm repeatedly merges appropriate clusters:

```text
{A,B} {C} {D,E}
        ↓
{A,B,C} {D,E}
        ↓
{A,B,C,D,E}
```

The complete sequence of merges forms the hierarchy.

---

# ⬆️ Bottom-Up Approach

Agglomerative clustering starts with many small clusters and gradually combines them.

```text
Individual Observations
          ↓
Small Clusters
          ↓
Larger Clusters
          ↓
One Hierarchy
```

This is the opposite of divisive clustering, which starts with one large cluster and repeatedly splits it.

---

# 📏 Distance Between Clusters

Hierarchical clustering needs a way to measure how close observations or clusters are.

This project uses **Euclidean distance**.

For two points:

```text
A = (x₁, y₁)
B = (x₂, y₂)
```

Euclidean distance is:

```text
Distance = √((x₂ − x₁)² + (y₂ − y₁)²)
```

The distance metric influences the resulting hierarchy.

---

# 🔗 Linkage

Linkage determines how the distance between clusters is defined.

Common linkage strategies include:

- Single linkage
- Complete linkage
- Average linkage
- Ward linkage

This project uses **Ward linkage**.

---

# 📐 Ward Linkage

The project calculates the hierarchy with:

```python
linkage(
    X_scaled,
    method="ward"
)
```

Ward linkage attempts to merge clusters while minimizing the increase in within-cluster variance.

The final AgglomerativeClustering model also uses:

```python
linkage="ward"
```

---

# 🌲 What is a Dendrogram?

A dendrogram is a tree-like visualization of hierarchical clustering.

It shows:

- Which groups are merged.
- The order of merges.
- The distance at which merges occur.
- The hierarchical structure.

Conceptually:

```text
Distance
   │          ┌──────────
   │      ┌───┤
   │  ┌───┤   └──────────
   │  │   │
   │  │   └──────────────
   └──┴─────────────────── Customers
```

The vertical axis represents merge distance.

---

# 📖 Reading a Dendrogram

A horizontal cut can be imagined across the dendrogram.

The number of major branches intersected by that cut provides a practical indication of the number of clusters.

The project examines the dendrogram and uses:

```text
K = 5
```

for the final model.

---

# ⚖️ Feature Scaling

Hierarchical clustering is distance-based, so scaling is important.

The project uses:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardization is conceptually:

```text
z = (x − mean) / standard deviation
```

This puts the selected features on a comparable scale before distances are calculated.

---

# 📂 Dataset Information

**Dataset:** `Mall_Customers.csv`

The local notebook path is:

```text
C:\Users\91958\Desktop\Machine Learning\datasets\Mall_Customers.csv
```

The standard dataset contains customer demographic and spending information.

---

# 📊 Dataset Features

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Genre` | Customer gender in the standard dataset |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Customer spending behavior score |

---

# 🎯 Clustering Features

The project selects:

```text
Annual Income (k$)
Spending Score (1-100)
```

using:

```python
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]
```

There is no target variable because this is an unsupervised learning problem.

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

---

# 📁 Project Structure

```text
Hierarchical-Clustering/
│
├── Hierarchical Clustering.ipynb
├── Mall_Customers.csv
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 🔄 Project Workflow

```text
Import Libraries
       ↓
Load Dataset
       ↓
Dataset Exploration
       ↓
Missing Value Check
       ↓
Duplicate Check
       ↓
Select Features
       ↓
Standardize Features
       ↓
Calculate Linkage Matrix
       ↓
Display Linkage Matrix
       ↓
Create Dendrogram
       ↓
Create Simplified Dendrogram
       ↓
Select K = 5
       ↓
Create Agglomerative Model
       ↓
Generate Cluster Labels
       ↓
Add Labels to Dataset
```

---

# 📖 Step-by-Step Notebook Explanation

The notebook contains 20 cells in the current implementation. Each cell is explained below.

---

## 🔹 Cell 1 – Import Libraries

```python
# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# SciPy for Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

### Explanation

This cell imports all libraries required for the project.

`NumPy` supports numerical operations. `Pandas` loads and manipulates the dataset. `Matplotlib` and `Seaborn` support visualization. `StandardScaler` performs feature scaling. `AgglomerativeClustering` provides the final clustering algorithm. SciPy's `linkage` and `dendrogram` functions create and visualize the hierarchy.

The warning filter keeps the notebook output cleaner.

---

## 🔹 Cell 2 – Load Dataset

```python
# Loading the Mall Customers dataset

df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\Mall_Customers.csv"
)

# Displaying the first five rows

df.head()
```

### Explanation

The CSV file is loaded into the DataFrame `df` using `pd.read_csv()`.

`df.head()` displays the first five rows and provides an initial look at the customer data.

If the dataset is stored beside the notebook, the path can be changed to:

```python
df = pd.read_csv("Mall_Customers.csv")
```

---

## 🔹 Cell 3 – Dataset Information

```python
# Displaying dataset information

df.info()
```

### Explanation

`df.info()` displays the DataFrame structure, including the number of records, column names, data types, and non-null counts.

This is an important exploratory step before preprocessing.

---

## 🔹 Cell 4 – Statistical Summary

```python
# Statistical summary of numerical columns

df.describe()
```

### Explanation

`df.describe()` generates descriptive statistics such as count, mean, standard deviation, minimum, quartiles, and maximum for numerical columns.

It provides an initial understanding of the numerical distributions.

---

## 🔹 Cell 5 – Check Missing Values

```python
# Checking for missing values

df.isnull().sum()
```

### Explanation

This cell counts missing values in every column.

`df.isnull()` identifies missing values and `sum()` counts them.

Missing values should be handled before applying a clustering algorithm if they are present.

---

## 🔹 Cell 6 – Dataset Dimensions

```python
# Displaying dataset dimensions

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
```

### Explanation

`df.shape[0]` returns the number of rows, while `df.shape[1]` returns the number of columns.

This gives a quick overview of dataset size.

---

## 🔹 Cell 7 – Check Duplicate Records

```python
# Checking for duplicate records

print("Duplicate Rows :", df.duplicated().sum())
```

### Explanation

`df.duplicated()` identifies duplicate rows and `sum()` counts them.

Duplicate observations can influence clustering because repeated observations can receive additional influence.

---

## 🔹 Cell 8 – Display Column Names

```python
# Displaying column names

print("Columns in the dataset:")
print(df.columns.tolist())
```

### Explanation

This cell prints all column names.

Checking the exact names is important before selecting the clustering variables.

---

## 🔹 Cell 9 – Select Features for Clustering

```python
# Selecting features for hierarchical clustering

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("Selected Features:")
display(X.head())

print("\nFeatures Shape:", X.shape)
```

### Explanation

This cell selects the two features used for clustering:

```text
Annual Income (k$)
Spending Score (1-100)
```

The selected data is stored in `X`.

There is no `y` variable because hierarchical clustering is unsupervised.

---

## 🔹 Cell 10 – Display Selected Features

```python
# Displaying the selected clustering features

display(X)
```

### Explanation

This cell displays the selected clustering features so they can be inspected before scaling.

---

## 🔹 Cell 11 – Standardize Features

```python
# Standardizing the selected features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaled Features:")
display(pd.DataFrame(
    X_scaled,
    columns=X.columns
).head())
```

### Explanation

A `StandardScaler` is created and fitted to the selected features.

`fit_transform()` calculates the required statistics and transforms the data.

The standardized data is stored in `X_scaled`.

Scaling is important because the clustering process uses distances.

---

## 🔹 Cell 12 – Check Scaled Features

```python
# Checking the mean and standard deviation after scaling

scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("Mean of scaled features:")
print(scaled_df.mean())

print("\nStandard deviation of scaled features:")
print(scaled_df.std())
```

### Explanation

This cell verifies the result of standardization.

The scaled array is converted back to a DataFrame and its mean and standard deviation are displayed.

The features should be centered around zero with approximately unit variance.

---

## 🔹 Cell 13 – Calculate Linkage Matrix

```python
# Calculating the linkage matrix

linkage_matrix = linkage(
    X_scaled,
    method="ward"
)

print("Linkage matrix shape:", linkage_matrix.shape)
```

### Explanation

This is the main hierarchical structure calculation.

The SciPy `linkage()` function builds the sequence of cluster merges.

The project uses:

```text
method = "ward"
```

The resulting `linkage_matrix` contains information needed for the dendrogram.

---

## 🔹 Cell 14 – Display Linkage Matrix

```python
# Displaying the first 10 rows of the linkage matrix

linkage_df = pd.DataFrame(
    linkage_matrix,
    columns=[
        "Cluster 1",
        "Cluster 2",
        "Distance",
        "Sample Count"
    ]
)

display(linkage_df.head(10))
```

### Explanation

This cell converts the linkage matrix into a DataFrame for easier inspection.

The four displayed fields represent the two groups being merged, the merge distance, and the number of observations contained in the newly formed group.

---

## 🔹 Cell 15 – Create Dendrogram

```python
# Creating the hierarchical clustering dendrogram

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")

plt.grid()
plt.show()
```

### Explanation

This cell visualizes the hierarchy created by the linkage matrix.

The X-axis represents customers and the Y-axis represents merge distance.

The dendrogram helps us inspect how customer groups are progressively merged.

---

## 🔹 Cell 16 – Create Simplified Dendrogram

```python
# Creating a simplified dendrogram

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=20
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Groups")
plt.ylabel("Distance")

plt.grid()
plt.show()
```

### Explanation

The full dendrogram can be visually dense when many observations are present.

`truncate_mode="lastp"` creates a simplified view of the final part of the hierarchy, while `p=20` controls the amount shown.

This makes the hierarchy easier to inspect.

---

## 🔹 Cell 17 – Select Number of Clusters

```python
# Selecting the number of clusters

optimal_k = 5

print("Optimal Number of Clusters:", optimal_k)
```

### Explanation

After examining the dendrogram, the project selects five clusters.

The value is stored in:

```python
optimal_k = 5
```

This value is passed to the final Agglomerative Clustering model.

---

## 🔹 Cell 18 – Create Agglomerative Clustering Model

```python
# Creating the Agglomerative Clustering model

model = AgglomerativeClustering(
    n_clusters=optimal_k,
    metric="euclidean",
    linkage="ward"
)

print("Agglomerative Clustering model created successfully!")
```

### Explanation

This cell creates the final model.

The configuration is:

```text
n_clusters = 5
metric = euclidean
linkage = ward
```

`n_clusters` specifies the desired final number of groups.

`metric="euclidean"` specifies the distance measure.

`linkage="ward"` specifies Ward's merging strategy.

---

## 🔹 Cell 19 – Generate Cluster Labels

```python
# Generating cluster labels

cluster_labels = model.fit_predict(X_scaled)

print("Cluster Labels:")
print(cluster_labels)
```

### Explanation

`fit_predict()` both fits the Agglomerative Clustering model and returns the cluster assignment for each observation.

The labels are stored in `cluster_labels`.

With five clusters, the labels are integer identifiers representing the five groups.

---

## 🔹 Cell 20 – Add Cluster Labels to Dataset

```python
# Adding cluster labels to the original dataset

df["Cluster"] = cluster_labels

display(df.head(10))
```

### Explanation

This cell adds a new `Cluster` column to the original DataFrame.

The resulting dataset can now be used to inspect customers together with their hierarchical cluster assignments.

---

# 📊 Understanding Cluster Labels

Cluster labels such as:

```text
0
1
2
3
4
```

are identifiers only.

A higher cluster number does not mean a better or worse customer.

The business meaning of a cluster should be determined by examining the characteristics of the observations assigned to it.

---

# 👥 Customer Segmentation

The selected variables allow us to interpret customer groups using:

```text
Annual Income
+
Spending Score
```

Possible interpretations include:

```text
Low Income / Low Spending
Low Income / High Spending
Average Income / Average Spending
High Income / Low Spending
High Income / High Spending
```

These are example interpretations. The exact meaning should be based on the actual cluster distributions produced by the notebook.

---

# 🆚 K-Means vs Hierarchical Clustering

| Feature | K-Means | Hierarchical Clustering |
|---|---|---|
| Learning Type | Unsupervised | Unsupervised |
| Main Idea | Centroid-based | Hierarchy-based |
| Dendrogram | No | Yes |
| Uses Centroids | Yes | No final centroid requirement |
| Requires K for final model | Yes | Yes for final cut/model |
| Main Visualization | Elbow/cluster plot | Dendrogram |
| Scaling Important | Yes | Yes |
| Output | Cluster labels | Hierarchy and labels |

K-Means and Hierarchical Clustering can both be used for customer segmentation, but their clustering processes are fundamentally different.

---

# 🌍 Real-World Applications

Hierarchical Clustering can be applied to:

- Customer segmentation
- Market research
- Retail analytics
- Healthcare grouping
- Biological data analysis
- Document clustering
- Image analysis
- Behavioral analysis
- Product segmentation
- Business intelligence

---

# ✅ Advantages

- Produces a hierarchy of clusters.
- Provides a dendrogram for visualization.
- Helps explore relationships at multiple levels.
- Does not require the final number of clusters to construct the hierarchy.
- Useful for exploratory analysis.
- Can reveal nested structures.
- Useful for relatively small and medium datasets.
- Cluster formation is visually interpretable through the dendrogram.

---

# ❌ Limitations

- Can be computationally expensive for large datasets.
- Sensitive to the distance metric.
- Sensitive to the linkage method.
- Early merges are not normally undone in standard agglomerative clustering.
- Outliers can affect distance calculations.
- Scaling can influence results.
- Large dendrograms can become difficult to interpret.
- Different linkage choices can produce different structures.

---

# 🔗 Why Linkage Matters

Different linkage methods define cluster similarity differently.

Common methods are:

```text
Single
Complete
Average
Ward
```

This project uses Ward linkage:

```python
method="ward"
```

and:

```python
linkage="ward"
```

Ward linkage aims to control the increase in within-cluster variance when clusters are merged.

---

# 📏 Why Distance Matters

Distance is central to hierarchical clustering.

This project uses:

```python
metric="euclidean"
```

Euclidean distance measures straight-line distance between points in the feature space.

The final model therefore uses:

```text
Euclidean Distance + Ward Linkage
```

---

# 📊 Linkage Matrix Summary

The linkage matrix records the hierarchy of merges.

The project displays:

```text
Cluster 1
Cluster 2
Distance
Sample Count
```

The matrix is then passed to `dendrogram()` to produce a visual representation of the hierarchy.

---

# 🎯 Final Model Configuration

The final model is:

```python
model = AgglomerativeClustering(
    n_clusters=optimal_k,
    metric="euclidean",
    linkage="ward"
)
```

with:

```text
Number of clusters = 5
Distance metric    = Euclidean
Linkage             = Ward
```

---

# 📌 Complete Notebook Structure

```text
Cell 1  → Import Libraries
Cell 2  → Load Dataset
Cell 3  → Dataset Information
Cell 4  → Statistical Summary
Cell 5  → Missing Values
Cell 6  → Dataset Dimensions
Cell 7  → Duplicate Records
Cell 8  → Column Names
Cell 9  → Select Features
Cell 10 → Display Features
Cell 11 → Standardize Features
Cell 12 → Check Scaling
Cell 13 → Linkage Matrix
Cell 14 → Display Linkage Matrix
Cell 15 → Full Dendrogram
Cell 16 → Simplified Dendrogram
Cell 17 → Select K = 5
Cell 18 → Agglomerative Model
Cell 19 → Cluster Labels
Cell 20 → Add Cluster Labels
```

---

# 🔄 End-to-End Summary

```text
Mall Customers Dataset
          ↓
Data Exploration
          ↓
Data Quality Checks
          ↓
Feature Selection
          ↓
StandardScaler
          ↓
Ward Linkage
          ↓
Linkage Matrix
          ↓
Dendrogram
          ↓
Select K = 5
          ↓
Agglomerative Clustering
          ↓
Cluster Labels
          ↓
Customer Segmentation
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand Unsupervised Learning.
- Understand clustering.
- Understand Hierarchical Clustering.
- Understand Agglomerative Clustering.
- Understand the bottom-up approach.
- Understand linkage methods.
- Understand Ward linkage.
- Understand Euclidean distance.
- Understand dendrograms.
- Read a dendrogram.
- Select a suitable clustering level.
- Apply feature scaling.
- Use `StandardScaler`.
- Calculate a linkage matrix.
- Visualize hierarchical relationships.
- Build `AgglomerativeClustering`.
- Generate cluster labels.
- Add labels to a DataFrame.
- Perform customer segmentation.
- Compare Hierarchical Clustering with K-Means.

---

# 🚀 Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy notebook
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the Hierarchical Clustering notebook and execute the cells sequentially.

---

# 📦 Requirements

```text
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
SciPy
Jupyter Notebook
```

---

# 📍 Dataset Path

The notebook currently uses:

```python
df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\Mall_Customers.csv"
)
```

If the dataset is stored next to the notebook:

```python
df = pd.read_csv("Mall_Customers.csv")
```

---

# 🧪 Reproducibility

The final model does not use a random-state parameter in this implementation.

The final clustering configuration is:

```text
n_clusters = 5
metric = euclidean
linkage = ward
```

The hierarchy is calculated with:

```python
linkage(X_scaled, method="ward")
```

---

# 🏁 Conclusion

Hierarchical Clustering is an important Unsupervised Learning algorithm because it provides both final cluster assignments and a view of how the clusters are formed.

In this project, the Mall Customers dataset is explored and the following features are selected:

```text
Annual Income (k$)
Spending Score (1-100)
```

The features are standardized using `StandardScaler`.

A linkage matrix is then generated using Ward linkage, and the resulting hierarchy is visualized with a dendrogram.

After examining the dendrogram, the project selects:

```text
K = 5
```

The final Agglomerative Clustering model uses:

```text
Euclidean distance
Ward linkage
5 clusters
```

Cluster labels are generated and added to the original dataset.

This project provides a practical understanding of hierarchical customer segmentation and demonstrates how a dendrogram can be used to study the relationships between customer groups.



# ⭐ Final Takeaway

```text
Unsupervised Learning
        ↓
     Clustering
        ↓
Hierarchical Clustering
        ↓
Agglomerative Approach
        ↓
Feature Scaling
        ↓
Ward Linkage
        ↓
Linkage Matrix
        ↓
Dendrogram
        ↓
Select Cluster Level
        ↓
Agglomerative Model
        ↓
Cluster Labels
        ↓
Customer Segmentation
```

Hierarchical Clustering is especially useful when we want to understand not only the final customer groups, but also the hierarchy and relationships through which those groups are formed.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project is part of a practical Machine Learning learning journey focused on implementing and understanding algorithms using Python.

---
