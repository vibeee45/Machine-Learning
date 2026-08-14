# 🔵 DBSCAN Clustering

A comprehensive Machine Learning project demonstrating **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** using the **Mall Customers** dataset.

This project is part of the Unsupervised Learning section of the Machine Learning roadmap.

The project uses:

```text
Annual Income (k$)
Spending Score (1-100)
```

to discover customer groups based on **density**, rather than centroids or hierarchical merging.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- Types of Machine Learning
- What is Unsupervised Learning?
- What is Clustering?
- What is DBSCAN?
- Why DBSCAN?
- Core Concepts
- Epsilon
- Min Samples
- Core Points
- Border Points
- Noise Points
- How DBSCAN Works
- Feature Scaling
- Dataset Information
- Dataset Features
- Clustering Features
- Technologies Used
- Project Structure
- Project Workflow
- Step-by-Step Notebook Explanation
- DBSCAN vs K-Means
- DBSCAN vs Hierarchical Clustering
- Advantages
- Limitations
- Real-World Applications
- Learning Outcomes
- Conclusion

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that enables computers to learn patterns from data and make decisions or discover useful structures without explicitly programming every rule.

Machine Learning is commonly used in healthcare, finance, banking, marketing, recommendation systems, fraud detection, customer analytics, and business intelligence.

Machine Learning can broadly be divided into:

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

DBSCAN belongs to **Unsupervised Learning**.

---

# 🧠 What is Unsupervised Learning?

Unsupervised Learning works with data where a predefined target variable is not supplied.

The algorithm attempts to discover:

- Hidden patterns
- Similar observations
- Natural groups
- Relationships
- Data structure

In supervised learning we usually have:

```text
X → Input Features
y → Target
```

In DBSCAN:

```text
X → Input Features
```

is sufficient.

There is no target variable.

---

# 🔍 What is Clustering?

Clustering is an Unsupervised Machine Learning technique used to divide observations into groups based on similarity.

The general process is:

```text
Customer Data
      ↓
Measure Relationships
      ↓
Find Similar Observations
      ↓
Create Groups
      ↓
Interpret Clusters
```

Different clustering algorithms use different definitions of similarity.

---

# 🔵 What is DBSCAN?

**DBSCAN** stands for:

> **Density-Based Spatial Clustering of Applications with Noise**

DBSCAN is a density-based clustering algorithm.

Unlike K-Means, DBSCAN does not require us to specify the number of clusters beforehand.

Instead, it identifies areas where observations are densely packed together.

It can also identify observations that do not belong to any sufficiently dense region.

These observations are called:

```text
Noise
```

---

# ⭐ Why DBSCAN?

DBSCAN is useful when:

- The number of clusters is unknown.
- Clusters are defined by density.
- Noise or outliers need to be identified.
- Clusters may not have simple spherical shapes.
- We want to separate dense regions from sparse regions.

This makes DBSCAN fundamentally different from K-Means.

---

# 🎯 Main DBSCAN Parameters

DBSCAN mainly depends on two parameters:

```text
eps
min_samples
```

They control how DBSCAN determines whether a region is sufficiently dense.

---

# 📏 What is `eps`?

`eps` represents the radius of the neighborhood around an observation.

In the notebook:

```python
eps_value = 0.5
```

This means DBSCAN considers observations within the specified radius when determining local density.

Because the project scales the features first, `eps` is applied in the standardized feature space.

---

# 🔢 What is `min_samples`?

`min_samples` specifies the minimum number of observations required within the neighborhood for a point to qualify as a core point.

The notebook uses:

```python
min_samples_value = 5
```

A higher value generally requires a denser region to form a cluster.

A lower value can allow smaller or less dense groups to become clusters.

---

# 🟢 Core Point

A **core point** has enough observations within its `eps` neighborhood to satisfy `min_samples`.

Core points form the dense foundation of DBSCAN clusters.

---

# 🟡 Border Point

A border point is close enough to a core point to be associated with a cluster but does not itself satisfy the density requirement to be a core point.

---

# ⚫ Noise Point

A noise point does not belong to a sufficiently dense region and is not assigned to a cluster.

In Scikit-learn DBSCAN, noise points are represented by:

```text
-1
```

This is one of the defining features of DBSCAN.

---

# 🔄 How DBSCAN Works

The general DBSCAN process is:

```text
Choose eps
     ↓
Choose min_samples
     ↓
Select an unvisited point
     ↓
Find its neighborhood
     ↓
Check density
     ↓
Core Point?
  ┌──┴──┐
 Yes    No
  ↓      ↓
Expand   Possible
Cluster  Border/Noise
  ↓
Repeat
  ↓
Final Clusters + Noise
```

DBSCAN grows clusters from dense regions.

---

# 🧩 Density-Based Clustering

The main idea behind DBSCAN is:

```text
High Density → Cluster
Low Density  → Noise / Sparse Region
```

This is different from K-Means, where every observation is assigned to the nearest centroid.

---

# ⚖️ Feature Scaling

DBSCAN uses distances to determine neighborhoods.

Therefore, feature scaling is important.

The project uses:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The standardized features have comparable scales.

Without scaling, one feature could dominate the neighborhood calculation.

---

# 📂 Dataset Information

## Dataset Name

```text
Mall_Customers.csv
```

The standard dataset contains:

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Genre` | Customer gender |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Customer spending behavior score |

The project uses:

```text
Annual Income (k$)
Spending Score (1-100)
```

for DBSCAN.

---

# 📍 Dataset Path

The notebook uses:

```python
df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\Mall_Customers.csv"
)
```

If the CSV is stored beside the notebook:

```python
df = pd.read_csv("Mall_Customers.csv")
```

---

# 🎯 Selected Features

The project selects:

```python
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]
```

These are the same features used in the previous K-Means and Hierarchical Clustering projects, making direct comparison easier.

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
DBSCAN-Clustering/
│
├── DBSCAN Clustering.ipynb
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
Explore Dataset
       ↓
Check Missing Values
       ↓
Check Duplicate Records
       ↓
Select Features
       ↓
Standardize Features
       ↓
Check Scaling
       ↓
Visualize Scaled Data
       ↓
Set eps
       ↓
Set min_samples
       ↓
Create DBSCAN Model
       ↓
Train Model
       ↓
Generate Cluster Labels
       ↓
Count Clusters
       ↓
Count Noise Points
       ↓
Add Labels to Dataset
```

---

# 📖 Step-by-Step Notebook Explanation

The following section explains all **20 cells** of the DBSCAN notebook in the same order as the implementation.

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
from sklearn.cluster import DBSCAN

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

### Purpose

This cell imports the libraries required for numerical operations, data manipulation, visualization, feature scaling, and DBSCAN clustering.

`DBSCAN` provides the clustering algorithm and `StandardScaler` is used because DBSCAN is distance-based.

---

# 🔹 Cell 2 – Load Dataset

```python
# Loading the Mall Customers dataset

df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\Mall_Customers.csv"
)

# Displaying the first five rows

df.head()
```

### Purpose

This cell loads the CSV into a Pandas DataFrame named `df`.

`df.head()` displays the first five records and verifies that the dataset was loaded.

---

# 🔹 Cell 3 – Dataset Information

```python
# Displaying dataset information

df.info()
```

### Purpose

This cell displays:

- Number of observations
- Number of columns
- Column names
- Data types
- Non-null values

It provides an initial understanding of the dataset structure.

---

# 🔹 Cell 4 – Statistical Summary

```python
# Statistical summary of numerical columns

df.describe()
```

### Purpose

This cell generates descriptive statistics including count, mean, standard deviation, minimum, quartiles, and maximum for numerical columns.

---

# 🔹 Cell 5 – Check Missing Values

```python
# Checking for missing values

df.isnull().sum()
```

### Purpose

This cell checks every column for missing values.

Missing observations should be investigated before applying a clustering algorithm.

---

# 🔹 Cell 6 – Dataset Dimensions

```python
# Displaying dataset dimensions

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
```

### Purpose

`df.shape[0]` gives the number of rows and `df.shape[1]` gives the number of columns.

---

# 🔹 Cell 7 – Check Duplicate Records

```python
# Checking for duplicate records

print("Duplicate Rows :", df.duplicated().sum())
```

### Purpose

This cell checks for duplicate observations.

Duplicate observations can affect density calculations because repeated points may increase local density.

---

# 🔹 Cell 8 – Display Column Names

```python
# Displaying column names

print("Columns in the dataset:")
print(df.columns.tolist())
```

### Purpose

This verifies the exact DataFrame column names before selecting the clustering features.

---

# 🔹 Cell 9 – Select Features for DBSCAN

```python
# Selecting features for DBSCAN clustering

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("Selected Features:")
display(X.head())

print("
Features Shape:", X.shape)
```

### Purpose

This selects the two variables used for DBSCAN:

```text
Annual Income (k$)
Spending Score (1-100)
```

The selected features are stored in `X`.

There is no target variable because DBSCAN is an unsupervised algorithm.

---

# 🔹 Cell 10 – Display Selected Features

```python
# Displaying the selected clustering features

display(X)
```

### Purpose

This displays the selected input features and verifies that the correct columns will be used.

---

# 🔹 Cell 11 – Standardize Features

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

### Purpose

This standardizes the selected features using `StandardScaler`.

The transformed data is stored in `X_scaled`.

Scaling is important because DBSCAN uses distance-based neighborhoods.

---

# 🔹 Cell 12 – Check Scaled Feature Statistics

```python
# Checking the mean and standard deviation after scaling

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

### Purpose

This cell verifies the effect of standardization by checking the mean and standard deviation of the scaled features.

---

# 🔹 Cell 13 – Visualize Scaled Data

```python
# Visualizing the scaled customer data

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1]
)

plt.title("Scaled Customer Data")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")

plt.grid()
plt.show()
```

### Purpose

This cell creates a scatter plot of the standardized customer data.

The plot helps us visually inspect the density structure before applying DBSCAN.

---

# 🔹 Cell 14 – Set DBSCAN Parameters

```python
# Setting DBSCAN parameters

eps_value = 0.5
min_samples_value = 5

print("Epsilon:", eps_value)
print("Minimum Samples:", min_samples_value)
```

### Purpose

This cell defines the two primary DBSCAN parameters.

```text
eps = 0.5
min_samples = 5
```

`eps` controls neighborhood radius and `min_samples` controls the minimum density requirement.

---

# 🔹 Cell 15 – Create DBSCAN Model

```python
# Creating the DBSCAN model

dbscan = DBSCAN(
    eps=eps_value,
    min_samples=min_samples_value
)

print("DBSCAN model created successfully!")
```

### Purpose

This creates the DBSCAN model using the selected parameter values.

---

# 🔹 Cell 16 – Train DBSCAN Model

```python
# Training the DBSCAN model

dbscan.fit(X_scaled)

print("DBSCAN model trained successfully!")
```

### Purpose

This fits DBSCAN to the standardized customer data.

The algorithm examines local density and identifies cluster membership and noise.

---

# 🔹 Cell 17 – Generate Cluster Labels

```python
# Generating cluster labels

cluster_labels = dbscan.labels_

print("Cluster Labels:")
print(cluster_labels)
```

### Purpose

This retrieves the labels generated by DBSCAN.

The labels are stored in `cluster_labels`.

DBSCAN uses:

```text
-1
```

to represent noise.

Other integer values represent actual clusters.

---

# 🔹 Cell 18 – Identify Number of Clusters

```python
# Identifying the number of clusters

unique_labels = set(cluster_labels)

n_clusters = len(
    unique_labels - {-1}
)

print("Number of Clusters:", n_clusters)
```

### Purpose

This counts the actual clusters while excluding the special noise label `-1`.

This demonstrates an important difference from K-Means: the number of clusters is discovered by DBSCAN rather than supplied through `n_clusters`.

---

# 🔹 Cell 19 – Count Noise Points

```python
# Counting noise points

noise_count = list(cluster_labels).count(-1)

print("Number of Noise Points:", noise_count)
```

### Purpose

This counts observations classified as noise.

The noise label in Scikit-learn DBSCAN is:

```text
-1
```

---

# 🔹 Cell 20 – Add Cluster Labels to Dataset

```python
# Adding DBSCAN cluster labels to the original dataset

df["Cluster"] = cluster_labels

display(df.head(10))
```

### Purpose

This adds a new `Cluster` column to the original DataFrame.

The final dataset can now be inspected together with each customer's DBSCAN cluster assignment.

---

# 📊 Understanding DBSCAN Labels

DBSCAN labels have special meanings:

```text
-1 → Noise
 0 → Cluster 0
 1 → Cluster 1
 2 → Cluster 2
...
```

The exact number of clusters depends on the density structure and selected `eps` and `min_samples` values.

---

# 🔵 DBSCAN vs K-Means

| Feature | K-Means | DBSCAN |
|---|---|---|
| Learning Type | Unsupervised | Unsupervised |
| Main Idea | Centroid-based | Density-based |
| Need K beforehand | Yes | No |
| Noise Detection | No special label | Yes |
| Outlier Label | None | `-1` |
| Main Parameters | `n_clusters` | `eps`, `min_samples` |
| Cluster Formation | Nearest centroid | Dense regions |
| Scaling | Important | Important |

---

# 🌳 DBSCAN vs Hierarchical Clustering

| Feature | Hierarchical | DBSCAN |
|---|---|---|
| Main Idea | Hierarchy | Density |
| Main Visualization | Dendrogram | Cluster plot |
| Requires K | Not for hierarchy construction | No |
| Noise Detection | Not a primary feature | Yes |
| Main Parameters | Linkage, distance, cut level | `eps`, `min_samples` |
| Cluster Formation | Repeated merging | Density expansion |
| Scaling | Important | Important |

---

# 🎯 Why DBSCAN Does Not Need K

K-Means requires:

```python
KMeans(n_clusters=5)
```

The hierarchical model eventually uses a chosen cluster count.

DBSCAN instead uses:

```python
DBSCAN(
    eps=0.5,
    min_samples=5
)
```

The number of clusters becomes an output discovered from the density structure.

---

# 📏 Effect of `eps`

A smaller `eps` creates smaller neighborhoods.

This can lead to:

- More noise
- Smaller clusters
- More fragmented groups

A larger `eps` creates larger neighborhoods.

This can lead to:

- Fewer noise points
- Larger clusters
- Different groups being merged

Conceptually:

```text
Small eps
→ Small neighborhoods
→ Strict density requirement

Large eps
→ Large neighborhoods
→ More points connected
```

---

# 🔢 Effect of `min_samples`

A smaller `min_samples` makes it easier for a region to satisfy the density requirement.

A larger `min_samples` requires more observations to form a dense region.

Therefore, both parameters need to be considered together.

---

# ⚠️ Parameter Selection

The notebook currently uses:

```python
eps_value = 0.5
min_samples_value = 5
```

These are the values used in the current implementation.

In a real project, parameter selection should be investigated using the structure of the data rather than assuming one combination is always optimal.

---

# 🌍 Real-World Applications

DBSCAN can be used for:

- Customer segmentation
- Geographic analysis
- Anomaly detection
- Fraud detection
- Image processing
- Spatial data analysis
- Network analysis
- Behavioral analysis
- Pattern discovery

---

# ✅ Advantages of DBSCAN

- Does not require the number of clusters beforehand.
- Can identify noise and outliers.
- Uses density instead of centroids.
- Can detect irregularly shaped clusters.
- Useful for exploratory analysis.
- Uses only a small number of main parameters.
- Automatically discovers the number of clusters.

---

# ❌ Limitations of DBSCAN

- Sensitive to the choice of `eps`.
- Sensitive to the choice of `min_samples`.
- Can struggle when clusters have very different densities.
- Distance calculations require appropriate scaling.
- Performance can depend on the selected distance metric.
- High-dimensional data can make density estimation more difficult.

---

# 📌 DBSCAN Core Concepts Summary

```text
DBSCAN
  │
  ├── eps
  │     └── Neighborhood radius
  │
  ├── min_samples
  │     └── Minimum density requirement
  │
  ├── Core Point
  │     └── Dense neighborhood
  │
  ├── Border Point
  │     └── Near a core point
  │
  └── Noise
        └── Label = -1
```

---

# 📁 Complete Notebook Structure

```text
Cell 1
Import Libraries

Cell 2
Load Dataset

Cell 3
Dataset Information

Cell 4
Statistical Summary

Cell 5
Check Missing Values

Cell 6
Dataset Dimensions

Cell 7
Check Duplicate Records

Cell 8
Display Column Names

Cell 9
Select Clustering Features

Cell 10
Display Selected Features

Cell 11
Standardize Features

Cell 12
Check Scaled Feature Statistics

Cell 13
Visualize Scaled Data

Cell 14
Set DBSCAN Parameters

Cell 15
Create DBSCAN Model

Cell 16
Train DBSCAN Model

Cell 17
Generate Cluster Labels

Cell 18
Identify Number of Clusters

Cell 19
Count Noise Points

Cell 20
Add Cluster Labels to Dataset
```

---

# 🔄 End-to-End DBSCAN Workflow

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
Scaled Customer Data
          ↓
Choose eps
          ↓
Choose min_samples
          ↓
Create DBSCAN
          ↓
Fit Model
          ↓
Identify Dense Regions
          ↓
Generate Cluster Labels
          ↓
Identify Noise
          ↓
Count Clusters
          ↓
Customer Segmentation
```

---

# 📊 Final Model Configuration

The notebook creates:

```python
dbscan = DBSCAN(
    eps=eps_value,
    min_samples=min_samples_value
)
```

with:

```text
eps = 0.5
min_samples = 5
```

The model is trained using:

```python
dbscan.fit(X_scaled)
```

and labels are obtained using:

```python
cluster_labels = dbscan.labels_
```

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand Unsupervised Learning.
- Understand clustering.
- Understand DBSCAN.
- Understand density-based clustering.
- Understand `eps`.
- Understand `min_samples`.
- Understand core points.
- Understand border points.
- Understand noise points.
- Understand DBSCAN labels.
- Understand why noise is represented by `-1`.
- Apply feature scaling.
- Use `StandardScaler`.
- Create a DBSCAN model using Scikit-learn.
- Train DBSCAN.
- Generate cluster labels.
- Count clusters automatically.
- Count noise points.
- Perform customer segmentation.
- Compare DBSCAN with K-Means.
- Compare DBSCAN with Hierarchical Clustering.

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

Open:

```text
DBSCAN Clustering.ipynb
```

Run the cells sequentially.

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

DBSCAN is an important density-based Unsupervised Learning algorithm.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand.

Instead, it uses:

```text
eps
min_samples
```

to identify dense regions.

In this project, the Mall Customers dataset is used with:

```text
Annual Income (k$)
Spending Score (1-100)
```

The features are standardized using `StandardScaler`.

The DBSCAN model is created using:

```text
eps = 0.5
min_samples = 5
```

The model generates cluster labels and uses:

```text
-1
```

to represent noise points.

The notebook also calculates the number of clusters while excluding noise and counts the total number of noise observations.

This project demonstrates a third major clustering approach:

```text
K-Means
→ Centroid-based

Hierarchical Clustering
→ Hierarchy-based

DBSCAN
→ Density-based
```

DBSCAN is especially valuable when the number of clusters is unknown and when identifying sparse or unusual observations is important.



# ⭐ Final Takeaway

```text
Unsupervised Learning
        ↓
     Clustering
        ↓
      DBSCAN
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
Choose eps
        ↓
Choose min_samples
        ↓
Find Dense Regions
        ↓
Create Clusters
        ↓
Identify Noise
        ↓
Analyze Customer Groups
```

DBSCAN is a powerful clustering technique because it can discover clusters without requiring the number of clusters beforehand and can explicitly identify observations that do not belong to sufficiently dense regions.
---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project is part of a practical Machine Learning learning journey focused on implementing and understanding algorithms using Python and Scikit-learn.

---
