# 🎯 K-Means Clustering

A comprehensive Machine Learning project demonstrating **K-Means Clustering** using the **Mall Customers** dataset.

This project covers the complete Unsupervised Learning workflow: dataset exploration, data-quality checks, feature selection, feature scaling, the Elbow Method, K-Means model creation, training, cluster assignment, and cluster-center analysis.

The main objective is to segment customers according to **Annual Income (k$)** and **Spending Score (1-100)**.

---

# 📑 Table of Contents

- Overview
- What is Machine Learning?
- Types of Machine Learning
- What is Unsupervised Learning?
- What is Clustering?
- What is K-Means Clustering?
- How K-Means Works
- Centroids
- Distance Calculation
- Inertia
- Elbow Method
- Feature Scaling
- Dataset Information
- Dataset Features
- Clustering Features
- Technologies Used
- Project Structure
- Project Workflow
- Step-by-Step Notebook Explanation
- Advantages
- Limitations
- Real-World Applications
- Learning Outcomes
- Conclusion

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that enables computers to learn patterns from data and make predictions or decisions without explicitly programming every rule.

Machine Learning is commonly used in healthcare, finance, banking, marketing, recommendation systems, fraud detection, customer analytics, image recognition, natural language processing, and business intelligence.

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

K-Means Clustering belongs to **Unsupervised Learning**.

---

# 🧠 What is Unsupervised Learning?

Unsupervised Learning works with data where there is no predefined target variable.

The algorithm attempts to discover hidden patterns, structures, or groups within the data.

Examples include:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis

Unlike supervised learning, there is no `y` target variable in this project.

---

# 🔍 What is Clustering?

Clustering is an unsupervised Machine Learning technique used to divide observations into groups based on similarity.

Each group is called a **cluster**.

Observations within the same cluster should be relatively similar, while observations in different clusters should be relatively different.

```text
Customer Data
      ↓
Find Similarities
      ↓
Discover Patterns
      ↓
Create Groups
      ↓
Interpret Clusters
```

---

# 🎯 What is K-Means Clustering?

**K-Means Clustering** is an unsupervised Machine Learning algorithm that divides data into **K clusters**.

The value of `K` represents the number of clusters.

For example:

```text
K = 5
```

means that the algorithm creates five customer groups.

K-Means attempts to make observations within a cluster as similar as possible while keeping different clusters separated.

---

# 🔄 How K-Means Works

K-Means follows an iterative process:

```text
Choose K
   ↓
Initialize Centroids
   ↓
Assign Points to Nearest Centroid
   ↓
Calculate New Centroids
   ↓
Reassign Points
   ↓
Repeat
   ↓
Final Clusters
```

The algorithm continues until the cluster assignments stabilize or the algorithm reaches its stopping condition.

---

# 📍 Centroids

A centroid represents the center of a cluster.

For this project, each centroid has two values:

```text
Annual Income (k$)
Spending Score (1-100)
```

The centroid is repeatedly updated during K-Means training.

---

# 📏 Distance Calculation

K-Means uses distance to determine which cluster a customer belongs to.

A common distance measure is Euclidean distance.

For two points:

```text
A = (x₁, y₁)
B = (x₂, y₂)
```

the Euclidean distance is:

```text
Distance = √((x₂ − x₁)² + (y₂ − y₁)²)
```

A customer is assigned to the nearest centroid.

---

# 🔁 Assignment and Update Steps

During the assignment step, every observation is assigned to its nearest centroid.

After assignments are made, K-Means calculates new centroids using the mean of the observations in each cluster.

The process repeats:

```text
Assignment
    ↓
Update Centroids
    ↓
Assignment
    ↓
Update Centroids
    ↓
...
```

This iterative process attempts to minimize within-cluster variation.

---

# 📉 Inertia

**Inertia** measures the total within-cluster squared distance between observations and their assigned centroids.

Conceptually:

```text
Inertia =
Sum of squared distances between
observations and assigned centroids
```

Lower inertia means observations are, overall, closer to their assigned centroids.

However, inertia generally decreases as K increases, so the smallest inertia is not automatically the best choice.

---

# 📐 Elbow Method

The **Elbow Method** helps select a reasonable value of K.

This project calculates inertia for:

```text
K = 1
K = 2
K = 3
...
K = 10
```

The values are plotted and the point where the curve begins to flatten is considered the elbow.

For this project, the selected value is:

```text
K = 5
```

---

# ⚖️ Feature Scaling

K-Means is distance-based, so feature scaling is important.

If one feature has a much larger numerical range than another, it can dominate the distance calculation.

The project uses:

```python
StandardScaler
```

to standardize the selected features before training K-Means.

The standardization is conceptually:

```text
z = (x − mean) / standard deviation
```

---

# 📂 Dataset Information

**Dataset Name:** `Mall_Customers.csv`

The dataset contains customer demographic and spending-related information.

The main clustering analysis uses:

```text
Annual Income (k$)
Spending Score (1-100)
```

The local dataset path used in the notebook is:

```text
C:\Users\91958\Desktop\Machine Learning\datasets\Mall_Customers.csv
```

---

# 🧩 Dataset Features

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Genre` | Customer gender in the standard dataset |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Customer spending behavior score |

---

# 🎯 Clustering Features

The project selects two features:

```text
Annual Income (k$)
Spending Score (1-100)
```

They are selected using:

```python
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]
```

No target variable is created because this is an unsupervised learning problem.

---

# 🛠 Technologies Used

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
K-Means-Clustering/
│
├── K-Means Clustering.ipynb
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
Select Clustering Features
       ↓
Standardize Features
       ↓
Calculate Inertia for K = 1 to 10
       ↓
Plot Elbow Curve
       ↓
Select K = 5
       ↓
Create K-Means Model
       ↓
Train Model
       ↓
Generate Cluster Labels
       ↓
Add Cluster Labels
       ↓
Calculate Cluster Centers
       ↓
Interpret Customer Segments
```

---

# 📖 Step-by-Step Notebook Explanation

The following sections explain the 20 cells in the K-Means notebook in the same order as the implementation.

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
from sklearn.cluster import KMeans

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

### Purpose

This cell imports the libraries required for numerical operations, data manipulation, visualization, feature scaling, and K-Means clustering.

`NumPy` supports numerical calculations, `Pandas` handles the dataset, `Matplotlib` and `Seaborn` support visualization, `StandardScaler` standardizes features, and `KMeans` provides the clustering algorithm.

The warning filter prevents unnecessary warning messages from cluttering the notebook.

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

### Purpose

This cell loads `Mall_Customers.csv` into a Pandas DataFrame named `df`.

`df.head()` displays the first five rows and provides an initial view of the dataset.

---

## 🔹 Cell 3 – Dataset Information

```python
# Displaying dataset information

df.info()
```

### Purpose

`df.info()` displays the number of entries, columns, data types, and non-null counts.

It is useful for understanding the structure of the dataset before preprocessing.

---

## 🔹 Cell 4 – Statistical Summary

```python
# Statistical summary of numerical columns

df.describe()
```

### Purpose

`df.describe()` generates descriptive statistics for numerical columns, including count, mean, standard deviation, minimum, quartiles, and maximum.

---

## 🔹 Cell 5 – Check Missing Values

```python
# Checking for missing values

df.isnull().sum()
```

### Purpose

This cell checks every column for missing values.

`df.isnull()` identifies missing values and `sum()` counts them column by column.

---

## 🔹 Cell 6 – Dataset Dimensions

```python
# Displaying dataset dimensions

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
```

### Purpose

This cell displays the number of rows and columns in the dataset.

`df.shape[0]` gives the number of rows and `df.shape[1]` gives the number of columns.

---

## 🔹 Cell 7 – Duplicate Records

```python
# Checking for duplicate records

print("Duplicate Rows :", df.duplicated().sum())
```

### Purpose

This cell checks whether duplicate observations exist.

`df.duplicated()` identifies duplicate rows and `sum()` counts them.

---

## 🔹 Cell 8 – Column Names

```python
# Displaying column names

print("Columns in the dataset:")
print(df.columns.tolist())
```

### Purpose

This cell displays all column names.

Checking the exact names is useful before selecting the clustering features.

---

## 🔹 Cell 9 – Select Clustering Features

```python
# Selecting features for clustering

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

### Purpose

This cell selects the two variables used for the main clustering analysis.

The selected features are `Annual Income (k$)` and `Spending Score (1-100)`.

The selected DataFrame is stored in `X`.

Unlike supervised learning, no `y` target is created.

---

## 🔹 Cell 10 – Display Selected Features

```python
# Displaying the selected clustering features

display(X)
```

### Purpose

This cell displays the complete selected feature DataFrame so that the clustering inputs can be inspected before scaling.

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

### Purpose

This cell standardizes the selected features using `StandardScaler`.

The scaled data is stored in `X_scaled`.

Scaling is important because K-Means uses distance calculations.

---

## 🔹 Cell 12 – Check Scaled Feature Statistics

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

### Purpose

This cell verifies the effect of standardization by checking the mean and standard deviation of the scaled features.

---

## 🔹 Cell 13 – Elbow Method Inertia

```python
# Finding the optimal number of clusters using the Elbow Method

inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

print("Inertia values:")
print(inertia)
```

### Purpose

This cell evaluates K values from 1 through 10.

For each K, a K-Means model is fitted and its `inertia_` value is stored.

The resulting list is used to create the Elbow Curve.

---

## 🔹 Cell 14 – Plot Elbow Curve

```python
# Plotting the Elbow Curve

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.grid()
plt.show()
```

### Purpose

This cell plots K against inertia.

The bend in the curve helps identify a practical value of K.

The project selects K = 5 after examining this curve.

---

## 🔹 Cell 15 – Select Optimal K

```python
# Selecting the optimal number of clusters

optimal_k = 5

print("Optimal Number of Clusters:", optimal_k)
```

### Purpose

This cell stores the selected number of clusters in `optimal_k`.

The project uses five clusters.

---

## 🔹 Cell 16 – Create K-Means Model

```python
# Creating the K-Means clustering model

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

print("K-Means model created successfully!")
```

### Purpose

This cell creates the final K-Means model.

`n_clusters` uses the selected K, `random_state=42` supports reproducibility, and `n_init=10` uses multiple centroid initializations.

---

## 🔹 Cell 17 – Train K-Means Model

```python
# Training the K-Means model

kmeans.fit(X_scaled)

print("K-Means model trained successfully!")
```

### Purpose

This cell trains K-Means on the standardized customer features.

During training, K-Means assigns points to centroids and repeatedly updates the centroid locations.

---

## 🔹 Cell 18 – Assign Cluster Labels

```python
# Assigning cluster labels to each customer

cluster_labels = kmeans.labels_

print("Cluster Labels:")
print(cluster_labels)
```

### Purpose

After training, each customer receives a cluster label.

The labels are stored in `cluster_labels`.

With five clusters, labels are represented by integers from 0 to 4.

---

## 🔹 Cell 19 – Add Cluster Labels

```python
# Adding cluster labels to the original dataset

df["Cluster"] = cluster_labels

display(df.head(10))
```

### Purpose

This cell adds the generated cluster assignment to the original DataFrame as a new `Cluster` column.

This allows customer information and cluster membership to be examined together.

---

## 🔹 Cell 20 – Display Cluster Centers

```python
# Displaying cluster centers

cluster_centers_scaled = kmeans.cluster_centers_

cluster_centers = scaler.inverse_transform(
    cluster_centers_scaled
)

cluster_centers_df = pd.DataFrame(
    cluster_centers,
    columns=X.columns
)

cluster_centers_df.index.name = "Cluster"

display(cluster_centers_df)
```

### Purpose

K-Means stores cluster centers in the scaled feature space.

`scaler.inverse_transform()` converts them back to the original income and spending-score scale.

The resulting DataFrame makes it possible to interpret each customer segment using the original units.

---

# 📊 Understanding Cluster Centers

Cluster centers are the most useful output for interpreting the customer segments.

Each center contains:

```text
Annual Income (k$)
Spending Score (1-100)
```

For example, a cluster with high income and high spending can be interpreted as a high-value customer segment, while a cluster with high income and low spending may represent customers who have purchasing power but are not spending heavily.

The exact interpretation should be based on the actual cluster-center values produced by the notebook.

---

# 🏷️ Cluster Labels vs Cluster Meaning

K-Means may return labels such as:

```text
Cluster 0
Cluster 1
Cluster 2
Cluster 3
Cluster 4
```

These numbers are only identifiers.

`Cluster 0` is not automatically better or worse than `Cluster 4`.

The meaning of each cluster must be determined from its centroid and customer characteristics.

---

# 🌍 Real-World Applications

K-Means Clustering can be used for:

- Customer segmentation
- Marketing segmentation
- Retail analytics
- Recommendation systems
- Financial customer grouping
- Healthcare grouping
- Image compression
- Behavioral analysis
- Product segmentation
- Business intelligence

---

# ✅ Advantages of K-Means

- Simple to understand.
- Easy to implement.
- Computationally efficient for many datasets.
- Works well with numerical data.
- Easy to visualize.
- Useful for customer segmentation.
- Produces interpretable cluster centers.
- Scales well for many practical applications.

---

# ❌ Limitations of K-Means

- The number of clusters must be selected.
- Sensitive to initial centroid placement.
- Sensitive to outliers.
- Requires suitable numerical features.
- Distance calculations make scaling important.
- Can struggle with non-spherical cluster structures.
- Results depend on feature selection.
- Different initializations can sometimes produce different assignments.

---

# ⚠️ Importance of Feature Scaling

Because K-Means is distance-based, scaling is an important preprocessing step.

The project uses:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This prevents a feature with a larger numerical scale from dominating the distance calculation.

---

# 📌 Difference Between Supervised and Unsupervised Learning

| Feature | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Target Variable | Present | Not required |
| Main Goal | Prediction | Pattern discovery |
| Examples | Regression, Classification | Clustering, PCA |
| Labels | Required | Not required |
| This Project | No | Yes |

---

# 📌 Difference Between Regression and Clustering

| Feature | Regression | K-Means Clustering |
|---|---|---|
| Learning Type | Supervised | Unsupervised |
| Target Variable | Required | Not required |
| Output | Numerical value | Cluster label |
| Main Goal | Prediction | Grouping |
| Example | Insurance charges | Customer segments |
| Typical Evaluation | MAE, MSE, RMSE, R² | Inertia and cluster analysis |

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand Unsupervised Learning.
- Understand clustering.
- Understand K-Means Clustering.
- Understand centroids.
- Understand distance-based clustering.
- Understand the assignment step.
- Understand centroid updates.
- Understand inertia.
- Understand the Elbow Method.
- Select an appropriate K.
- Apply feature scaling.
- Use `StandardScaler`.
- Build a K-Means model with Scikit-learn.
- Generate cluster labels.
- Add labels to a DataFrame.
- Analyze cluster centers.
- Perform customer segmentation.
- Interpret clustering results.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/K-Means-Clustering.git
```

Navigate to the project directory:

```bash
cd K-Means-Clustering
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
K-Means Clustering.ipynb
```

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

Install the libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

---

# 📍 Dataset Path

The notebook currently uses:

```python
df = pd.read_csv(
    r"C:\Users\91958\Desktop\Machine Learning\datasets\Mall_Customers.csv"
)
```

If the CSV is placed in the same directory as the notebook, use:

```python
df = pd.read_csv("Mall_Customers.csv")
```

---

# 🧪 Reproducibility

The K-Means implementation uses:

```python
random_state=42
```

and:

```python
n_init=10
```

The same settings are also used while calculating inertia for the Elbow Method.

---

# 📊 Final Model Configuration

```python
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)
```

The project selects:

```text
Number of Clusters = 5
Random State = 42
Number of Initializations = 10
```

The final features are:

```text
Annual Income (k$)
Spending Score (1-100)
```

---

# 🏁 Conclusion

K-Means Clustering is an important introductory algorithm for Unsupervised Learning.

Unlike regression and classification, K-Means does not require a target variable. Instead, it discovers groups based on similarity.

In this project, the Mall Customers dataset is explored and two customer attributes are selected:

```text
Annual Income (k$)
Spending Score (1-100)
```

The features are standardized with `StandardScaler`, because K-Means relies on distance calculations.

The Elbow Method evaluates K values from 1 to 10. The project then selects:

```text
K = 5
```

A final K-Means model is trained, cluster labels are assigned to customers, and cluster centers are converted back to the original feature scale for interpretation.

The project demonstrates the practical workflow of Unsupervised Learning and provides a foundation for customer segmentation.

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project is part of a Machine Learning learning journey focused on understanding algorithms through practical Python implementations.

---

