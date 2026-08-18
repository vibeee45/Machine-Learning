# 🧠 Gaussian Mixture Model (GMM) Clustering

A comprehensive Machine Learning project demonstrating **Gaussian Mixture Model (GMM) Clustering** using the **Wine dataset** provided by Scikit-learn.

This project follows a complete unsupervised learning workflow.

The notebook begins by importing the required libraries.

It then loads the Wine dataset.

The dataset is converted into a Pandas DataFrame.

The data is explored using information, descriptive statistics, missing-value checks, dimensions, and feature names.

All 13 Wine features are used as clustering inputs.

The features are standardized before applying GMM.

A Gaussian Mixture Model with three initial components is fitted.

Cluster labels are generated.

Cluster probabilities are calculated.

AIC and BIC are calculated.

Different component counts from 1 through 7 are compared.

The final number of components is selected using the lowest BIC.

The final GMM is trained with two components.

Final cluster assignments and probabilities are inspected.

Low-confidence observations are checked.

PCA is used to reduce the standardized data to two dimensions for visualization.

The GMM clusters are visualized in PCA space.

The discovered clusters are compared with the original Wine class labels.

Finally, the Silhouette Score is calculated.

The notebook therefore demonstrates both the mathematical ideas and practical implementation of probabilistic clustering.

---

# 📑 Table of Contents

- Project Overview
- What is Machine Learning?
- What is Unsupervised Learning?
- What is Clustering?
- What is a Gaussian Mixture Model?
- Why GMM?
- Gaussian Distribution
- Mixture Distribution
- Probability Density
- Soft Clustering
- Responsibilities
- Expectation-Maximization
- E-Step
- M-Step
- Means
- Covariance
- Mixing Weights
- Likelihood
- AIC
- BIC
- Model Selection
- PCA
- Silhouette Score
- GMM vs K-Means
- Dataset Information
- Dataset Features
- Technologies Used
- Project Structure
- Notebook Workflow
- Cell-by-Cell Explanation
- Results
- Cluster Interpretation
- Advantages
- Limitations
- Applications
- Installation
- Requirements
- Learning Outcomes
- Conclusion
- Author

---

# 📌 Project Overview

The purpose of this project is to understand how a **Gaussian Mixture Model** can discover groups inside an unlabeled dataset.

The Wine dataset contains chemical measurements for wine samples.

The notebook intentionally treats the dataset as an unsupervised clustering problem.

The original Wine class labels are not used to train the GMM.

Instead, the GMM receives the standardized feature matrix.

The algorithm estimates Gaussian components from the feature distribution.

Each observation receives a cluster assignment.

Each observation also receives a probability for every cluster.

This probabilistic output is one of the most important characteristics of GMM.

The notebook starts with three components for the initial GMM.

It later evaluates component counts from one through seven.

The evaluation uses AIC and BIC.

The lowest BIC occurs at two components in the executed notebook.

Therefore, the final GMM uses two components.

The final model produces 122 observations in Cluster 0.

The final model produces 56 observations in Cluster 1.

The final probability matrix has shape `(178, 2)`.

No observations fall below the notebook's clustering-confidence threshold of 0.60.

PCA is then used only for visualization.

The final notebook also compares the GMM assignments against the original Wine class labels.

The comparison is exploratory because the original labels are not used to fit the unsupervised model.

---

# 🤖 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that enables computers to learn useful patterns from data.

Instead of writing every rule manually, a Machine Learning algorithm learns relationships from examples or data structure.

Machine Learning can be divided into several major categories.

The most common categories are supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning uses labeled observations.

Unsupervised learning works without target labels during model training.

Reinforcement learning learns through interactions, rewards, and penalties.

GMM is an **unsupervised learning** algorithm.

The goal in this project is not to predict the original Wine class directly.

The goal is to discover groups in the feature space.

---

# 🔍 What is Unsupervised Learning?

Unsupervised learning works with data where the target output is not supplied to the algorithm.

The algorithm searches for structure within the input data.

Typical tasks include clustering, dimensionality reduction, and density estimation.

A simplified workflow is:

```text
Input Data
    ↓
Preprocessing
    ↓
Unsupervised Algorithm
    ↓
Hidden Structure
    ↓
Clusters / Representation
```

GMM is useful because it provides a probabilistic description of that hidden structure.

Instead of only saying that a sample belongs to a cluster, GMM can report how strongly the sample belongs to every component.

---

# 🎯 What is Clustering?

Clustering is the process of grouping similar observations.

A clustering algorithm attempts to make observations inside the same group relatively similar.

It also attempts to keep different groups distinguishable.

Clustering is useful when predefined labels are unavailable.

Examples include customer segmentation, document grouping, anomaly exploration, image segmentation, and biological data analysis.

Different clustering algorithms make different assumptions about the shape and structure of groups.

K-Means uses distances to centroids.

Hierarchical clustering creates a hierarchy of groups.

DBSCAN identifies dense regions.

GMM models groups as probability distributions.

---

# 🧠 What is a Gaussian Mixture Model?

A **Gaussian Mixture Model**, commonly abbreviated as GMM, is a probabilistic model that represents data as a mixture of Gaussian distributions.

Each Gaussian component can be interpreted as a cluster.

For a model with `K` components, the probability density can be written conceptually as:

```text
p(x) = Σ πₖ N(x | μₖ, Σₖ)
```

Here:

- `K` is the number of Gaussian components.
- `πₖ` is the mixing weight.
- `μₖ` is the mean vector.
- `Σₖ` is the covariance matrix.
- `N` represents a Gaussian density.

The mixing weights describe the contribution of the components.

The mean describes the center of a component.

The covariance describes the spread and orientation.

The complete mixture combines all Gaussian components into one probability model.

---

# ⭐ Why Use GMM?

GMM is particularly useful when clusters overlap.

It is also useful when clusters are not perfectly spherical.

GMM can model elliptical distributions.

It can represent different covariance structures.

It provides probabilities rather than only hard assignments.

This makes GMM more expressive than simple centroid-based clustering in many situations.

The notebook uses GMM because the Wine dataset contains multiple chemical measurements whose distributions can be represented through a probabilistic mixture model.

---

# 📈 Gaussian Distribution

A Gaussian distribution is also called a Normal distribution.

In one dimension, it is commonly visualized as a bell-shaped curve.

The location of the curve is controlled by its mean.

The spread is controlled by its variance.

In multiple dimensions, covariance determines how features vary together.

A multivariate Gaussian therefore has:

- A mean vector.
- A covariance matrix.

The GMM combines several such Gaussian distributions.

---

# 🧩 Mixture Distribution

A mixture distribution combines several probability distributions.

In GMM, each component is Gaussian.

For example, a three-component model can be represented as:

```text
Dataset
   │
   ├── Gaussian Component 1
   │
   ├── Gaussian Component 2
   │
   └── Gaussian Component 3
```

The complete model combines these components.

The model estimates the parameters of each component from the data.

The parameters are learned during model fitting.

The notebook first creates a three-component model.

It later tests seven possible component counts.

The final selection is based on BIC.

---

# 🎲 Probability Density

Probability density describes how compatible an observation is with a probability distribution.

For every observation, GMM evaluates its relationship with every Gaussian component.

An observation can therefore have values such as:

```text
Cluster 0 → 0.20
Cluster 1 → 0.70
Cluster 2 → 0.10
```

The largest value represents the most probable component.

The complete probability vector is still useful because it describes uncertainty.

---

# 🌫️ Soft Clustering

GMM performs **soft clustering**.

Soft clustering means that an observation can have partial membership in several clusters.

For example:

```text
Observation A

Cluster 0 → 0.15
Cluster 1 → 0.80
Cluster 2 → 0.05
```

The predicted cluster would normally be Cluster 1.

However, the probability values show that the model has also considered the other components.

This is different from hard clustering.

Hard clustering assigns one categorical label.

Soft clustering provides a probability distribution.

---

# 🔗 Responsibilities

The probability assigned to a component for a particular observation is often called its responsibility.

Responsibilities are central to the Expectation-Maximization algorithm.

They represent how strongly each Gaussian component explains an observation.

High responsibility means that the component explains the observation well.

Low responsibility means that another component provides a better explanation.

The notebook obtains these probabilities through:

```python
gmm.predict_proba(X_scaled)
```

---

# 🔄 Expectation-Maximization

GMM commonly uses the **Expectation-Maximization**, or EM, algorithm.

EM estimates hidden membership information and updates model parameters iteratively.

The two major stages are:

1. Expectation Step.
2. Maximization Step.

The E-Step estimates responsibilities.

The M-Step updates the Gaussian parameters.

The process repeats until the model converges.

The algorithm is useful because cluster membership is not known beforehand.

The model therefore alternates between estimating memberships and improving component parameters.

---

# 🔹 E-Step

The Expectation Step calculates the responsibility of each Gaussian component for every observation.

Suppose there are two components.

An observation may receive:

```text
Component 0 → 0.90
Component 1 → 0.10
```

Another observation may receive:

```text
Component 0 → 0.45
Component 1 → 0.55
```

The second observation is more ambiguous.

The E-Step uses the current Gaussian parameters to calculate these responsibilities.

---

# 🔹 M-Step

The Maximization Step updates the model parameters using the responsibilities.

The parameters include:

- Component means.
- Covariance matrices.
- Mixing weights.

Observations with higher responsibility contribute more strongly to the corresponding component.

The objective is to improve the likelihood of the observed data.

After the M-Step, another E-Step is performed.

The process continues until convergence.

---

# 📍 Mean

Each Gaussian component has a mean vector.

The mean represents the center of the component in feature space.

For a dataset with 13 features, each Gaussian mean has 13 values.

The mean therefore represents the component's location across all Wine measurements.

Different components can have different mean vectors.

This allows GMM to place Gaussian components in different regions of the feature space.

---

# 📊 Covariance

Covariance describes the relationship between features.

In GMM, covariance determines the shape and orientation of Gaussian components.

A full covariance matrix can model correlations between features.

This allows GMM to represent elliptical clusters.

The covariance structure is one reason GMM can model more flexible clusters than basic K-Means.

---

# ⚖️ Mixing Weights

Every Gaussian component has a mixing weight.

The mixing weight represents the contribution of that component to the overall mixture distribution.

The weights are non-negative.

The weights sum to one.

For example:

```text
Component 0 → 0.65
Component 1 → 0.35
```

Together, they describe how much each component contributes to the overall model.

---

# 📈 Likelihood

Likelihood measures how well the model explains the observed data.

GMM attempts to find parameters that provide a strong likelihood for the observations.

The EM algorithm improves the likelihood during its iterative optimization process.

AIC and BIC use model likelihood together with complexity penalties.

This makes them useful for comparing models with different numbers of components.

---

# 📉 Akaike Information Criterion

AIC stands for **Akaike Information Criterion**.

It evaluates model quality by balancing goodness of fit against model complexity.

The general form is:

```text
AIC = 2k - 2 ln(L)
```

where:

- `k` is the number of estimated parameters.
- `L` is the maximum likelihood.

A lower AIC is generally preferred when comparing candidate models.

In this notebook, the lowest AIC occurs for four components.

The lowest AIC recorded is approximately:

```text
4473.653
```

AIC therefore favors four components in this experiment.

---

# 📉 Bayesian Information Criterion

BIC stands for **Bayesian Information Criterion**.

BIC also balances model fit and complexity.

Its general form is:

```text
BIC = k ln(n) - 2 ln(L)
```

where:

- `k` is the number of estimated parameters.
- `n` is the number of observations.
- `L` is the maximum likelihood.

Lower BIC values are generally preferred.

In this notebook, the lowest BIC occurs for two components.

The lowest recorded BIC is approximately:

```text
5629.573
```

The notebook therefore selects two components for the final GMM.

---

# ⚖️ AIC vs BIC

AIC and BIC both penalize model complexity.

They can sometimes recommend different models.

That happens in this notebook.

AIC selects:

```text
4 components
```

BIC selects:

```text
2 components
```

The notebook uses BIC for the final component selection.

This produces a final model with two components.

This difference is important because model-selection criteria can prefer different complexity levels.

---

# 🧮 Model Selection

Model selection is the process of choosing an appropriate model configuration.

For GMM, one important decision is the number of components.

Too few components can underrepresent the structure.

Too many components can make the model unnecessarily complex.

The notebook tests:

```python
n_components_range = range(1, 8)
```

Therefore, component counts from one through seven are evaluated.

For every value, the model is fitted.

AIC is calculated.

BIC is calculated.

The results are stored in lists.

A comparison DataFrame is then created.

---

# 📊 Component Comparison

The executed notebook produced the following model-selection values:

| Components | AIC | BIC |
|---:|---:|---:|
| 1 | 5410.396 | 5741.302 |
| 2 | 4964.581 | 5629.573 |
| 3 | 4843.250 | 5842.330 |
| 4 | 4473.653 | 5806.820 |
| 5 | 4795.747 | 6463.002 |
| 6 | 4862.213 | 6863.554 |
| 7 | 4582.118 | 6917.547 |

The minimum AIC occurs at four components.

The minimum BIC occurs at two components.

The final notebook chooses two components.

---

# 🧭 PCA

PCA stands for **Principal Component Analysis**.

The Wine dataset has 13 features.

Visualizing all 13 dimensions directly is difficult.

PCA projects the standardized feature space into a smaller number of dimensions.

The notebook uses:

```python
PCA(n_components=2)
```

This creates two principal components.

They are called PC1 and PC2.

The resulting matrix has shape:

```text
(178, 2)
```

PCA is used here for visualization.

It does not train the GMM.

The GMM is trained using the complete standardized feature matrix.

---

# 📐 Silhouette Score

The Silhouette Score evaluates how well observations fit within their assigned clusters.

It considers:

- Similarity to observations in the same cluster.
- Separation from observations in other clusters.

The score generally ranges from -1 to 1.

Higher values indicate better-separated clusters.

The notebook reports:

```text
GMM Silhouette Score:
0.2682319543159216
```

This indicates moderate cluster separation rather than extremely strong separation.

The score should be interpreted together with the other model-selection and visualization results.

---

# ⚖️ GMM vs K-Means

| K-Means | GMM |
|---|---|
| Hard clustering | Soft clustering |
| Uses centroids | Uses Gaussian distributions |
| Uses distance | Uses probability |
| Usually assumes spherical clusters | Can model elliptical clusters |
| Produces one label | Produces probabilities |
| Simpler | More probabilistic |
| Usually faster | Usually more computationally expensive |

K-Means and GMM can both be used for clustering.

The correct choice depends on the structure of the data.

---

# 📂 Dataset Information

The notebook uses the **Wine dataset** from Scikit-learn.

The dataset is loaded using:

```python
from sklearn.datasets import load_wine
```

The dataset contains:

```text
178 observations
13 features
```

All 13 features are numerical.

The notebook does not use a separate CSV file.

The dataset is loaded directly through Scikit-learn.

The Wine dataset also contains original target classes.

Those labels are retained only for later comparison.

They are not supplied to the GMM during training.

---

# 🍷 Wine Dataset Features

The 13 input features are:

1. alcohol
2. malic_acid
3. ash
4. alcalinity_of_ash
5. magnesium
6. total_phenols
7. flavanoids
8. nonflavanoid_phenols
9. proanthocyanins
10. color_intensity
11. hue
12. od280/od315_of_diluted_wines
13. proline

All of these features are used in the GMM.

The notebook copies the DataFrame into:

```python
X = df.copy()
```

This creates the feature matrix used for preprocessing.

---

# 🔢 Dataset Dimensions

The executed notebook reports:

```text
Number of Rows    : 178
Number of Features: 13
```

The DataFrame therefore has 178 rows and 13 feature columns.

All columns contain 178 non-null values.

The notebook's missing-value check returns zero for every feature.

Therefore, no missing-value treatment is required before standardization.

---

# 🧹 Data Preparation

The notebook creates a Pandas DataFrame from the Wine dataset.

The DataFrame contains only the 13 feature columns.

The original target is stored inside the Scikit-learn dataset object.

It is not added to the clustering feature DataFrame.

This is important because the task is unsupervised clustering.

Using the target as an input would introduce label information into the clustering process.

The notebook correctly keeps the feature matrix separate.

---

# 📏 Feature Standardization

The Wine features have very different scales.

For example, proline values are much larger numerically than hue values.

A distance or probability-based algorithm can be influenced by feature scale.

The notebook therefore uses:

```python
scaler = StandardScaler()
```

The transformation is:

```text
z = (x - mean) / standard deviation
```

The standardized feature matrix is stored in:

```python
X_scaled
```

The resulting shape remains:

```text
(178, 13)
```

Scaling changes the numerical scale.

It does not change the number of observations or features.

---

# 🛠 Technologies Used

The notebook uses Python.

It uses NumPy for numerical operations.

It uses Pandas for DataFrame creation and analysis.

It uses Matplotlib for plotting.

It uses Seaborn for scatter plots.

It uses Scikit-learn for the dataset, scaling, GMM, PCA, and Silhouette Score.

The main GMM class is:

```python
GaussianMixture
```

The main preprocessing class is:

```python
StandardScaler
```

The dimensionality-reduction class is:

```python
PCA
```

The clustering evaluation function is:

```python
silhouette_score
```

---

# 📁 Recommended Project Structure

```text
Gaussian-Mixture-Model/
│
├── Gaussian Mixture Model.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Because the Wine dataset is loaded through Scikit-learn, a separate dataset CSV is not required for this notebook.

---

# 📦 Required Libraries

The main libraries are:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

A requirements file can contain:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

The notebook does not require XGBoost or other external Machine Learning packages.

---

# 🔄 Complete Project Workflow

The complete workflow is:

```text
Load Libraries
      ↓
Load Wine Dataset
      ↓
Create DataFrame
      ↓
Inspect Data
      ↓
Check Missing Values
      ↓
Inspect Dimensions
      ↓
List Features
      ↓
Prepare X
      ↓
Standardize X
      ↓
Create Initial GMM
      ↓
Fit Initial GMM
      ↓
Generate Clusters
      ↓
Calculate Probabilities
      ↓
Calculate AIC
      ↓
Calculate BIC
      ↓
Test 1–7 Components
      ↓
Compare AIC and BIC
      ↓
Select Lowest BIC
      ↓
Train Final GMM
      ↓
Generate Final Labels
      ↓
Inspect Probabilities
      ↓
Check Uncertainty
      ↓
Create Final DataFrame
      ↓
Apply PCA
      ↓
Visualize GMM Clusters
      ↓
Compare Original Classes
      ↓
Calculate Silhouette Score
```

---

# 📖 Notebook Cell-by-Cell Explanation

The notebook contains 40 executable code cells.

The final cell is empty.

The following sections explain the executable cells in the same order as the notebook.

---

# 🔹 Cell 1 — Import Required Libraries

The first cell imports the core Python libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

NumPy is used for numerical operations.

Pandas is used for DataFrame operations.

Matplotlib is used for plotting.

Seaborn is used for statistical visualization.

The cell then imports Scikit-learn tools:

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
```

`load_wine` provides the dataset.

`StandardScaler` standardizes the features.

`GaussianMixture` implements the GMM algorithm.

Warnings are disabled for cleaner notebook output.

---

# 🔹 Cell 2 — Load the Wine Dataset

The second cell loads the Wine dataset.

```python
wine = load_wine()
```

The dataset object contains the feature data.

It also contains feature names and the original target information.

The notebook prints:

```text
Wine dataset loaded successfully!
```

The original target is not used for GMM training.

It is only used later for exploratory comparison.

---

# 🔹 Cell 3 — Create the Pandas DataFrame

The third cell creates the main DataFrame.

```python
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)
```

This is an important step because the notebook's later operations use Pandas.

The DataFrame contains 178 observations.

It contains 13 feature columns.

The original class label is intentionally not added to this DataFrame.

The first five rows are displayed.

---

# 🔹 Cell 4 — Dataset Information

The fourth cell runs:

```python
df.info()
```

This displays the DataFrame structure.

The executed notebook reports 178 entries.

It reports 13 columns.

All 13 columns are `float64`.

Every column contains 178 non-null values.

This confirms that the dataset is completely populated.

---

# 🔹 Cell 5 — Statistical Summary

The fifth cell runs:

```python
df.describe()
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

The summary helps identify differences in feature scales.

For example, proline has much larger numerical values than several other features.

This reinforces the need for standardization.

---

# 🔹 Cell 6 — Missing-Value Check

The sixth cell runs:

```python
df.isnull().sum()
```

Every feature reports zero missing values.

Therefore, no imputation step is necessary.

The dataset can proceed directly to feature preparation and scaling.

---

# 🔹 Cell 7 — Dataset Dimensions

The seventh cell displays:

```python
df.shape[0]
df.shape[1]
```

The executed notebook reports:

```text
Number of Rows    : 178
Number of Features: 13
```

This confirms the expected dataset dimensions.

---

# 🔹 Cell 8 — Display Feature Names

The eighth cell loops through:

```python
wine.feature_names
```

and prints all 13 feature names.

This is useful for understanding exactly which measurements are being supplied to GMM.

The features include chemical properties such as alcohol, magnesium, phenols, flavanoids, color intensity, hue, and proline.

---

# 🔹 Cell 9 — Prepare Features for GMM

The ninth cell creates the feature matrix:

```python
X = df.copy()
```

This makes a copy of the DataFrame.

The resulting feature matrix contains all 13 Wine measurements.

The notebook prints:

```text
Feature Shape: (178, 13)
```

No target column is included.

This maintains the unsupervised nature of the clustering task.

---

# 🔹 Cell 10 — Standardize the Features

The tenth cell uses:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardization puts the features onto a comparable scale.

The output retains the shape:

```text
(178, 13)
```

The scaler is fitted to the feature matrix.

The transformed values are stored in `X_scaled`.

This standardized matrix is passed to GMM.

---

# 🔹 Cell 11 — Create the Initial GMM

The eleventh cell creates:

```python
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)
```

The initial model has three Gaussian components.

The random state is fixed at 42.

Fixing the random state improves reproducibility.

At this stage, the model has been created but not yet fitted.

---

# 🔹 Cell 12 — Fit the Initial GMM

The twelfth cell runs:

```python
gmm.fit(X_scaled)
```

This trains the GMM on the standardized features.

The EM procedure estimates the component parameters.

The notebook prints:

```text
GMM model fitted successfully!
```

The model is now ready to produce cluster labels and probabilities.

---

# 🔹 Cell 13 — Predict Cluster Labels

The thirteenth cell runs:

```python
cluster_labels = gmm.predict(X_scaled)
```

Each observation receives one predicted component label.

The labels are integer values corresponding to the Gaussian components.

These labels are used for cluster summaries and later analysis.

---

# 🔹 Cell 14 — Display First 20 Labels

The fourteenth cell prints:

```python
cluster_labels[:20]
```

The executed notebook reports the first twenty labels as:

```text
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
```

This demonstrates the cluster assignment output.

The numeric cluster identifiers do not inherently represent class names.

Cluster numbering is an internal model label.

---

# 🔹 Cell 15 — Count Observations in Each Cluster

The fifteenth cell counts cluster membership.

```python
cluster_counts = pd.Series(
    cluster_labels
).value_counts().sort_index()
```

The executed three-component model produces:

```text
Cluster 0 → 65
Cluster 1 → 51
Cluster 2 → 62
```

The distribution is relatively balanced for the initial three-component model.

This is only an intermediate model.

The notebook later evaluates different component counts.

---

# 🔹 Cell 16 — Calculate Cluster Probabilities

The sixteenth cell calculates:

```python
cluster_probabilities = gmm.predict_proba(X_scaled)
```

The output shape is:

```text
(178, 3)
```

There are 178 observations.

There are three probabilities per observation.

Each row contains the estimated probability of belonging to each Gaussian component.

---

# 🔹 Cell 17 — Display Cluster Probabilities

The seventeenth cell creates a DataFrame containing:

```text
Cluster_0_Probability
Cluster_1_Probability
Cluster_2_Probability
```

The first rows show very high probabilities for the predicted component.

This illustrates the soft-clustering output.

The probability values for every row sum to approximately one.

---

# 🔹 Cell 18 — Calculate AIC

The eighteenth cell calculates:

```python
aic_score = gmm.aic(X_scaled)
```

The executed three-component model has an AIC of approximately:

```text
4843.250257
```

AIC combines model fit with a complexity penalty.

Lower AIC values are preferred when comparing candidate models.

The notebook later calculates AIC for one through seven components.

---

# 🔹 Cell 19 — Calculate BIC

The nineteenth cell calculates:

```python
bic_score = gmm.bic(X_scaled)
```

The executed three-component model has a BIC of approximately:

```text
5842.330292
```

Lower BIC values are preferred.

The BIC is later compared across all candidate component counts.

---

# 🔹 Cell 20 — Display AIC and BIC

The twentieth cell creates:

```python
model_scores = pd.DataFrame({
    "Metric": ["AIC", "BIC"],
    "Score": [aic_score, bic_score]
})
```

The output presents the two model-selection metrics in a compact DataFrame.

For the initial three-component model:

```text
AIC → 4843.250257
BIC → 5842.330292
```

---

# 🔹 Cell 21 — Test Different Numbers of Components

The twenty-first cell tests:

```python
n_components_range = range(1, 8)
```

The notebook therefore evaluates seven candidate models.

For every component count:

1. A GMM is created.
2. The GMM is fitted.
3. AIC is calculated.
4. BIC is calculated.
5. Scores are stored.

This allows model complexity to be evaluated systematically.

---

# 🔹 Cell 22 — Create AIC/BIC Comparison DataFrame

The twenty-second cell creates:

```python
model_comparison = pd.DataFrame({
    "Number of Components": list(n_components_range),
    "AIC": aic_scores,
    "BIC": bic_scores
})
```

This produces a clear table for model comparison.

The table contains seven candidate component counts.

It allows the minimum AIC and minimum BIC to be identified.

---

# 🔹 Cell 23 — Find Best AIC Component Count

The twenty-third cell uses:

```python
np.argmin(aic_scores)
```

The executed notebook reports:

```text
Best Components according to AIC: 4
Lowest AIC: 4473.65296618721
```

Therefore, AIC favors a four-component model.

The notebook does not use this value for the final model.

---

# 🔹 Cell 24 — Find Best BIC Component Count

The twenty-fourth cell identifies the lowest BIC.

The executed notebook reports:

```text
Best Components according to BIC: 2
Lowest BIC: 5629.57338899323
```

Therefore, BIC favors two components.
---

## 🔹 Cell 25 — Visualizing AIC Scores

```python
plt.figure(figsize=(10, 6))

plt.plot(
    list(n_components_range),
    aic_scores,
    marker="o"
)

plt.title("AIC vs Number of GMM Components")
plt.xlabel("Number of Components")
plt.ylabel("AIC Score")

plt.xticks(list(n_components_range))
plt.grid()

plt.show()
```

### Explanation

Cell 25 visualizes the AIC values calculated for different numbers of Gaussian components.

The notebook previously tested:

```python
n_components_range = range(1, 8)
```

Therefore, the graph contains AIC values for:

```text
1
2
3
4
5
6
7
```

The x-axis represents the number of GMM components.

The y-axis represents the corresponding AIC score.

The purpose of this graph is to make model selection easier to understand visually.

A lower AIC value is preferred when comparing candidate models.

From the calculations performed earlier, the lowest AIC occurs at:

```text
4 components
```

Therefore:

```text
AIC → prefers 4 components
```

This visualization supports the numerical result obtained in Cell 23.

---

## 🔹 Cell 26 — Visualizing BIC Scores

```python
plt.figure(figsize=(10, 6))

plt.plot(
    list(n_components_range),
    bic_scores,
    marker="o"
)

plt.title("BIC vs Number of GMM Components")
plt.xlabel("Number of Components")
plt.ylabel("BIC Score")

plt.xticks(list(n_components_range))
plt.grid()

plt.show()
```

### Explanation

Cell 26 creates a graph of BIC against the number of GMM components.

The x-axis contains the candidate component counts from 1 through 7.

The y-axis contains the corresponding BIC scores.

BIC penalizes model complexity and therefore helps identify an appropriate number of Gaussian components.

The lowest BIC from the notebook is obtained with:

```text
2 components
```

Therefore:

```text
BIC → prefers 2 components
```

This is important because the notebook uses BIC as the criterion for selecting the final GMM.

The graph provides a visual confirmation of the model-selection result obtained in Cell 24.

---

## 🔹 Cell 27 — Select the Final Number of Components

```python
final_components = best_bic_components

print(
    "Selected Number of Components:",
    final_components
)
```

### Output

```text
Selected Number of Components: 2
```

### Explanation

Cell 27 uses the component count selected by BIC.

The variable:

```python
best_bic_components
```

was calculated earlier from the component comparison.

Because BIC identified two components as the best configuration:

```python
final_components = best_bic_components
```

results in:

```text
final_components = 2
```

The notebook therefore moves forward with a two-component GMM.

### Important Decision

The model-selection process produced:

```text
AIC → 4 components
BIC → 2 components
```

The notebook chooses:

```text
Final GMM → 2 components
```

because BIC is used as the final selection criterion.

---

## 🔹 Cell 28 — Create and Train the Final GMM

```python
final_gmm = GaussianMixture(
    n_components=final_components,
    random_state=42
)

final_gmm.fit(X_scaled)

print("Final GMM model trained successfully!")
```

### Output

```text
Final GMM model trained successfully!
```

### Explanation

Cell 28 creates a new Gaussian Mixture Model using the selected number of components.

Since:

```text
final_components = 2
```

the model contains two Gaussian distributions.

The random state is fixed:

```python
random_state=42
```

This improves reproducibility.

The model is then trained using:

```python
final_gmm.fit(X_scaled)
```

The complete standardized Wine feature matrix is supplied to the final model.

At this point, the final GMM has learned its Gaussian parameters from the data.

Conceptually:

```text
Standardized Wine Features
          ↓
      Final GMM
          ↓
   2 Gaussian Components
```

---

## 🔹 Cell 29 — Predict Final Cluster Assignments

```python
final_labels = final_gmm.predict(X_scaled)

print("Final cluster labels generated successfully!")

print("\nFirst 20 Cluster Labels:")
print(final_labels[:20])
```

### Output

```text
Final cluster labels generated successfully!

First 20 Cluster Labels:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

### Explanation

Cell 29 generates the final cluster assignment for every Wine observation.

The method:

```python
final_gmm.predict(X_scaled)
```

assigns each observation to the Gaussian component with the highest posterior probability.

The resulting array is stored in:

```python
final_labels
```

Each value is either:

```text
0
```

or:

```text
1
```

because the final model contains two components.

The first 20 observations are assigned to Cluster 0 in the executed notebook.

### Important Note

Cluster numbers are model-generated identifiers.

Therefore:

```text
Cluster 0 ≠ automatically Wine Class 0
Cluster 1 ≠ automatically Wine Class 1
```

The relationship with the original Wine classes is examined later in Cell 38.

---

## 🔹 Cell 30 — Display the Final Cluster Distribution

```python
final_cluster_counts = (
    pd.Series(final_labels)
    .value_counts()
    .sort_index()
)

print("Final Cluster Distribution:")

display(
    final_cluster_counts.to_frame(
        name="Number of Samples"
    )
)
```

### Output

```text
Final Cluster Distribution:

   Number of Samples
0                122
1                 56
```

### Explanation

Cell 30 counts how many observations belong to each final GMM cluster.

The final distribution is:

| GMM Cluster | Number of Samples |
|---:|---:|
| 0 | 122 |
| 1 | 56 |

The total is:

```text
122 + 56 = 178
```

which matches the total number of observations in the Wine dataset.

Therefore, every observation has received exactly one final cluster assignment.

The distribution also shows that the two clusters are not equally sized.

```text
Cluster 0 → 122 observations
Cluster 1 → 56 observations
```

---

## 🔹 Cell 31 — Calculate Final Cluster Probabilities

```python
final_probabilities = final_gmm.predict_proba(X_scaled)

print(
    "Probability Shape:",
    final_probabilities.shape
)

display(
    pd.DataFrame(
        final_probabilities,
        columns=[
            f"Cluster_{i}_Probability"
            for i in range(final_components)
        ]
    ).head(10)
)
```

### Output

```text
Probability Shape: (178, 2)
```

The first observations have probabilities similar to:

```text
Cluster 0 Probability   Cluster 1 Probability
1.0                     2.639315e-43
1.0                     4.787913e-36
1.0                     6.055181e-46
1.0                     2.013129e-67
1.0                     1.245628e-17
```

### Explanation

Cell 31 calculates the probability that every observation belongs to each final Gaussian component.

The method:

```python
final_gmm.predict_proba(X_scaled)
```

returns a probability matrix.

Its shape is:

```text
(178, 2)
```

because there are:

```text
178 observations
2 clusters
```

Each row contains the probability distribution for one observation.

For example:

```text
Cluster 0 → 0.90
Cluster 1 → 0.10
```

would indicate stronger membership in Cluster 0.

In the executed notebook, many observations have probabilities extremely close to either 0 or 1.

This indicates that the final two-component model makes very confident assignments for those observations.

---

## 🔹 Cell 32 — Find the Highest Cluster Probability

```python
max_probability = final_probabilities.max(axis=1)

print("Maximum cluster probabilities:")

display(
    pd.DataFrame({
        "Max_Cluster_Probability": max_probability
    }).head(10)
)
```

### Output

The first observations have:

```text
Max_Cluster_Probability
1.0
1.0
1.0
1.0
...
```

### Explanation

Cell 32 extracts the highest probability for each observation.

For every row of the probability matrix, the maximum value represents the confidence of the model's selected cluster.

For example:

```text
Cluster 0 → 0.92
Cluster 1 → 0.08
```

gives:

```text
Maximum probability = 0.92
```

The notebook stores these values in:

```python
max_probability
```

This array is later used to identify observations that have low clustering confidence.

Conceptually:

```text
Final Probabilities
        ↓
Maximum Probability
        ↓
Cluster Confidence
```

---

## 🔹 Cell 33 — Identify Low-Confidence Observations

```python
uncertainty_threshold = 0.60

uncertain_mask = (
    max_probability < uncertainty_threshold
)

uncertain_samples = np.where(
    uncertain_mask
)[0]

print(
    "Number of uncertain observations:",
    len(uncertain_samples)
)

print("\nUncertain observation indices:")
print(uncertain_samples)
```

### Output

```text
Number of uncertain observations: 0

Uncertain observation indices:
[]
```

### Explanation

Cell 33 checks whether any observation has a maximum cluster probability below:

```text
0.60
```

The threshold is defined as:

```python
uncertainty_threshold = 0.60
```

An observation is considered uncertain when:

```text
Maximum cluster probability < 0.60
```

The executed notebook reports:

```text
0 uncertain observations
```

Therefore, every observation has a maximum cluster probability of at least 0.60.

This means that according to the notebook's chosen threshold, no observations are considered low-confidence.

### Important Interpretation

This does not mean that the GMM is universally certain or perfect.

It only means:

```text
No observation falls below the selected 0.60 threshold.
```

The threshold itself is a project-level rule used to inspect uncertainty.

---

## 🔹 Cell 34 — Create the Final GMM DataFrame

```python
gmm_df = df.copy()

gmm_df["GMM_Cluster"] = final_labels
gmm_df["Cluster_Probability"] = max_probability

display(gmm_df.head(10))
```

### Explanation

Cell 34 creates a final DataFrame containing the original Wine features together with the GMM results.

The original DataFrame is copied:

```python
gmm_df = df.copy()
```

Two new columns are added.

### Column 1 — `GMM_Cluster`

Contains the final cluster assignment:

```text
0
1
```

### Column 2 — `Cluster_Probability`

Contains the highest probability associated with the assigned cluster.

The resulting structure is conceptually:

```text
Original Wine Features
        +
GMM_Cluster
        +
Cluster_Probability
```

This creates a useful table for inspecting the clustering results at the observation level.

The notebook displays the first 10 observations.

---

## 🔹 Cell 35 — Reduce Features Using PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("PCA Visualization Shape:", X_pca.shape)
```

### Output

```text
PCA Visualization Shape: (178, 2)
```

### Explanation

The Wine dataset contains 13 features.

Visualizing all 13 dimensions simultaneously is difficult.

Cell 35 therefore applies PCA to reduce the standardized feature matrix to two dimensions.

The model is:

```python
PCA(n_components=2)
```

The transformation is:

```python
X_pca = pca.fit_transform(X_scaled)
```

The resulting shape is:

```text
(178, 2)
```

This means:

```text
178 observations
2 principal components
```

### Important Distinction

PCA is used here for:

```text
Visualization
```

It is not used to train the GMM.

The GMM was trained using:

```text
All 13 standardized features
```

PCA is applied afterward to create a two-dimensional representation for plotting.

Conceptually:

```text
13-D Standardized Data
          ↓
        GMM
          ↓
    Final Clusters

13-D Standardized Data
          ↓
         PCA
          ↓
       2-D Plot
```

---

## 🔹 Cell 36 — Create the Visualization DataFrame

```python
visualization_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "GMM_Cluster": final_labels
})

display(visualization_df.head())
```

### Output

The first rows contain values such as:

```text
        PC1       PC2  GMM_Cluster
0  3.316751  1.443463            0
1  2.209465 -0.333393            0
2  2.516740  1.031151            0
3  3.757066  2.756372            0
4  1.008908  0.869831            0
```

### Explanation

Cell 36 creates a smaller DataFrame specifically for visualization.

It contains three columns:

```text
PC1
PC2
GMM_Cluster
```

`PC1` and `PC2` are the two PCA coordinates.

`GMM_Cluster` contains the final GMM assignment.

This structure makes it convenient to create a two-dimensional scatter plot.

---

## 🔹 Cell 37 — Visualize the GMM Clusters

```python
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=visualization_df,
    x="PC1",
    y="PC2",
    hue="GMM_Cluster",
    palette="deep",
    s=80
)

plt.title("GMM Clusters Visualized in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid()
plt.legend(title="GMM Cluster")

plt.show()
```

### Explanation

Cell 37 visualizes the final GMM clusters in PCA space.

The x-axis represents:

```text
Principal Component 1
```

The y-axis represents:

```text
Principal Component 2
```

The points are grouped visually according to:

```text
GMM_Cluster
```

The plot therefore provides a two-dimensional representation of the clustering result.

### What This Plot Shows

The visualization helps answer:

- Where are the discovered clusters located?
- Do the clusters appear separated?
- Are there overlapping regions?
- Are there observations near the apparent cluster boundaries?

The plot should be interpreted as a visualization of the original 13-dimensional feature space after PCA projection.

It does not mean that the GMM itself was trained on only two PCA dimensions.

---

## 🔹 Cell 38 — Compare GMM Clusters With Original Wine Classes

```python
comparison_df = pd.DataFrame({
    "Actual_Wine_Class": wine.target,
    "GMM_Cluster": final_labels
})

display(
    pd.crosstab(
        comparison_df["Actual_Wine_Class"],
        comparison_df["GMM_Cluster"]
    )
)
```

### Output

```text
GMM_Cluster         0   1
Actual_Wine_Class        
0                  59   0
1                  63   8
2                   0  48
```

### Explanation

Cell 38 compares the unsupervised GMM assignments with the original Wine dataset classes.

The original classes are:

```text
Class 0
Class 1
Class 2
```

The final GMM contains:

```text
Cluster 0
Cluster 1
```

The cross-tabulation is:

| Actual Wine Class | GMM Cluster 0 | GMM Cluster 1 |
|---:|---:|---:|
| 0 | 59 | 0 |
| 1 | 63 | 8 |
| 2 | 0 | 48 |

### Interpretation

The GMM does not reproduce the three original classes exactly because the selected GMM contains only two clusters.

However, the table shows a strong structural relationship:

```text
Wine Class 0
→ 59 observations in GMM Cluster 0
→ 0 observations in GMM Cluster 1
```

```text
Wine Class 2
→ 0 observations in GMM Cluster 0
→ 48 observations in GMM Cluster 1
```

Wine Class 1 is distributed across both GMM clusters:

```text
63 → Cluster 0
8  → Cluster 1
```

This indicates that the two-component GMM separates portions of the Wine feature space in a way that corresponds strongly to Classes 0 and 2, while Class 1 is split between the two discovered groups.

### Important Unsupervised-Learning Note

The original class labels were **not used to train the GMM**.

They are introduced here only for comparison after clustering.

Therefore, this table is an exploratory comparison rather than supervised training accuracy.

---

## 🔹 Cell 39 — Visualize Original Wine Classes

```python
visualization_df["Actual_Class"] = wine.target

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=visualization_df,
    x="PC1",
    y="PC2",
    hue="Actual_Class",
    palette="deep",
    s=80
)

plt.title("Original Wine Classes in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid()
plt.legend(title="Actual Class")

plt.show()
```

### Explanation

Cell 39 adds the original Wine class labels to the visualization DataFrame.

The new column is:

```text
Actual_Class
```

The notebook then creates another PCA scatter plot.

This time, the points are colored according to the original Wine classes instead of the GMM clusters.

Therefore, the notebook now has two related visualizations:

```text
Cell 37
→ GMM Clusters in PCA Space

Cell 39
→ Original Wine Classes in PCA Space
```

This makes it possible to visually compare:

```text
Unsupervised structure discovered by GMM
                vs
Original dataset class structure
```

Again, PCA is being used only to make the high-dimensional data visually interpretable.

---

## 🔹 Cell 40 — Calculate the Silhouette Score

```python
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(
    X_scaled,
    final_labels
)

print("GMM Silhouette Score:", silhouette)
```

### Output

```text
GMM Silhouette Score: 0.2682319543159216
```

### Explanation

Cell 40 calculates the Silhouette Score for the final GMM cluster assignments.

The calculation uses:

```python
silhouette_score(
    X_scaled,
    final_labels
)
```

The inputs are:

```text
X_scaled
→ Standardized feature matrix

final_labels
→ Final GMM cluster assignments
```

The resulting score is:

```text
0.2682319543159216
```

The Silhouette Score measures how well observations fit within their assigned clusters while considering separation from other clusters.

It generally ranges from:

```text
-1 to +1
```

Higher values indicate better-defined separation.

The notebook's value:

```text
0.2682
```

indicates that the two clusters have some separation, but the separation is not extremely strong.

This result should be interpreted together with:

```text
AIC
BIC
Cluster Distribution
Cluster Probabilities
PCA Visualization
Original-Class Comparison
```

rather than being considered in isolation.

---

# 📊 Final GMM Results

The completed notebook produces the following major results.

## Final Number of Components

The model-selection stage gives:

```text
Best AIC Components → 4
Best BIC Components → 2
```

The notebook uses BIC for final selection:

```text
Final Components = 2
```

---

## Final Cluster Distribution

The final model produces:

| Cluster | Samples |
|---:|---:|
| 0 | 122 |
| 1 | 56 |
| **Total** | **178** |

The total matches the number of Wine observations.

---

## Final Probability Matrix

The final probability matrix has shape:

```text
(178, 2)
```

This means each of the 178 observations receives two cluster probabilities.

Conceptually:

```text
Observation
     ↓
 ┌───────────────┐
 │ Cluster 0     │
 │ Cluster 1     │
 └───────────────┘
```

The probabilities for each observation sum approximately to one.

---

## Clustering Confidence

The notebook uses:

```text
Uncertainty Threshold = 0.60
```

The result is:

```text
Uncertain Observations = 0
```

Therefore, no observation falls below the selected confidence threshold.

---

## PCA Visualization

The standardized 13-dimensional feature space is projected into:

```text
PC1
PC2
```

The PCA matrix has shape:

```text
(178, 2)
```

PCA is used only for visualization and does not replace the 13-dimensional input used by GMM.

---

## Original-Class Comparison

The final comparison is:

| Actual Wine Class | Cluster 0 | Cluster 1 |
|---:|---:|---:|
| 0 | 59 | 0 |
| 1 | 63 | 8 |
| 2 | 0 | 48 |

This shows that the discovered two-cluster structure has a strong relationship with the original Wine classes, although it does not reproduce all three classes because the selected model contains only two components.

---

## Silhouette Score

The final Silhouette Score is:

```text
0.2682319543159216
```

This indicates moderate/limited separation between the final clusters rather than extremely strong separation.

---

# 🔍 Cluster Interpretation

The GMM discovers two probabilistic groups from the 13 standardized Wine features.

The cluster labels themselves do not have inherent semantic meaning.

For example:

```text
Cluster 0
```

does not automatically mean:

```text
Wine Class 0
```

Cluster numbering is assigned by the GMM implementation.

The comparison table is therefore necessary to understand how the discovered clusters relate to the original labels.

---

## Cluster 0

The final model places:

```text
122 observations
```

into Cluster 0.

The cross-tabulation shows:

```text
Wine Class 0 → 59
Wine Class 1 → 63
Wine Class 2 → 0
```

Therefore, Cluster 0 contains all observations from original Class 0 and a substantial portion of Class 1.

---

## Cluster 1

The final model places:

```text
56 observations
```

into Cluster 1.

The cross-tabulation shows:

```text
Wine Class 0 → 0
Wine Class 1 → 8
Wine Class 2 → 48
```

Therefore, Cluster 1 contains all observations from original Class 2 and a smaller portion of Class 1.

---

## Overall Interpretation

The discovered structure can be summarized as:

```text
Wine Class 0 ──────────────┐
                           ├── GMM Cluster 0
Part of Wine Class 1 ──────┘


Wine Class 2 ──────────────┐
                           ├── GMM Cluster 1
Part of Wine Class 1 ──────┘
```

The GMM therefore identifies two broad groups within the feature space.

The original Class 1 observations are distributed between the two discovered clusters.

---

# 🧠 Why the GMM Does Not Produce Three Clusters

The original Wine dataset contains three known classes.

However, the final GMM contains only two components because:

```text
BIC selected 2 components
```

This is an important distinction.

The GMM is not instructed to reproduce the original target classes.

Instead, it estimates the number of Gaussian components based on the model-selection procedure implemented in the notebook.

Therefore:

```text
Original Dataset
→ 3 known classes

Final GMM
→ 2 discovered components
```

This is a valid outcome for an unsupervised clustering experiment.

---

# 📈 AIC vs BIC — Final Comparison

The notebook demonstrates that AIC and BIC can select different model complexities.

```text
AIC
↓
4 components

BIC
↓
2 components
```

The notebook chooses BIC.

This can be represented as:

```text
Candidate GMMs
     │
     ├── 1 component
     ├── 2 components
     ├── 3 components
     ├── 4 components
     ├── 5 components
     ├── 6 components
     └── 7 components
             │
             ↓
       Compare AIC/BIC
             │
             ↓
       BIC minimum = 2
             │
             ↓
      Final GMM = 2
```

---

# 📊 Complete Model-Selection Summary

| Criterion | Selected Components | Interpretation |
|---|---:|---|
| AIC | 4 | Lower AIC achieved at 4 |
| BIC | 2 | Lower BIC achieved at 2 |
| Final Model | 2 | BIC is used for selection |

The notebook explicitly chooses the BIC result for the final model.

---

# 🔐 Probabilistic Nature of the Final GMM

One of the main advantages demonstrated by the notebook is that GMM does not only return labels.

It also returns probabilities.

For example, an observation could theoretically receive:

```text
Cluster 0 → 0.70
Cluster 1 → 0.30
```

The assigned cluster would be:

```text
Cluster 0
```

but the probability vector still communicates uncertainty.

In the executed notebook, many observations have probabilities extremely close to:

```text
1.0
0.0
```

This leads to high maximum probabilities for the observations.

---

# 📐 GMM and PCA in This Project

The notebook uses GMM and PCA for different purposes.

## GMM

Used for:

```text
Clustering
```

Input:

```text
178 × 13 standardized features
```

Output:

```text
Cluster assignments
Cluster probabilities
```

## PCA

Used for:

```text
Visualization
```

Input:

```text
178 × 13 standardized features
```

Output:

```text
178 × 2 representation
```

Therefore:

```text
GMM → Finds clusters

PCA → Helps visualize clusters
```

---

# 📊 Complete Notebook Structure

The complete 40-cell implementation can now be summarized as:

```text
Cell 1  → Import Libraries
Cell 2  → Load Wine Dataset
Cell 3  → Create Pandas DataFrame
Cell 4  → Dataset Information
Cell 5  → Statistical Summary
Cell 6  → Check Missing Values
Cell 7  → Dataset Dimensions
Cell 8  → Display Feature Names
Cell 9  → Prepare Features
Cell 10 → Standardize Features

Cell 11 → Create Initial GMM
Cell 12 → Fit Initial GMM
Cell 13 → Predict Initial Cluster Labels
Cell 14 → Display First 20 Labels
Cell 15 → Count Initial Cluster Distribution
Cell 16 → Calculate Cluster Probabilities
Cell 17 → Display Cluster Probabilities
Cell 18 → Calculate AIC
Cell 19 → Calculate BIC
Cell 20 → Display AIC and BIC

Cell 21 → Test 1–7 Components
Cell 22 → Create AIC/BIC Comparison DataFrame
Cell 23 → Find Best AIC Components
Cell 24 → Find Best BIC Components
Cell 25 → Visualize AIC
Cell 26 → Visualize BIC
Cell 27 → Select Final Components
Cell 28 → Train Final GMM
Cell 29 → Predict Final Labels
Cell 30 → Final Cluster Distribution

Cell 31 → Final Cluster Probabilities
Cell 32 → Maximum Cluster Probability
Cell 33 → Low-Confidence Observations
Cell 34 → Final GMM DataFrame
Cell 35 → PCA Reduction
Cell 36 → Visualization DataFrame
Cell 37 → GMM Cluster Visualization
Cell 38 → GMM vs Original Classes
Cell 39 → Original Class Visualization
Cell 40 → Silhouette Score
```

---

# 🔄 Complete GMM Project Pipeline

The entire project can be represented as:

```text
Wine Dataset
     ↓
Create DataFrame
     ↓
Explore Dataset
     ↓
Check Missing Values
     ↓
Select 13 Features
     ↓
Standardize Features
     ↓
Initial GMM
     ↓
Initial Cluster Assignments
     ↓
Cluster Probabilities
     ↓
AIC
     ↓
BIC
     ↓
Test 1–7 Components
     ↓
AIC/BIC Comparison
     ↓
Select Lowest BIC
     ↓
Final GMM with 2 Components
     ↓
Final Cluster Labels
     ↓
Cluster Distribution
     ↓
Cluster Probabilities
     ↓
Confidence Analysis
     ↓
PCA
     ↓
GMM Visualization
     ↓
Original-Class Comparison
     ↓
Silhouette Score
```

---

# 🔥 Advantages of GMM

The implementation demonstrates several important advantages of Gaussian Mixture Models.

## 1. Soft Clustering

GMM provides probabilities for cluster membership.

```text
Observation
   ↓
Cluster 0 → Probability
Cluster 1 → Probability
```

This gives more information than a simple hard label.

---

## 2. Probabilistic Interpretation

The model describes the data using Gaussian probability distributions.

This makes the clustering result interpretable in terms of estimated component membership.

---

## 3. Flexible Cluster Shapes

With appropriate covariance structures, GMM can represent clusters that are not restricted to simple spherical shapes.

---

## 4. Model Selection

AIC and BIC can be used to compare models with different numbers of components.

This project demonstrates this process directly.

---

## 5. Uncertainty Analysis

Because GMM produces probabilities, observations with lower maximum probabilities can be inspected as potentially ambiguous points.

---

## 6. Useful for Exploratory Analysis

GMM can discover hidden structure when predefined labels are not used during training.

---

# ❌ Limitations of GMM

Despite its flexibility, GMM also has limitations.

## 1. Number of Components Must Be Considered

The model requires a component count or a model-selection procedure.

In this project, AIC and BIC are used to compare several choices.

---

## 2. Gaussian Assumption

GMM assumes that the data can be represented using Gaussian components.

If the actual distribution is very different, the model may not represent it well.

---

## 3. Sensitivity to Initialization

The EM optimization process can depend on initialization.

The notebook uses:

```python
random_state=42
```

to improve reproducibility.

---

## 4. Computational Cost

GMM can be more computationally expensive than simple centroid-based methods such as K-Means.

---

## 5. Interpretation of Cluster Numbers

Cluster numbers are arbitrary identifiers.

A cluster labeled:

```text
0
```

does not inherently represent a real-world category numbered 0.

---

## 6. Model-Selection Criteria Can Disagree

This project demonstrates:

```text
AIC → 4
BIC → 2
```

Different criteria can therefore suggest different levels of model complexity.

---

# 🌍 Real-World Applications of GMM

Gaussian Mixture Models can be applied to many unsupervised learning problems.

Common applications include:

- Customer segmentation
- Market segmentation
- Anomaly exploration
- Image segmentation
- Speech processing
- Pattern recognition
- Document clustering
- Financial data analysis
- Biological data analysis
- Density estimation
- Sensor data analysis
- Fraud exploration
- Medical data analysis
- Feature distribution modeling

The key idea is to identify hidden groups or probability distributions inside unlabeled data.

---

# 🆚 GMM vs K-Means

| Feature | K-Means | GMM |
|---|---|---|
| Learning Type | Unsupervised | Unsupervised |
| Main Idea | Centroid-based clustering | Probability distribution-based clustering |
| Membership | Hard | Soft |
| Output | Cluster label | Cluster label + probabilities |
| Cluster Shape | Typically spherical | Can represent elliptical structures |
| Probability | No | Yes |
| Model Selection | Requires choosing K | AIC/BIC can help |
| Complexity | Generally simpler | Generally more complex |
| Uncertainty | Limited | Explicitly represented |

The main distinction demonstrated by this project is:

```text
K-Means
→ "Which cluster?"

GMM
→ "How probable is membership in each cluster?"
```

---

# 🆚 GMM vs Hierarchical Clustering

| Feature | Hierarchical Clustering | GMM |
|---|---|---|
| Main Concept | Nested groups | Gaussian mixture |
| Output | Hierarchy / clusters | Probabilities + clusters |
| Soft Membership | Not the primary output | Yes |
| Distribution Model | No explicit Gaussian model | Yes |
| Visualization | Dendrogram | Probability/cluster plots |
| Model Selection | Distance/linkage choices | AIC/BIC and other criteria |

---

# 🧪 Interpretation of the Silhouette Score

The final score is:

```text
0.2682319543159216
```

The Silhouette Score considers both:

```text
Within-cluster similarity
```

and:

```text
Between-cluster separation
```

A score closer to 1 generally indicates stronger separation.

A score around 0 indicates overlapping or weakly separated groups.

Negative values can indicate that observations may be assigned to inappropriate clusters.

The notebook's score of approximately:

```text
0.268
```

suggests that the final two clusters have some structure but are not extremely well separated.

This should not be interpreted as a failure of the entire GMM experiment.

The score is one evaluation measure and should be considered alongside the AIC/BIC analysis, probability outputs, and visualizations.

---

# 🧠 Important Learning Point — Unsupervised vs Original Labels

A particularly important concept demonstrated by this project is the difference between:

```text
Training
```

and:

```text
Evaluation / Comparison
```

The GMM is trained using:

```text
X_scaled
```

only.

The original target labels:

```python
wine.target
```

are not supplied to:

```python
final_gmm.fit(X_scaled)
```

The original labels are used later in Cell 38 for comparison.

Therefore:

```text
GMM Training
→ Unsupervised

Original Class Comparison
→ Post-clustering analysis
```

This preserves the unsupervised nature of the experiment.

---

# 📦 Installation

Create or open the project directory.

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the Gaussian Mixture Model notebook and execute the cells sequentially.

---

# 📋 Requirements

```text
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

The Wine dataset is loaded directly from Scikit-learn, so a separate CSV dataset is not required.

---

# 🛠 Technologies Used

```text
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

Main Scikit-learn components used include:

```python
load_wine
StandardScaler
GaussianMixture
PCA
silhouette_score
```

---

# 📁 Project Structure

```text
Gaussian-Mixture-Model/
│
├── Gaussian Mixture Model.ipynb
├── README.md
├── README_Part_2.md
├── requirements.txt
└── .gitignore
```

The notebook loads the Wine dataset directly from Scikit-learn.

---

# 🎓 Learning Outcomes

After completing this project, you will be able to:

- Understand Gaussian Mixture Models.
- Understand unsupervised learning.
- Understand clustering.
- Understand Gaussian probability distributions.
- Understand mixture distributions.
- Understand soft clustering.
- Understand responsibilities.
- Understand the Expectation-Maximization approach.
- Understand Gaussian means and covariance.
- Understand mixing weights.
- Understand likelihood.
- Calculate AIC.
- Calculate BIC.
- Compare different GMM component counts.
- Select a model using BIC.
- Train a final GMM.
- Generate cluster assignments.
- Generate cluster probabilities.
- Analyze clustering confidence.
- Identify low-confidence observations.
- Use PCA for visualization.
- Visualize GMM clusters.
- Compare discovered clusters with original labels.
- Calculate the Silhouette Score.
- Interpret clustering results.
- Understand the difference between GMM and K-Means.
- Understand advantages and limitations of probabilistic clustering.

---

# 📝 Final Conclusion

This project demonstrates a complete Gaussian Mixture Model clustering workflow using the Wine dataset from Scikit-learn.

The notebook begins by exploring the 178 observations and 13 numerical features and then standardizes the feature matrix before training the GMM.

The initial model uses three Gaussian components.

The notebook then evaluates different component counts from:

```text
1 through 7
```

using:

```text
AIC
BIC
```

The model-selection results show:

```text
AIC → 4 components
BIC → 2 components
```

The notebook selects the BIC result and trains the final GMM using:

```text
2 components
```

The final model produces:

```text
Cluster 0 → 122 observations
Cluster 1 → 56 observations
```

It also produces probability estimates for all 178 observations.

The confidence analysis identifies:

```text
0 uncertain observations
```

using the selected threshold of:

```text
0.60
```

PCA is then used to project the 13-dimensional standardized feature space into two dimensions for visualization.

The final clusters are compared with the original Wine classes. The comparison shows that all original Class 0 observations are placed in GMM Cluster 0, all original Class 2 observations are placed in GMM Cluster 1, while original Class 1 is distributed between the two clusters.

Finally, the model receives a Silhouette Score of:

```text
0.2682319543159216
```

This indicates some cluster separation, but not extremely strong separation.

The complete learning pipeline is therefore:

```text
Wine Dataset
      ↓
Data Exploration
      ↓
Feature Preparation
      ↓
Standardization
      ↓
Initial GMM
      ↓
Cluster Probabilities
      ↓
AIC / BIC
      ↓
Model Selection
      ↓
Final GMM
      ↓
Cluster Assignments
      ↓
Probability Analysis
      ↓
Confidence Analysis
      ↓
PCA Visualization
      ↓
Original-Class Comparison
      ↓
Silhouette Evaluation
```

The project demonstrates how GMM can discover probabilistic structure in unlabeled data and how model-selection criteria, probability analysis, visualization, and clustering metrics can be combined to evaluate the resulting groups.

---

# ⭐ Final Takeaway

```text
Unlabeled Wine Features
          ↓
   Standardization
          ↓
      Gaussian
   Mixture Models
          ↓
     AIC / BIC
          ↓
    BIC Selection
          ↓
   2 Final Clusters
          ↓
 Cluster Probabilities
          ↓
 Confidence Analysis
          ↓
       PCA
          ↓
 Visualization
          ↓
 Class Comparison
          ↓
 Silhouette Score
```

The most important concept from the project is:

```text
GMM does not simply assign an observation to a cluster.

It models the probability that the observation
belongs to each Gaussian component.
```

This probabilistic approach makes GMM a useful technique for exploratory clustering, density modeling, and discovering hidden structure in complex datasets.



# 🔗 README Continuation

This file is intentionally written as a continuation of the main README.

```text
README.md
   │
   ├── Project Overview
   ├── GMM Concepts
   ├── Dataset
   ├── Technologies
   ├── Complete Workflow
   └── Cell 1 → Cell 24
   │
   ├── Cell 25 → Cell 40
   ├── Final Results
   ├── Cluster Interpretation
   ├── Model Selection Summary
   ├── GMM/PCA Analysis
   ├── Advantages
   ├── Limitations
   ├── Applications
   ├── Installation
   ├── Requirements
   ├── Learning Outcomes
   ├── Conclusion
   └── Final Takeaway
```

---

# 👨‍💻 Author

**Vansh Bhardwaj**

Machine Learning Enthusiast | Data Science Learner | Python Developer

This project is part of a practical Machine Learning learning journey focused on implementing and understanding Machine Learning algorithms using Python and Scikit-learn.

---
The notebook uses this result to select the final GMM.
