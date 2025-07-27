# Student Spending Clustering & Segmentation

This project applies unsupervised machine learning techniques to cluster university students based on their spending behavior. The primary objective is to identify meaningful customer segments using K-Means and Hierarchical Clustering. This analysis helps understand different student profiles based on how they allocate spending across categories such as food, entertainment, books, and miscellaneous items.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Process Workflow](#process-workflow)
- [Clustering Methods](#clustering-methods)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

Customer segmentation is widely used in marketing to group customers with similar characteristics. In this project, we simulate this for students to categorize them based on how they manage their spending. This can provide insights for targeted financial services, budgeting tools, or campus services.

---

## Dataset

A synthetic dataset was used representing student spending across various categories. The dataset contains the following columns:

- `food`: Monthly food expenses
- `entertainment`: Monthly entertainment expenses
- `books`: Monthly academic expenses
- `misc`: Miscellaneous monthly expenses

> Note: You may use a publicly available or custom survey dataset such as the ["Student Spending Habits Dataset" from Kaggle](https://www.kaggle.com/code/shroukelnagdy/student-spending-habits/input).

---

## Objectives

- Perform data exploration and preprocessing
- Apply standardization to normalize features
- Perform clustering using K-Means and Hierarchical methods
- Visualize clusters using PCA and box plots
- Evaluate model performance using Silhouette Score

---

## Technologies Used

- Python 3.x
- Pandas, NumPy
- scikit-learn
- Seaborn, Matplotlib
- SciPy

---

## Process Workflow

1. **Data Collection**: Synthetic survey dataset representing student spending.
2. **Data Preprocessing**:
   - Handle missing values
   - Normalize data using StandardScaler
3. **Model Selection**:
   - K-Means Clustering
   - Agglomerative (Hierarchical) Clustering
4. **Training**: Fit models to standardized data
5. **Evaluation**:
   - Silhouette Score
   - Visual inspection via PCA and box plots
6. **Interpretation**: Cluster profiling and behavior insights

---

## Clustering Methods

### K-Means Clustering

- Used to partition students into `k` distinct groups based on spending similarity.
- Centroids represent the average position of all points in a cluster.
- Clusters evaluated using silhouette score and visualized via PCA and box plots.

### Hierarchical Clustering

- Agglomerative clustering with Ward linkage was used.
- Dendrogram visualization allows identifying natural cluster boundaries.
- Euclidean distance metric was applied.

---

## Evaluation

- **Silhouette Score**: Measures how similar each point is to its own cluster compared to other clusters.
- **Box Plots**: Visualize category-wise spending patterns across clusters.
- **Dendrogram**: Evaluate hierarchical structure and determine optimal cluster cuts.
- **PCA Projection**: Reduce dimensionality for scatter plot visualization.

---

## Results

- Clear segmentation was observed across spending patterns:
  - High spenders on entertainment vs. academic-focused spenders
  - Balanced spenders vs. those with heavy miscellaneous expenses
- Optimal clusters found using both visual (dendrogram) and quantitative (silhouette score) techniques

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-spending-clustering.git
   cd student-spending-clustering
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python clustering_analysis.py
   ```

4. Modify the script if using a custom dataset.

---

## Future Work

* Integrate demographic variables (age, major, gender, etc.)
* Automate optimal cluster selection using the Elbow Method
* Create a web dashboard for interactive cluster exploration
* Incorporate other clustering algorithms (DBSCAN, Gaussian Mixture Models)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

```

