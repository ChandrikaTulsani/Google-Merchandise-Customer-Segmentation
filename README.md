# Customer Segmentation using K-Means Clustering

An unsupervised machine learning project that segments e-commerce users into distinct behavioral groups based on their session activity, interaction patterns, and engagement frequency.

---

## Overview

Understanding customer behavior is essential for targeted marketing and personalized user experiences. This project applies **K-Means clustering** to e-commerce event data to group users into meaningful segments — enabling data-driven strategies for retention, upselling, and re-engagement.

---

## Dataset

**`events1.csv`** — an e-commerce event log containing user interactions over a period of time.

| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `ga_session_id` | Google Analytics session ID |
| `country` | User's country |
| `device` | Device type (mobile, desktop, tablet) |
| `type` | Event type (e.g., `purchase`, `add_to_cart`) |
| `item_id` | Item involved in the event |
| `date` | Timestamp of the event |

> **Note:** 4,555 missing values were found in the `country` column and filled with `0`.

---

## Project Structure

```
├── CustomerSegmentation.ipynb   # Main notebook
├── events1.csv                  # E-commerce event dataset
└── README.md
```

---

## Methodology

### 1. Data Loading & Cleaning
The dataset is loaded from CSV. Null values are checked across all columns and filled with `0` where missing.

### 2. Feature Engineering
Raw event-level data is aggregated per user into three behavioral features:

| Feature | Description |
|---|---|
| `session_count` | Number of unique sessions |
| `interaction_count` | Total number of item interactions |
| `active_days` | Number of unique active days |

The most frequent `device` and `country` per user are extracted and one-hot encoded, though only the core three behavioral features are used for clustering.

### 3. Feature Scaling
Features are standardized using `StandardScaler` to ensure equal contribution across dimensions before clustering.

### 4. Optimal Cluster Selection (Elbow Method)
WCSS (Within-Cluster Sum of Squares) is computed for k = 1 to 9. The elbow in the curve indicates **k = 4** as the optimal number of clusters.

### 5. K-Means Clustering
```python
KMeans(n_clusters=4, random_state=42)
```
Each user is assigned to one of four clusters based on their behavioral profile.

### 6. Cluster Labeling
Clusters are interpreted and labeled based on their mean feature values:

| Cluster | Label | Sessions | Interactions | Active Days |
|---|---|---|---|---|
| 0 | High Value | 2.37 | 70.61 | 8.22 |
| 1 | Active Users | 1.00 | 24.51 | 2.80 |
| 2 | Occasional Users | 2.99 | 483.67 | 43.06 |
| 3 | Low Engagement | 1.34 | 202.01 | 18.48 |

---

## Visualizations

- **Scatter Plot** — Sessions vs Interactions, colored by cluster
- **Bar Chart** — Customer segment counts
- **Heatmap** — Normalized cluster feature comparison
- **Pie Chart** — Segment distribution across the user base

---

## Business Insights & Marketing Strategies

| Segment | Behavior | Recommended Strategy |
|---|---|---|
| **High Value** | Frequent visits, moderate interactions | Offer loyalty rewards and premium services |
| **Active Users** | Low session count, fewer interactions | Upsell products and personalized recommendations |
| **Occasional Users** | High interactions, most active days | Nurture with exclusive offers and content |
| **Low Engagement** | Infrequent, low interaction | Run awareness ads and re-engagement campaigns |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

1. Place `events1.csv` in the `/content/` directory (or update the file path in the notebook).
2. Open `CustomerSegmentation.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells sequentially.

---

## Technologies Used

- **Python 3**
- **scikit-learn** — StandardScaler, KMeans, MinMaxScaler
- **pandas / numpy** — data aggregation and feature engineering
- **matplotlib / seaborn** — cluster and business insight visualizations
