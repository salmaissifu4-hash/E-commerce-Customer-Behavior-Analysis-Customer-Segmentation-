# E-commerce Customer Behavior Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

Segment e-commerce customers by their multi-channel behavior — app usage, online orders, and in-store visits — to identify distinct groups for targeting, retention, and personalization.

---

## Overview

This project analyzes 360 multi-channel customers using unsupervised clustering. The goal is to surface actionable customer segments that marketing, product, and operations teams can act on directly.

**Three clustering methods are compared:**

| Method | Approach | Silhouette Score |
|---|---|---|
| K-Means | Centroid-based (5 clusters) | ~0.10 |
| Hierarchical | Agglomerative, Ward linkage | ~0.07 |
| DBSCAN | Density-based, noise-aware | ~0.33 |

**Five identified customer segments:**

- **Premium Loyal Customers** — High spend, long tenure, low churn risk
- **High-Spend Omnichannel Buyers** — Active across app, web, and store
- **Digital-first Professionals** — Heavy app users, low store visits
- **Discount-driven Families** — Promo-responsive, higher return rates
- **Value Seekers** — Mid-tier spend, price-sensitive

---

## Tech Stack

- **Data:** Python, NumPy, pandas
- **ML:** scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, PCA)
- **Visualization:** Matplotlib, Seaborn
- **Notebook:** Jupyter
- **Persistence:** joblib

---

## Project Structure

```
retail-customer-intelligence/
├── data/
│   └── customer_segmentation_enhanced.csv   # Source dataset
├── notebooks/
│   └── customer_behavior_analysis.ipynb     # Full analysis
├── reports/
│   ├── cluster_profile.csv                  # Segment summary stats
│   ├── pca_cluster_view.png                 # 2D cluster visualization
│   ├── project_summary.md                   # Key results
│   └── artifacts/
│       └── segmentation_model_bundle.joblib # Trained model + preprocessor
├── src/
│   ├── segmentation_utils.py                # Data generation, preprocessing, prediction
│   └── generate_customer_data.py            # Standalone dataset generator
├── requirements.txt
└── README.md
```

---

## Getting Started

**1. Clone the repo**
```bash
git clone <your-repo-url>
cd retail-customer-intelligence
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Regenerate the dataset**
```bash
python src/generate_customer_data.py
```

**4. Run the notebook**

Open `notebooks/customer_behavior_analysis.ipynb` and run all cells. The notebook auto-generates the dataset on first run if the CSV is missing.

---

## Features Engineered

| Feature | Description |
|---|---|
| `digital_affinity` | Ratio of digital interactions to all channel activity |
| `store_vs_digital_ratio` | Store visits relative to digital touchpoints |
| `value_per_visit` | Average order value normalized by visit frequency |
| `engagement_band` | Low / Mid / High label based on overall activity |

---

## License

MIT
