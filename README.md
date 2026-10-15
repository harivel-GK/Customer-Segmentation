# Customer Segmentation using K-Means Clustering

An interactive Streamlit dashboard for customer segmentation using K-Means clustering, exploratory data analysis, PCA, and business recommendations.

## Features
- Customer dataset overview and data-quality checks
- Income and spending visualizations, correlation heatmap, and gender comparisons
- K-Means clustering with Elbow Method and Silhouette Score
- PCA and cluster scatter visualizations
- Cluster summaries and marketing recommendations
- Upload your own CSV dataset

## Requirements
Python 3.9+ recommended.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run app.py
```

The default dataset is `data/Mall_Customers.csv`. You can also upload a compatible CSV from the sidebar.

## Train model
`train_model.py` imports `utils.preprocessing`; ensure that module exists in your project before running the training script. The dashboard itself trains K-Means dynamically when run.
