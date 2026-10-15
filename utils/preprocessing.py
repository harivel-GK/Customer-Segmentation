import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    processed = df.copy()
    if 'Genre' in processed.columns:
        processed.rename(columns={'Genre': 'Gender'}, inplace=True)
    if 'CustomerID' in processed.columns:
        processed.drop(columns=['CustomerID'], inplace=True)
    if 'Gender' in processed.columns:
        processed['Gender'] = processed['Gender'].map({'Male': 0, 'Female': 1})
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = processed[features].apply(pd.to_numeric, errors='coerce').dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return processed, X, X_scaled, scaler
