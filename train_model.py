import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.preprocessing import load_data, preprocess_data

def train_and_save():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'Mall_Customers.csv')
    model_path = os.path.join(base_dir, 'models', 'kmeans_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    print("Loading data...")
    df = load_data(data_path)
    print("Preprocessing data...")
    df_processed, X, X_scaled, scaler = preprocess_data(df)
    optimal_k = 5
    print(f"Training K-Means with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {score:.2f}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    print("Training complete! Model and Scaler saved.")

if __name__ == "__main__":
    train_and_save()
