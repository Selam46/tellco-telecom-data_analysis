import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_user_engagement_metrics(connection):
    """
    Get user engagement metrics from the database
    """
    query = """
    SELECT 
        "MSISDN/Number" as msisdn,
        COUNT(*) as session_frequency,
        SUM("Dur. (ms)") as total_duration,
        SUM("Total DL (Bytes)" + "Total UL (Bytes)") as total_traffic,
        SUM("Social Media DL (Bytes)" + "Social Media UL (Bytes)") as social_media_usage,
        SUM("Google DL (Bytes)" + "Google UL (Bytes)") as google_usage,
        SUM("Email DL (Bytes)" + "Email UL (Bytes)") as email_usage,
        SUM("Youtube DL (Bytes)" + "Youtube UL (Bytes)") as youtube_usage,
        SUM("Netflix DL (Bytes)" + "Netflix UL (Bytes)") as netflix_usage,
        SUM("Gaming DL (Bytes)" + "Gaming UL (Bytes)") as gaming_usage,
        SUM("Other DL (Bytes)" + "Other UL (Bytes)") as other_usage
    FROM xdr_data
    GROUP BY "MSISDN/Number"
    """
    return pd.read_sql_query(query, connection)

def get_top_users_per_metric(df, metrics, n=10):
    """
    Get top n users for each engagement metric
    """
    top_users = {}
    for metric in metrics:
        top_users[metric] = df.nlargest(n, metric)[['msisdn', metric]]
    return top_users

def perform_kmeans_clustering(df, features, n_clusters=3):
    """
    Perform k-means clustering on normalized features
    """
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    # Calculate cluster statistics
    cluster_stats = df_clustered.groupby('cluster')[features].agg(['min', 'max', 'mean', 'sum'])
    
    return df_clustered, cluster_stats, kmeans, X_scaled

def find_optimal_k(X, k_range):
    """
    Find optimal k using elbow method and silhouette score
    Optimized for large datasets
    """
    # If dataset is too large, take a random sample
    max_samples = 10000
    if X.shape[0] > max_samples:
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
    
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        # Use more efficient kmeans parameters
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=5,  # Reduced from default 10
            max_iter=100,
            algorithm='elkan',  # More efficient for low-dimensional data
            tol=1e-3  # Slightly relaxed tolerance
        )
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            # Calculate silhouette score on a smaller sample if dataset is large
            if X.shape[0] > 10000:
                sample_size = 10000
                sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
                sil_score = silhouette_score(X[sample_indices], labels[sample_indices])
            else:
                sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
    
    return inertias, silhouette_scores

def get_top_apps_by_usage(df):
    """
    Get top applications by total usage
    """
    app_columns = ['social_media_usage', 'google_usage', 'email_usage', 
                  'youtube_usage', 'netflix_usage', 'gaming_usage', 'other_usage']
    
    app_usage = df[app_columns].sum().sort_values(ascending=False)
    return app_usage

def get_top_users_per_app(df, n=10):
    """
    Get top n users for each application
    """
    app_columns = ['social_media_usage', 'google_usage', 'email_usage', 
                  'youtube_usage', 'netflix_usage', 'gaming_usage', 'other_usage']
    
    top_users = {}
    for app in app_columns:
        top_users[app] = df.nlargest(n, app)[['msisdn', app]]
    return top_users 