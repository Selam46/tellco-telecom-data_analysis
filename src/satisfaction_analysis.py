import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from engagement_analysis import get_user_engagement_metrics, perform_kmeans_clustering
from sqlalchemy import create_engine

def calculate_engagement_score(df, kmeans_model, X_scaled):
    """
    Calculate engagement score based on Euclidean distance from least engaged cluster
    """
    # Get cluster centers
    cluster_centers = kmeans_model.cluster_centers_
    
    # Find the least engaged cluster (assuming cluster 0 is least engaged)
    least_engaged_center = cluster_centers[0]
    
    # Calculate Euclidean distance from least engaged cluster
    engagement_scores = np.linalg.norm(X_scaled - least_engaged_center, axis=1)
    
    # Normalize scores to 0-1 range
    engagement_scores = (engagement_scores - engagement_scores.min()) / (engagement_scores.max() - engagement_scores.min())
    
    return engagement_scores

def calculate_experience_score(df, kmeans_model, X_scaled):
    """
    Calculate experience score based on Euclidean distance from worst experience cluster
    """
    # Get cluster centers
    cluster_centers = kmeans_model.cluster_centers_
    
    # Find the worst experience cluster (assuming cluster 0 is worst)
    worst_experience_center = cluster_centers[0]
    
    # Calculate Euclidean distance from worst experience cluster
    experience_scores = np.linalg.norm(X_scaled - worst_experience_center, axis=1)
    
    # Normalize scores to 0-1 range
    experience_scores = (experience_scores - experience_scores.min()) / (experience_scores.max() - experience_scores.min())
    
    return experience_scores

def calculate_satisfaction_score(engagement_scores, experience_scores):
    """
    Calculate satisfaction score as average of engagement and experience scores
    """
    return (engagement_scores + experience_scores) / 2

def get_top_satisfied_customers(df, satisfaction_scores, n=10):
    """
    Get top n satisfied customers
    """
    df['satisfaction_score'] = satisfaction_scores
    return df.nlargest(n, 'satisfaction_score')[['msisdn', 'satisfaction_score']]

def build_satisfaction_regression_model(df, features, target):
    """
    Build a regression model to predict satisfaction score
    """
    X = df[features]
    y = df[target]
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

def cluster_satisfaction_experience(df, engagement_scores, experience_scores, n_clusters=2):
    """
    Perform k-means clustering on satisfaction and experience scores
    """
    # Combine scores
    X = np.column_stack((engagement_scores, experience_scores))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Add cluster labels and scores to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    df_clustered['engagement_score'] = engagement_scores
    df_clustered['experience_score'] = experience_scores
    df_clustered['satisfaction_score'] = calculate_satisfaction_score(engagement_scores, experience_scores)
    
    # Calculate cluster statistics
    cluster_stats = df_clustered.groupby('cluster').agg({
        'satisfaction_score': ['mean', 'std', 'count'],
        'experience_score': ['mean', 'std']
    })
    
    return df_clustered, cluster_stats

def export_to_postgres(df, connection):
    """
    Export results to PostgreSQL database
    """
    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS user_satisfaction (
        msisdn VARCHAR(20) PRIMARY KEY,
        engagement_score FLOAT,
        experience_score FLOAT,
        satisfaction_score FLOAT,
        cluster INTEGER
    )
    """
    
    # Use cursor to execute the create table query
    with connection.cursor() as cursor:
        cursor.execute(create_table_query)
        connection.commit()
    
    # Create SQLAlchemy engine from the connection for pandas to_sql
    engine = create_engine('postgresql://', creator=lambda: connection)
    
    # Export data using pandas to_sql
    df[['msisdn', 'engagement_score', 'experience_score', 'satisfaction_score', 'cluster']].to_sql(
        'user_satisfaction', 
        engine, 
        if_exists='replace', 
        index=False
    )