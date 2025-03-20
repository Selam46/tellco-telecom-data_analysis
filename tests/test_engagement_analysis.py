import unittest
import pandas as pd
import numpy as np
from src.engagement_analysis import (
    get_user_engagement_metrics,
    get_top_users_per_metric,
    perform_kmeans_clustering,
    find_optimal_k,
    get_top_apps_by_usage,
    get_top_users_per_app
)

class TestEngagementAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame to use for testing
        self.data = pd.DataFrame({
            'msisdn': ['123', '456', '789', '101', '112'],
            'session_frequency': [10, 20, 30, 40, 50],
            'total_duration': [1000, 2000, 3000, 4000, 5000],
            'total_traffic': [500, 1000, 1500, 2000, 2500],
            'social_media_usage': [100, 200, 300, 400, 500],
            'google_usage': [50, 100, 150, 200, 250],
            'email_usage': [10, 20, 30, 40, 50],
            'youtube_usage': [5, 10, 15, 20, 25],
            'netflix_usage': [2, 4, 6, 8, 10],
            'gaming_usage': [1, 2, 3, 4, 5],
            'other_usage': [0, 0, 0, 0, 0]
        })

    def test_get_top_users_per_metric(self):
        """Test getting top users per engagement metric."""
        metrics = ['session_frequency', 'total_duration']
        top_users = get_top_users_per_metric(self.data, metrics, n=3)
        
        self.assertEqual(len(top_users['session_frequency']), 3)
        self.assertEqual(top_users['session_frequency'].iloc[0]['msisdn'], '112')  # Highest session_frequency
        self.assertEqual(len(top_users['total_duration']), 3)
        self.assertEqual(top_users['total_duration'].iloc[0]['msisdn'], '112')  # Highest total_duration

    def test_perform_kmeans_clustering(self):
        """Test k-means clustering."""
        features = ['session_frequency', 'total_duration', 'total_traffic']
        clustered_data, cluster_stats, kmeans, X_scaled = perform_kmeans_clustering(self.data, features, n_clusters=2)
        
        self.assertEqual(len(clustered_data), len(self.data))  # Check if the number of rows is unchanged
        self.assertIn('cluster', clustered_data.columns)  # Check if cluster labels are added
        self.assertEqual(len(cluster_stats), 2)  # Check if two clusters are created

    def test_find_optimal_k(self):
        """Test finding optimal k."""
        X = self.data[['session_frequency', 'total_duration', 'total_traffic']].values
        k_range = range(1, 5)
        inertias, silhouette_scores = find_optimal_k(X, k_range)
        
        self.assertEqual(len(inertias), len(k_range))  # Check if inertias are calculated for all k
        self.assertEqual(len(silhouette_scores), len(k_range))  # Check if silhouette scores are calculated for all k

    def test_get_top_apps_by_usage(self):
        """Test getting top applications by total usage."""
        top_apps = get_top_apps_by_usage(self.data)
        
        self.assertIn('social_media_usage', top_apps.index)  # Check if social_media_usage is in the result
        self.assertEqual(top_apps['social_media_usage'], 1500)  # Check total usage for social media

    def test_get_top_users_per_app(self):
        """Test getting top users per application."""
        top_users = get_top_users_per_app(self.data, n=2)
        
        self.assertEqual(len(top_users['social_media_usage']), 2)  # Check if top 2 users are returned
        self.assertEqual(top_users['social_media_usage'].iloc[0]['msisdn'], '112')  # Highest social media usage

if __name__ == '__main__':
    unittest.main() 