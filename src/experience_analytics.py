import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def get_experience_metrics(connection):
    """
    Get user experience metrics from the database
    """
    query = """
    SELECT 
        "MSISDN/Number" as msisdn,
        CAST("TCP DL Retrans. Vol (Bytes)" AS FLOAT) + CAST("TCP UL Retrans. Vol (Bytes)" AS FLOAT) as tcp_retransmission,
        (CAST("Avg RTT DL (ms)" AS FLOAT) + CAST("Avg RTT UL (ms)" AS FLOAT))/2.0 as avg_rtt,
        "Handset Type",
        (CAST("Avg Bearer TP DL (kbps)" AS FLOAT) + CAST("Avg Bearer TP UL (kbps)" AS FLOAT))/2.0 as avg_throughput,
        "Bearer Id"
    FROM xdr_data
    WHERE "TCP DL Retrans. Vol (Bytes)" IS NOT NULL 
    AND "TCP UL Retrans. Vol (Bytes)" IS NOT NULL
    AND "Avg RTT DL (ms)" IS NOT NULL
    AND "Avg RTT UL (ms)" IS NOT NULL
    AND "Avg Bearer TP DL (kbps)" IS NOT NULL
    AND "Avg Bearer TP UL (kbps)" IS NOT NULL
    """
    df = pd.read_sql_query(query, connection)
    
    # Additional data cleaning after fetching
    df['tcp_retransmission'] = pd.to_numeric(df['tcp_retransmission'], errors='coerce')
    df['avg_rtt'] = pd.to_numeric(df['avg_rtt'], errors='coerce')
    df['avg_throughput'] = pd.to_numeric(df['avg_throughput'], errors='coerce')
    
    return df

def aggregate_customer_experience(df):
    """
    Aggregate experience metrics per customer, handling missing values and outliers
    """
    # Handle missing values
    numeric_cols = ['tcp_retransmission', 'avg_rtt', 'avg_throughput']
    for col in numeric_cols:
        # Replace infinite values with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with mean, excluding infinite values
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    
    df['Handset Type'].fillna('undefined', inplace=True)
    
    # Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # Aggregate per customer
    agg_data = df.groupby('msisdn').agg({
        'tcp_retransmission': 'mean',
        'avg_rtt': 'mean',
        'Handset Type': lambda x: x.mode()[0] if not x.mode().empty else 'undefined',
        'avg_throughput': 'mean'
    }).reset_index()
    
    # Rename columns
    agg_data.columns = ['msisdn', 'avg_tcp_retransmission', 'avg_rtt', 'handset_type', 'avg_throughput']
    
    return agg_data

def get_metric_statistics(df, metric_col, n=10):
    """
    Get top, bottom, and most frequent values for a metric
    """
    stats = {
        'top_values': df[metric_col].nlargest(n).tolist(),
        'bottom_values': df[metric_col].nsmallest(n).tolist(),
        'most_frequent': df[metric_col].value_counts().head(n).index.tolist()
    }
    return stats

def analyze_throughput_by_handset(df):
    """
    Analyze throughput distribution per handset type
    """
    throughput_stats = df.groupby('handset_type')['avg_throughput'].agg([
        'mean', 'median', 'std', 'count'
    ]).round(2)
    
    return throughput_stats

def analyze_tcp_by_handset(df):
    """
    Analyze TCP retransmission per handset type
    """
    tcp_stats = df.groupby('handset_type')['avg_tcp_retransmission'].agg([
        'mean', 'median', 'std', 'count'
    ]).round(2)
    
    return tcp_stats

def perform_experience_clustering(df, n_clusters=3):
    """
    Perform k-means clustering on experience metrics
    """
    # Select features for clustering
    features = ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['experience_cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('experience_cluster')[features].agg(['mean', 'std']).round(2)
    
    return df, cluster_stats, kmeans

def plot_experience_distributions(df):
    """
    Create visualizations for experience metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Throughput by handset type
    sns.boxplot(x='handset_type', y='avg_throughput', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Average Throughput Distribution by Handset Type')
    axes[0, 0].set_xlabel('Handset Type')
    axes[0, 0].set_ylabel('Average Throughput (kbps)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # TCP retransmission by handset type
    sns.boxplot(x='handset_type', y='avg_tcp_retransmission', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('TCP Retransmission Volume by Handset Type')
    axes[0, 1].set_xlabel('Handset Type')
    axes[0, 1].set_ylabel('TCP Retransmission Volume (Bytes)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # RTT distribution
    sns.histplot(data=df, x='avg_rtt', ax=axes[1, 0])
    axes[1, 0].set_title('Average RTT Distribution')
    axes[1, 0].set_xlabel('Average RTT (ms)')
    axes[1, 0].set_ylabel('Count')
    
    # Cluster distribution
    if 'experience_cluster' in df.columns:
        sns.scatterplot(data=df, x='avg_throughput', y='avg_tcp_retransmission', 
                       hue='experience_cluster', ax=axes[1, 1])
        axes[1, 1].set_title('Experience Clusters')
        axes[1, 1].set_xlabel('Average Throughput (kbps)')
        axes[1, 1].set_ylabel('TCP Retransmission Volume (Bytes)')
    
    plt.tight_layout()
    return fig 