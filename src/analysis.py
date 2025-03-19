import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def get_top_handsets(connection, limit=10):
    """
    Get top handsets used by customers
    """
    query = """
    SELECT "Handset Type" as handset_type, COUNT(*) as count
    FROM xdr_data
    WHERE "Handset Type" IS NOT NULL
    GROUP BY "Handset Type"
    ORDER BY count DESC
    LIMIT %s
    """
    return pd.read_sql_query(query, connection, params=[limit])

def get_top_manufacturers(connection, limit=3):
    """
    Get top handset manufacturers
    """
    query = """
    SELECT "Handset Manufacturer" as handset_manufacturer, COUNT(*) as count
    FROM xdr_data
    WHERE "Handset Manufacturer" IS NOT NULL
    GROUP BY "Handset Manufacturer"
    ORDER BY count DESC
    LIMIT %s
    """
    return pd.read_sql_query(query, connection, params=[limit])

def get_top_handsets_per_manufacturer(connection, manufacturers, limit=5):
    """
    Get top handsets per manufacturer
    """
    placeholders = ','.join(['%s'] * len(manufacturers))
    query = f"""
    WITH RankedHandsets AS (
        SELECT 
            "Handset Manufacturer" as handset_manufacturer,
            "Handset Type" as handset_type,
            COUNT(*) as count,
            ROW_NUMBER() OVER (PARTITION BY "Handset Manufacturer" ORDER BY COUNT(*) DESC) as rank
        FROM xdr_data
        WHERE "Handset Manufacturer" IN ({placeholders})
        GROUP BY "Handset Manufacturer", "Handset Type"
    )
    SELECT *
    FROM RankedHandsets
    WHERE rank <= %s
    ORDER BY handset_manufacturer, rank
    """
    params = manufacturers + [limit]
    return pd.read_sql_query(query, connection, params=params)

def aggregate_user_behavior(connection):
    """
    Aggregate user behavior metrics
    """
    query = """
    SELECT 
        "MSISDN/Number" as msisdn,
        COUNT(*) as number_of_sessions,
        SUM("Dur. (ms)") as total_duration,
        SUM("Total DL (Bytes)") as total_dl_data,
        SUM("Total UL (Bytes)") as total_ul_data,
        SUM("Total DL (Bytes)" + "Total UL (Bytes)") as total_data_volume,
        SUM("Social Media DL (Bytes)" + "Social Media UL (Bytes)") as social_media_data,
        SUM("Google DL (Bytes)" + "Google UL (Bytes)") as google_data,
        SUM("Email DL (Bytes)" + "Email UL (Bytes)") as email_data,
        SUM("Youtube DL (Bytes)" + "Youtube UL (Bytes)") as youtube_data,
        SUM("Netflix DL (Bytes)" + "Netflix UL (Bytes)") as netflix_data,
        SUM("Gaming DL (Bytes)" + "Gaming UL (Bytes)") as gaming_data,
        SUM("Other DL (Bytes)" + "Other UL (Bytes)") as other_data
    FROM xdr_data
    GROUP BY "MSISDN/Number"
    """
    return pd.read_sql_query(query, connection)

def segment_users_by_duration(df):
    """
    Segment users into decile classes based on total duration
    """
    df['duration_decile'] = pd.qcut(df['total_duration'], q=10, labels=False)
    return df.groupby('duration_decile')['total_data_volume'].sum()

def perform_pca(df, columns):
    """
    Perform PCA on specified columns
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    return pca, pca_result 