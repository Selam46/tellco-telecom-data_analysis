import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set page config
st.set_page_config(
    page_title="Telecom Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample data for the dashboard"""
    # Generate dates for the last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate user data
    np.random.seed(42)
    n_days = len(dates)
    
    # Base metrics with some randomness
    base_users = 1000
    base_sessions = 5000
    base_duration = 30
    
    data = pd.DataFrame({
        'Date': dates,
        'Total_Users': np.random.normal(base_users, 50, n_days).astype(int),
        'Active_Users': np.random.normal(base_users * 0.7, 30, n_days).astype(int),
        'New_Users': np.random.normal(50, 10, n_days).astype(int),
        'Sessions': np.random.normal(base_sessions, 200, n_days).astype(int),
        'Avg_Session_Duration': np.random.normal(base_duration, 5, n_days),
        'Network_Quality': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_days, p=[0.4, 0.3, 0.2, 0.1]),
        'Network_Type': np.random.choice(['4G', '5G', 'WiFi'], n_days, p=[0.3, 0.4, 0.3]),
        'Network_Speed': np.random.normal(50, 10, n_days),
        'Satisfaction_Score': np.random.normal(4.2, 0.3, n_days)
    })
    
    # Ensure no negative values
    data[['Total_Users', 'Active_Users', 'New_Users', 'Sessions']] = data[['Total_Users', 'Active_Users', 'New_Users', 'Sessions']].clip(lower=0)
    data['Avg_Session_Duration'] = data['Avg_Session_Duration'].clip(lower=0)
    data['Network_Speed'] = data['Network_Speed'].clip(lower=0)
    data['Satisfaction_Score'] = data['Satisfaction_Score'].clip(lower=0, upper=5)
    
    return data

def load_data():
    """Load and prepare data for the dashboard"""
    return generate_sample_data()

def aggregate_data(data, time_period):
    """Aggregate data based on selected time period"""
    # Set 'Date' as the index
    data.set_index('Date', inplace=True)
    
    if time_period == "Daily":
        return data.reset_index()  # Reset index to return 'Date' as a column
    elif time_period == "Weekly":
        # Handle numeric columns separately from categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Aggregate numeric columns with mean
        numeric_agg = data[numeric_cols].resample('W').mean()
        
        # For categorical columns, take the most frequent value
        categorical_agg = data[categorical_cols].resample('W').agg(lambda x: x.mode().iloc[0])
        
        # Combine the results
        result = pd.concat([numeric_agg, categorical_agg], axis=1)
        return result.reset_index()
    else:  # Monthly
        # Handle numeric columns separately from categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Aggregate numeric columns with mean
        numeric_agg = data[numeric_cols].resample('M').mean()
        
        # For categorical columns, take the most frequent value
        categorical_agg = data[categorical_cols].resample('M').agg(lambda x: x.mode().iloc[0])
        
        # Combine the results
        result = pd.concat([numeric_agg, categorical_agg], axis=1)
        return result.reset_index()

def user_overview_page():
    st.title("👥 User Overview Analysis")
    
    # Get data
    data = load_data()
    
    # Calculate metrics
    total_users = data['Total_Users'].iloc[-1]
    active_users = data['Active_Users'].iloc[-1]
    new_users = data['New_Users'].sum()
    
    # Display metrics with improved styling
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div style="background-color: #1f2937; color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #9ca3af; margin-bottom: 0.5rem;">Total Users</h3>
                <h2 style="color: white; margin: 0;">{total_users:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="background-color: #1f2937; color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #9ca3af; margin-bottom: 0.5rem;">Active Users</h3>
                <h2 style="color: white; margin: 0;">{active_users:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div style="background-color: #1f2937; color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #9ca3af; margin-bottom: 0.5rem;">New Users (This Month)</h3>
                <h2 style="color: white; margin: 0;">{new_users:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    # User growth visualization
    st.subheader("User Growth Over Time")
    fig = px.line(
        data,
        x='Date',
        y=['Total_Users', 'Active_Users', 'New_Users'],
        title='User Growth Trends',
        labels={'value': 'Number of Users', 'variable': 'Metric'}
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # User distribution
    st.subheader("User Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            data.iloc[-1:],
            values=['Active_Users', 'Total_Users - Active_Users'],
            names=['Active', 'Inactive'],
            title='Active vs Inactive Users'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            data,
            y=['Total_Users', 'Active_Users', 'New_Users'],
            title='User Distribution Statistics'
        )
        st.plotly_chart(fig, use_container_width=True)

def user_engagement_page():
    st.title("🎯 User Engagement Analysis")
    
    # Get data
    data = load_data()
    
    # Interactive filters
    col1, col2 = st.columns(2)
    with col1:
        time_period = st.selectbox(
            "Select Time Period",
            ["Daily", "Weekly", "Monthly"]
        )
    with col2:
        metric = st.selectbox(
            "Select Metric",
            ["Session Duration", "Number of Sessions", "Active Users"]
        )
    
    # Aggregate data based on selection
    agg_data = aggregate_data(data, time_period)
    
    # Create visualization based on selected metric
    st.subheader(f"{metric} Over Time")
    
    if metric == "Session Duration":
        fig = px.line(
            agg_data,
            x='Date',
            y='Avg_Session_Duration',
            title=f'Average Session Duration ({time_period})'
        )
        fig.update_layout(
            yaxis_title="Duration (minutes)",
            hovermode='x unified'
        )
    elif metric == "Number of Sessions":
        fig = px.bar(
            agg_data,
            x='Date',
            y='Sessions',
            title=f'Number of Sessions ({time_period})'
        )
        fig.update_layout(
            yaxis_title="Number of Sessions",
            hovermode='x unified'
        )
    else:  # Active Users
        fig = px.line(
            agg_data,
            x='Date',
            y='Active_Users',
            title=f'Active Users ({time_period})'
        )
        fig.update_layout(
            yaxis_title="Number of Active Users",
            hovermode='x unified'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.subheader("Engagement Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_sessions = agg_data['Sessions'].mean()
        st.markdown(f"""
            <div style="background-color: #1f2937; color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #9ca3af; margin-bottom: 0.5rem;">Average Sessions</h3>
                <h2 style="color: white; margin: 0;">{avg_sessions:.0f}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_duration = agg_data['Avg_Session_Duration'].mean()
        st.markdown(f"""
            <div style="background-color: #1f2937; color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #9ca3af; margin-bottom: 0.5rem;">Average Duration</h3>
                <h2 style="color: white; margin: 0;">{avg_duration:.1f} min</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        engagement_rate = (agg_data['Active_Users'] / agg_data['Total_Users']).mean() * 100
        st.markdown(f"""
            <div style="background-color: #1f2937; color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #9ca3af; margin-bottom: 0.5rem;">Engagement Rate</h3>
                <h2 style="color: white; margin: 0;">{engagement_rate:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True) 

def experience_analysis_page():
    st.title("🌟 Experience Analysis")
    
    # Get data
    data = load_data()
    
    # Network Performance Metrics
    st.subheader("Network Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Network Quality Distribution
        quality_counts = data['Network_Quality'].value_counts()
        fig = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title='Network Quality Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Network Speed Distribution
        fig = px.box(
            data,
            x='Network_Type',
            y='Network_Speed',
            title='Network Speed by Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional Network Metrics
    st.subheader("Network Performance Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        # Network Speed Over Time
        fig = px.line(
            data,
            x='Date',
            y='Network_Speed',
            color='Network_Type',
            title='Network Speed Trends'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Network Type Distribution
        network_counts = data['Network_Type'].value_counts()
        fig = px.bar(
            x=network_counts.index,
            y=network_counts.values,
            title='Network Type Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

def satisfaction_analysis_page():
    st.title("😊 Satisfaction Analysis")
    
    # Get data
    data = load_data()
    
    # Interactive filters
    st.subheader("Customer Satisfaction Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction_type = st.selectbox(
            "Select Satisfaction Metric",
            ["Overall Satisfaction", "Network Quality", "Customer Service"]
        )
    
    with col2:
        time_range = st.selectbox(
            "Select Time Range",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days"]
        )
    
    # Filter data based on time range
    days = int(time_range.split()[1])
    filtered_data = data.tail(days)
    
    # Create visualization
    fig = px.scatter(
        filtered_data,
        x='Date',
        y='Satisfaction_Score',
        color='Network_Type',
        title=f'Customer Satisfaction Score ({time_range})'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional satisfaction metrics
    st.subheader("Satisfaction Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_satisfaction = filtered_data['Satisfaction_Score'].mean()
        st.metric("Average Satisfaction", f"{avg_satisfaction:.1f}/5")
    
    with col2:
        satisfaction_trend = filtered_data['Satisfaction_Score'].pct_change().mean() * 100
        st.metric("Satisfaction Trend", f"{satisfaction_trend:.1f}%")
    
    with col3:
        satisfaction_std = filtered_data['Satisfaction_Score'].std()
        st.metric("Satisfaction Variability", f"{satisfaction_std:.2f}")

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["User Overview", "User Engagement", "Experience Analysis", "Satisfaction Analysis"]
    )
    
    # Load data
    data = load_data()
    
    # Page routing
    if page == "User Overview":
        user_overview_page()
    elif page == "User Engagement":
        user_engagement_page()
    elif page == "Experience Analysis":
        experience_analysis_page()
    elif page == "Satisfaction Analysis":
        satisfaction_analysis_page()

if __name__ == "__main__":
    main()
