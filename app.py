import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from models.prophet_model import train_prophet_model, forecast_aqi
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #f5f7f9;
        color: #1E293B;
    }
    
    /* Card styling */
    .css-card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Dashboard title */
    .dashboard-title {
        font-size: 36px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 15px;
        border-left: 5px solid #1E3A8A;
        padding-left: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1E3A8A;
    }
    
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Menu item styling */
    .menu-item {
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
        color: white;
    }
    
    .menu-item:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    .menu-item.active {
        background-color: rgba(255, 255, 255, 0.3);
        border-left: 4px solid white;
    }
    
    /* Custom radio buttons for menu */
    .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0;
    }
    
    .stRadio > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        padding: 10px 15px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .stRadio > div > div:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    .stRadio > div > div[data-testid*="StRadioOption"] > label {
        color: white !important;
        font-weight: 500;
    }
    
    /* AQI category badges */
    .aqi-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        margin-right: 10px;
        color: white;
    }
    
    /* Button styling */
    .custom-button {
        background-color: #1E3A8A;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .custom-button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6B7280;
    }
    
    /* Welcome page styling */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        text-align: center;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 15px;
        padding: 40px;
        color: white;
    }
    
    .welcome-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 20px;
        color: white;
    }
    
    .welcome-description {
        font-size: 18px;
        margin-bottom: 30px;
        max-width: 800px;
        line-height: 1.6;
    }
    
    .team-section {
        margin-top: 30px;
        margin-bottom: 40px;
    }
    
    .team-title {
        font-size: 24px;
        margin-bottom: 15px;
        color: white;
    }
    
    .team-members {
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    
    .team-member {
        font-size: 18px;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px 20px;
        border-radius: 20px;
    }
    
    .enter-button {
        background-color: #2563EB;
        color: white;
        padding: 12px 30px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        border: none;
        margin-top: 20px;
    }
    
    .enter-button:hover {
        background-color: #1E40AF;
        transform: scale(1.05);
    }
    
    /* Data table styling */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 5px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        background-color: #f1f5f9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border-top: 1px solid #e5e7eb;
        color: #6B7280;
    }
    
    /* Fix text color in all elements */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #1E293B !important;
    }
    
    /* Fix sidebar text color */
    .css-1d391kg .stMarkdown, .css-1d391kg p, .css-1lcbmhc .stMarkdown, .css-1lcbmhc p {
        color: white !important;
    }
    
    /* Database table styling */
    .db-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
        background-color: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .db-table th {
        background-color: #1E3A8A;
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
    }
    
    .db-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .db-table tr:last-child td {
        border-bottom: none;
    }
    
    .db-table tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    .db-table tr:hover {
        background-color: #f1f5f9;
    }
    </style>
    """, unsafe_allow_html=True)

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def welcome_page():
    set_background("https://i.imgur.com/w9qgmgO.jpeg")
    
    st.markdown("""
    <div class="welcome-container">
        <h1 class="welcome-title">Air Quality Prediction and Forecasting</h1>
        
        <p class="welcome-description">
            This interactive dashboard provides real-time prediction and forecasting of the Air Quality Index (AQI) 
            for Kathmandu using advanced machine learning models and the Prophet forecasting system.
            Visualize historical data, predict AQI based on environmental parameters, and forecast future air quality trends.
        </p>
        
        <div class="team-section">
            <h2 class="team-title">Our Team</h2>
            <div class="team-members">
                <div class="team-member">Sangam Paudel</div>
                <div class="team-member">Saroj Rawal</div>
                <div class="team-member">Subesh Yadav</div>
            </div>
        </div>
        
        <button class="enter-button" id="enter-dashboard-btn" onclick="document.getElementById('enter-dashboard-form').submit();">
            Enter Dashboard
        </button>
        <form id="enter-dashboard-form" method="post">
            <input type="hidden" name="entered_dashboard" value="true">
        </form>
    </div>
    """, unsafe_allow_html=True)
    
    # Handle the button click
    if st.button("Enter Dashboard", key="enter_dashboard_hidden", help="Click to enter the dashboard"):
        st.session_state.entered_dashboard = True
        return True
    return False

def create_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🌬️ Air Quality Dashboard</div>', unsafe_allow_html=True)
        
        # Navigation menu with custom styling for hover effects
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] > div:first-child > div {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 10px 15px;
            margin-bottom: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] > div:first-child > div:hover {
            background-color: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        selected = st.radio(
            "Navigation",
            ["Visualize", "Predict", "Forecast", "Database"],
            label_visibility="collapsed"
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # About section
        st.markdown('<div class="sidebar-title" style="font-size: 18px;">About</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: white; padding: 10px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.1);">
                This web app predicts and forecasts Air Quality Index (AQI) of Kathmandu using a pre-trained Random Forest Regressor and Facebook Prophet model.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Links section
        st.markdown('<div class="sidebar-title" style="font-size: 18px; margin-top: 20px;">Resources</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <a href="https://www.kaggle.com/sarojrawal" target="_blank" style="text-decoration: none;">
                    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px; text-align: center; color: white; transition: all 0.3s;">
                        <img src="https://www.kaggle.com/static/images/site-logo.svg" width="20" style="filter: brightness(0) invert(1);"> Kaggle
                    </div>
                </a>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <a href="https://github.com/saroj-2004/saroj" target="_blank" style="text-decoration: none;">
                    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px; text-align: center; color: white; transition: all 0.3s;">
                        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="filter: brightness(0) invert(1);"> GitHub
                    </div>
                </a>
                """, 
                unsafe_allow_html=True
            )
        
        # Rating section
        st.markdown('<div class="sidebar-title" style="font-size: 18px; margin-top: 20px;">⭐ Rate Us</div>', unsafe_allow_html=True)
        rating = st.slider("How would you rate our app?", 0, 10, 7)
        
        if rating > 0:
            st.success(f"Thank you for your rating of {rating}/10! 😊")
        
        # Footer
        st.markdown(
            """
            <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(0, 0, 0, 0.2); padding: 10px; text-align: center; color: white; font-size: 12px;">
                © 2025 Air Quality Dashboard
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    return selected

def visualize_page(data):
    st.markdown('<h1 class="dashboard-title">Air Quality Visualization</h1>', unsafe_allow_html=True)
    
    if data.empty:
        st.warning("No data available for visualization.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Statistics", "🗺️ Data Table"])
    
    with tab1:
        st.markdown('<div class="section-header">Historical AQI Trends</div>', unsafe_allow_html=True)
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=data["Datetime"].min().date(),
                min_value=data["Datetime"].min().date(),
                max_value=data["Datetime"].max().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=data["Datetime"].max().date(),
                min_value=data["Datetime"].min().date(),
                max_value=data["Datetime"].max().date()
            )
        
        # Filter data based on date range
        filtered_data = data[(data["Datetime"].dt.date >= start_date) & 
                             (data["Datetime"].dt.date <= end_date)]
        
        # Create interactive time series plot with Plotly
        fig = px.line(
            filtered_data, 
            x="Datetime", 
            y="AQI",
            title="Air Quality Index Over Time",
            labels={"AQI": "Air Quality Index", "Datetime": "Date"},
            line_shape="spline",
            template="plotly_white"
        )
        
        fig.update_layout(
            height=500,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="AQI Value",
            legend_title="Legend",
            font=dict(family="Arial", size=12),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        # Add AQI threshold lines
        thresholds = [
            {"value": 50, "name": "Good", "color": "green"},
            {"value": 100, "name": "Moderate", "color": "yellow"},
            {"value": 150, "name": "Unhealthy for Sensitive Groups", "color": "orange"},
            {"value": 200, "name": "Unhealthy", "color": "red"},
            {"value": 300, "name": "Very Unhealthy", "color": "purple"}
        ]
        
        for threshold in thresholds:
            fig.add_shape(
                type="line",
                x0=filtered_data["Datetime"].min(),
                y0=threshold["value"],
                x1=filtered_data["Datetime"].max(),
                y1=threshold["value"],
                line=dict(color=threshold["color"], width=1, dash="dash"),
            )
            
            fig.add_annotation(
                x=filtered_data["Datetime"].max(),
                y=threshold["value"],
                text=threshold["name"],
                showarrow=False,
                xshift=10,
                font=dict(size=10, color=threshold["color"])
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly average AQI
        if len(filtered_data) > 30:  # Only show if we have enough data
            st.markdown('<div class="section-header">Monthly Average AQI</div>', unsafe_allow_html=True)
            
            # Extract month and year from datetime
            filtered_data['Month'] = filtered_data['Datetime'].dt.strftime('%b %Y')
            monthly_avg = filtered_data.groupby('Month')['AQI'].mean().reset_index()
            
            fig = px.bar(
                monthly_avg, 
                x='Month', 
                y='AQI',
                title='Monthly Average AQI',
                labels={'AQI': 'Average AQI', 'Month': 'Month'},
                color='AQI',
                color_continuous_scale=px.colors.sequential.Viridis,
                template="plotly_white"
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Month",
                yaxis_title="Average AQI",
                font=dict(family="Arial", size=12),
                margin=dict(l=40, r=40, t=40, b=40),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">AQI Statistics</div>', unsafe_allow_html=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Average AQI</div>
                    <div class="metric-value" style="color: #2563EB;">{data['AQI'].mean():.1f}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Maximum AQI</div>
                    <div class="metric-value" style="color: #DC2626;">{data['AQI'].max():.1f}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Minimum AQI</div>
                    <div class="metric-value" style="color: #10B981;">{data['AQI'].min():.1f}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Data Points</div>
                    <div class="metric-value" style="color: #6366F1;">{len(data)}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # AQI Distribution
        st.markdown('<div class="section-header">AQI Distribution</div>', unsafe_allow_html=True)
        
        fig = px.histogram(
            data, 
            x="AQI",
            nbins=30,
            title="Distribution of AQI Values",
            labels={"AQI": "Air Quality Index", "count": "Frequency"},
            color_discrete_sequence=["#1E3A8A"],
            template="plotly_white"
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="AQI Value",
            yaxis_title="Frequency",
            font=dict(family="Arial", size=12),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AQI Category Distribution
        st.markdown('<div class="section-header">AQI Category Distribution</div>', unsafe_allow_html=True)
        
        # Function to categorize AQI
        def categorize_aqi_simple(aqi):
            if aqi <= 50:
                return "Good"
            elif aqi <= 100:
                return "Moderate"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                return "Unhealthy"
            elif aqi <= 300:
                return "Very Unhealthy"
            else:
                return "Hazardous"
        
        # Add category column
        data['AQI_Category'] = data['AQI'].apply(categorize_aqi_simple)
        
        # Count categories
        category_counts = data['AQI_Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Define category colors
        category_colors = {
            "Good": "#10B981",
            "Moderate": "#FBBF24",
            "Unhealthy for Sensitive Groups": "#F97316",
            "Unhealthy": "#DC2626",
            "Very Unhealthy": "#7C3AED",
            "Hazardous": "#7F1D1D"
        }
        
        # Create pie chart
        fig = px.pie(
            category_counts, 
            values='Count', 
            names='Category',
            title='Distribution of AQI Categories',
            color='Category',
            color_discrete_map=category_colors,
            hole=0.4,
            template="plotly_white"
        )
        
        fig.update_layout(
            height=500,
            font=dict(family="Arial", size=12),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">Air Quality Data Table</div>', unsafe_allow_html=True)
        
        # Add search functionality
        search_term = st.text_input("Search data (by date format YYYY-MM-DD):", "")
        
        if search_term:
            filtered_data = data[data["Datetime"].dt.strftime("%Y-%m-%d").str.contains(search_term)]
        else:
            filtered_data = data
        
        # Add pagination
        page_size = st.selectbox("Rows per page:", [10, 25, 50, 100])
        total_pages = len(filtered_data) // page_size + (1 if len(filtered_data) % page_size > 0 else 0)
        
        if total_pages > 0:
            page_number = st.slider("Page:", 1, max(1, total_pages), 1)
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, len(filtered_data))
            
            # Display page info
            st.markdown(f"Showing {start_idx + 1} to {end_idx} of {len(filtered_data)} entries")
            
            # Display the data
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(filtered_data.iloc[start_idx:end_idx], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No data matches your search criteria.")

def predict_page(model):
    st.markdown('<h1 class="dashboard-title">Air Quality Prediction</h1>', unsafe_allow_html=True)
    
    # Create two columns layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #1E3A8A;">
                Enter the environmental parameters below to predict the Air Quality Index (AQI).
                The model will use these values to estimate the current air quality level.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display all input parameters together in a single form
        st.markdown('<div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
        
        # Particulate Matter section
        st.markdown('<div style="font-weight: 600; color: #1E3A8A; margin-bottom: 10px; font-size: 16px;">Particulate Matter</div>', unsafe_allow_html=True)
        col_pm1, col_pm2 = st.columns(2)
        with col_pm1:
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, value=20.3, format="%.2f",
                                help="Fine particulate matter with diameter less than 2.5 micrometers")
        with col_pm2:
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, value=30.4, format="%.2f",
                                help="Particulate matter with diameter less than 10 micrometers")
        
        # Gases section
        st.markdown('<div style="font-weight: 600; color: #1E3A8A; margin-top: 15px; margin-bottom: 10px; font-size: 16px;">Gases</div>', unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            co = st.number_input("CO (μg/m³)", min_value=0.0, value=0.5, format="%.2f",
                              help="Carbon Monoxide concentration")
            no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, value=10.3, format="%.2f",
                               help="Nitrogen Dioxide concentration")
        with col_g2:
            so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, value=5.0, format="%.2f",
                               help="Sulfur Dioxide concentration")
            o3 = st.number_input("O3 (μg/m³)", min_value=0.0, value=30.0, format="%.2f",
                              help="Ozone concentration")
        
        # Weather section
        st.markdown('<div style="font-weight: 600; color: #1E3A8A; margin-top: 15px; margin-bottom: 10px; font-size: 16px;">Weather</div>', unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            temp = st.number_input("Temperature (°C)", min_value=-10.0, value=25.2, format="%.2f",
                                help="Ambient temperature in degrees Celsius")
        with col_w2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, format="%.2f",
                                    help="Relative humidity percentage")
        with col_w3:
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0, format="%.2f",
                                      help="Wind speed in kilometers per hour")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button with enhanced styling
        predict_clicked = st.button("Predict Air Quality", type="primary", use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        # Function to categorize AQI
        def categorize_aqi(aqi):
            if aqi <= 50:
                return "Good", "green", "Enjoy your day outside!", "😊"
            elif aqi <= 100:
                return "Moderate", "yellow", "Reduce outdoor exercises!", "😐"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups", "orange", "Wear a mask outdoors. Close windows to avoid dirty air.", "😷"
            elif aqi <= 200:
                return "Unhealthy", "red", "Everyone may experience health effects. Avoid outdoor activities.", "🚫"
            elif aqi <= 300:
                return "Very Unhealthy", "purple", "Serious health risks. Stay indoors and use an air purifier if possible.", "⚠️"
            else:
                return "Hazardous", "maroon", "Health warning. Avoid going outside, wear a high-quality mask.", "☣️"
        
        # Display prediction or placeholder
        if predict_clicked:
            try:
                # Prepare input data
                input_data = np.array([[pm25, pm10, co, no2, so2, o3, temp, humidity, wind_speed]])
                
                # Make prediction
                predicted_aqi = model.predict(input_data)[0]
                
                # Get AQI category
                category, color, advice, emoji = categorize_aqi(predicted_aqi)
                
                # Display prediction with nice styling
                st.markdown(
                    f"""
                    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;">
                        <div style="font-size: 18px; color: #6B7280; margin-bottom: 10px;">Predicted AQI</div>
                        <div style="font-size: 48px; font-weight: 700; color: #{color if color != 'yellow' else 'CC9900'}; margin-bottom: 15px;">{predicted_aqi:.1f}</div>
                        <div style="background-color: #{color if color != 'yellow' else 'CC9900'}; color: white; padding: 8px 15px; border-radius: 20px; display: inline-block; font-weight: 600; margin-bottom: 20px;">
                            {category} {emoji}
                        </div>
                        <div style="font-size: 16px; line-height: 1.5; color: #4B5563; background-color: #f8fafc; padding: 15px; border-radius: 8px; text-align: left;">
                            <strong>Recommendation:</strong><br>
                            {advice}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Add gauge chart for visual representation
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_aqi,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Air Quality Index", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': 'green'},
                            {'range': [50, 100], 'color': '#CC9900'},
                            {'range': [100, 150], 'color': 'orange'},
                            {'range': [150, 200], 'color': 'red'},
                            {'range': [200, 300], 'color': 'purple'},
                            {'range': [300, 500], 'color': 'maroon'}
                        ],
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="white",
                    font=dict(family="Arial", size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"⚠️ Prediction failed. Please check the input values or try again later.")
                st.error(f"Error details: {e}")
        else:
            # Display placeholder when no prediction is made
            st.markdown(
                """
                <div style="background-color: #f8fafc; padding: 30px; border-radius: 10px; text-align: center; height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <img src="https://i.imgur.com/LDtA4Nw.png" width="100" style="opacity: 0.5; margin-bottom: 20px;">
                    <div style="font-size: 18px; color: #6B7280; margin-bottom: 10px;">No Prediction Yet</div>
                    <div style="font-size: 14px; color: #9CA3AF; max-width: 300px;">
                        Enter the environmental parameters and click the Predict button to see the air quality prediction.
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

def forecast_page(data):
    st.markdown('<h1 class="dashboard-title">Air Quality Forecasting</h1>', unsafe_allow_html=True)
    
    if data.empty:
        st.warning("No data available for forecasting. Please upload historical data first.")
        return
    
    # Create a more compact forecast configuration
    st.markdown('<div class="section-header">Forecast Configuration</div>', unsafe_allow_html=True)
    
    # Use a simpler layout with fewer columns
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_options = [24, 48, 72, 96, 120, 144, 168]
        forecast_hours = st.radio(
            "Forecast Duration",
            options=forecast_options,
            format_func=lambda x: f"{x} hours ({x//24} days)",
            horizontal=True
        )
    
    with col2:
        forecast_clicked = st.button("Generate Forecast", type="primary", use_container_width=True)
    
    if forecast_clicked:
        with st.spinner("Generating forecast..."):
            try:
                # Create forecast time points more efficiently
                current_time = datetime.now()
                forecast_time = [current_time + timedelta(hours=i) for i in range(1, forecast_hours + 1)]
                
                # Get forecasted AQI data
                # Use st.cache to avoid retraining the model if possible
                @st.cache_data(ttl=3600)  # Cache for 1 hour
                def get_forecast(data, hours):
                    pmodel = train_prophet_model(data)
                    return forecast_aqi(pmodel, hours)
                
                forecast_df = get_forecast(data, forecast_hours)
                forecast_df["Datetime"] = forecast_time
                
                # Function to categorize AQI - simplified
                def categorize_aqi_simple(aqi):
                    if aqi <= 50:
                        return "Good", "green"
                    elif aqi <= 100:
                        return "Moderate", "#CC9900"
                    elif aqi <= 150:
                        return "Unhealthy for Sensitive Groups", "orange"
                    elif aqi <= 200:
                        return "Unhealthy", "red"
                    elif aqi <= 300:
                        return "Very Unhealthy", "purple"
                    else:
                        return "Hazardous", "maroon"
                
                # Calculate statistics once
                avg_aqi = forecast_df["Forecasted AQI"].mean()
                max_aqi = forecast_df["Forecasted AQI"].max()
                min_aqi = forecast_df["Forecasted AQI"].min()
                
                avg_category, avg_color = categorize_aqi_simple(avg_aqi)
                max_category, max_color = categorize_aqi_simple(max_aqi)
                min_category, min_color = categorize_aqi_simple(min_aqi)
                
                # Display forecast results in a more compact format
                tab1, tab2 = st.tabs(["📈 Forecast Chart", "📊 Data Table"])
                
                with tab1:
                    # Create a simpler chart with fewer elements
                    fig = go.Figure()
                    
                    # Add the main forecast line
                    fig.add_trace(go.Scatter(
                        x=forecast_df["Datetime"],
                        y=forecast_df["Forecasted AQI"],
                        mode='lines',
                        name='Forecasted AQI',
                        line=dict(color='#1E3A8A', width=3)
                    ))
                    
                    # Add confidence interval - simplified
                    fig.add_trace(go.Scatter(
                        x=forecast_df["Datetime"],
                        y=forecast_df["Upper Bound"],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df["Datetime"],
                        y=forecast_df["Lower Bound"],
                        mode='lines',
                        name='Lower Bound',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(30, 58, 138, 0.2)',
                        showlegend=False
                    ))
                    
                    # Simplified layout
                    fig.update_layout(
                        title=f"{forecast_hours}-Hour AQI Forecast",
                        xaxis_title="Date & Time",
                        yaxis_title="Air Quality Index (AQI)",
                        height=400,
                        template="plotly_white",
                        margin=dict(l=30, r=30, t=40, b=30),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add forecast summary in a more compact format
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); text-align: center;">
                                <div style="font-size: 14px; color: #6B7280;">Average AQI</div>
                                <div style="font-size: 24px; font-weight: 700; color: {avg_color};">{avg_aqi:.1f}</div>
                                <div style="font-size: 14px; color: {avg_color};">{avg_category}</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); text-align: center;">
                                <div style="font-size: 14px; color: #6B7280;">Maximum AQI</div>
                                <div style="font-size: 24px; font-weight: 700; color: {max_color};">{max_aqi:.1f}</div>
                                <div style="font-size: 14px; color: {max_color};">{max_category}</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            f"""
                            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); text-align: center;">
                                <div style="font-size: 14px; color: #6B7280;">Minimum AQI</div>
                                <div style="font-size: 24px; font-weight: 700; color: {min_color};">{min_aqi:.1f}</div>
                                <div style="font-size: 14px; color: {min_color};">{min_category}</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                with tab2:
                    # Format the datetime for better display
                    display_df = forecast_df.copy()
                    display_df["Datetime"] = display_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
                    
                    # Round the values for cleaner display
                    display_df["Forecasted AQI"] = display_df["Forecasted AQI"].round(1)
                    display_df["Lower Bound"] = display_df["Lower Bound"].round(1)
                    display_df["Upper Bound"] = display_df["Upper Bound"].round(1)
                    
                    # Add AQI category
                    display_df["AQI Category"] = display_df["Forecasted AQI"].apply(lambda x: categorize_aqi_simple(x)[0])
                    
                    # Reorder columns
                    display_df = display_df[["Datetime", "Forecasted AQI", "Lower Bound", "Upper Bound", "AQI Category"]]
                    
                    # Rename columns for better display
                    display_df.columns = ["Date & Time", "Forecasted AQI", "Lower Bound", "Upper Bound", "AQI Category"]
                    
                    # Display only a subset of data for faster loading
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(display_df.head(50), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if len(display_df) > 50:
                        st.info(f"Showing first 50 of {len(display_df)} records. Download the full dataset using the button below.")
                    
                    # Add download button
                    csv = display_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="aqi_forecast.csv" class="custom-button" style="display: inline-block; margin-top: 10px; text-decoration: none; padding: 10px 15px; background-color: #1E3A8A; color: white; border-radius: 5px;">Download Forecast Data</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.info("Try with a shorter forecast duration or refresh the page.")
    else:
        # Show a placeholder when no forecast is generated
        st.info("Select a forecast duration and click 'Generate Forecast' to see the prediction.")
        
        # Display a sample image to give users an idea of what to expect
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <img src="https://i.imgur.com/LDtA4Nw.png" width="80" style="opacity: 0.5; margin-bottom: 10px;">
                <div style="font-size: 16px; color: #6B7280;">Forecast will appear here</div>
            </div>
            """,
            unsafe_allow_html=True
        )
def database_page(data):
    st.markdown('<h1 class="dashboard-title">Database Management</h1>', unsafe_allow_html=True)
    
    # Create tabs for different database operations
    tab1, tab2, tab3 = st.tabs(["📋 View Data", "➕ Add Data", "🔄 Update Data"])
    
    with tab1:
        st.markdown('<div class="section-header">Air Quality Database</div>', unsafe_allow_html=True)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not data.empty:
                min_date = data["Datetime"].min().date()
                max_date = data["Datetime"].max().date()
                date_filter = st.date_input(
                    "Filter by Date",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
        
        with col2:
            if not data.empty:
                min_aqi = float(data["AQI"].min())
                max_aqi = float(data["AQI"].max())
                aqi_range = st.slider(
                    "AQI Range",
                    min_value=min_aqi,
                    max_value=max_aqi,
                    value=(min_aqi, max_aqi)
                )
        
        with col3:
            search = st.text_input("Search", placeholder="Enter keywords...")
        
        # Apply filters
        filtered_data = data.copy()
        if not data.empty:
            if len(date_filter) == 2:
                start_date, end_date = date_filter
                filtered_data = filtered_data[(filtered_data["Datetime"].dt.date >= start_date) & 
                                             (filtered_data["Datetime"].dt.date <= end_date)]
            
            filtered_data = filtered_data[(filtered_data["AQI"] >= aqi_range[0]) & 
                                         (filtered_data["AQI"] <= aqi_range[1])]
            
            if search:
                filtered_data = filtered_data[filtered_data.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
        
        # Display data
        if not filtered_data.empty:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(filtered_data, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"Showing {len(filtered_data)} of {len(data)} records")
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = filtered_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="air_quality_data.csv" class="custom-button" style="display: inline-block; text-decoration: none; padding: 10px 15px; background-color: #1E3A8A; color: white; border-radius: 5px; width: 100%; text-align: center;">Export to CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                excel = filtered_data.to_csv(index=False)
                b64 = base64.b64encode(excel.encode()).decode()
                href = f'<a href="data:application/vnd.ms-excel;base64,{b64}" download="air_quality_data.xlsx" class="custom-button" style="display: inline-block; text-decoration: none; padding: 10px 15px; background-color: #1E3A8A; color: white; border-radius: 5px; width: 100%; text-align: center;">Export to Excel</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No data available or no records match your filters.")
    
    with tab2:
        st.markdown('<div class="section-header">Add New Data</div>', unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #1E3A8A;">
                Add new air quality measurements to the database. All fields are required.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_date = st.date_input("Date", value=datetime.now().date())
            new_time = st.time_input("Time", value=datetime.now().time())
            new_aqi = st.number_input("AQI", min_value=0.0, format="%.2f")
            new_pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, format="%.2f")
            new_pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, format="%.2f")
        
        with col2:
            new_co = st.number_input("CO (μg/m³)", key="new_co", min_value=0.0, format="%.2f")
            new_no2 = st.number_input("NO2 (μg/m³)", key="new_no2", min_value=0.0, format="%.2f")
            new_so2 = st.number_input("SO2 (μg/m³)", key="new_so2", min_value=0.0, format="%.2f")
            new_o3 = st.number_input("O3 (μg/m³)", key="new_o3", min_value=0.0, format="%.2f")
            new_temp = st.number_input("Temperature (°C)", key="new_temp", format="%.2f")
        
        # Submit button
        if st.button("Add Record", type="primary", use_container_width=True):
            st.success("Record added successfully! (This is a simulation - no actual database update)")
            
            # In a real application, you would add the record to the database here
            st.info("In a production environment, this would add the record to your database.")
    
    with tab3:
        st.markdown('<div class="section-header">Update Existing Data</div>', unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #1E3A8A;">
                Select a record to update its values. Changes will be saved to the database.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Record selection
        if not data.empty:
            # Format datetime for display
            date_options = data["Datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            selected_record = st.selectbox("Select Record to Update", date_options)
            
            # Get the selected record
            record_idx = date_options.index(selected_record)
            record = data.iloc[record_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                update_aqi = st.number_input("AQI", value=float(record["AQI"]), format="%.2f")
                update_pm25 = st.number_input("PM2.5 (μg/m³)", key="update_pm25", value=20.3, format="%.2f")
                update_pm10 = st.number_input("PM10 (μg/m³)", key="update_pm10", value=30.4, format="%.2f")
            
            with col2:
                update_co = st.number_input("CO (μg/m³)", key="update_co", value=0.5, format="%.2f")
                update_no2 = st.number_input("NO2 (μg/m³)", key="update_no2", value=10.3, format="%.2f")
                update_o3 = st.number_input("O3 (μg/m³)", key="update_o3", value=30.0, format="%.2f")
            
            # Update button
            if st.button("Update Record", type="primary", use_container_width=True):
                st.success("Record updated successfully! (This is a simulation - no actual database update)")
                
                # In a real application, you would update the record in the database here
                st.info("In a production environment, this would update the record in your database.")
        else:
            st.info("No data available to update.")

def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state if it doesn't exist
    if "entered_dashboard" not in st.session_state:
        st.session_state.entered_dashboard = False
    
    # If the user hasn't entered the dashboard yet, show the welcome page
    if not st.session_state.entered_dashboard:
        if welcome_page():
            return  # Stop further code execution and stay on the welcome page
    
    else:
        # Load model & data
        try:
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            st.error("Model file not found. Please make sure 'model.pkl' is in the current directory.")
            model = None
        
        @st.cache_data
        def load_data():
            try:
                return pd.read_csv("Air_Quality_dataset_of_kathmandu_modified.csv", parse_dates=["Datetime"])
            except FileNotFoundError:
                return pd.DataFrame(columns=["Datetime", "AQI"])
        
        data = load_data()
        
        # Create sidebar and get selected option
        selection = create_sidebar()
        
        # Display the selected page
        if selection == "Visualize":
            visualize_page(data)
        elif selection == "Predict":
            if model is not None:
                predict_page(model)
            else:
                st.error("Model not available. Please check if the model file exists.")
        elif selection == "Forecast":
            forecast_page(data)
        elif selection == "Database":
            database_page(data)
        
        # Add footer
        st.markdown(
            """
            <div class="footer">
                <p>© 2025 Air Quality Dashboard | Developed by Sangam Paudel, Saroj Rawal, Subesh Yadav</p>
                <p>Data source: <a href="https://www.kaggle.com/sarojrawal" target="_blank">Kaggle</a> | 
                Code: <a href="https://github.com/saroj-2004/saroj" target="_blank">GitHub</a></p>
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()