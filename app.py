# app.py
# Streamlit app: Prediction | Forecasting | Visualizations | Chatbot
# pip install: streamlit numpy pandas joblib tensorflow statsmodels scikit-learn matplotlib plotly

import os, io, math, json, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Deep learning & forecasting
import tensorflow as tf
from tensorflow.keras.models import load_model
import statsmodels.api as sm

st.set_page_config(page_title="✈️ Flight Fare Intelligence", layout="wide")

# -------------------------
# Sidebar: Model paths
# -------------------------
st.sidebar.header("⚙️ Settings")
default_ann_path = "/workspaces/Vamsithesis/best_model.h5"
default_scaler_path = "/workspaces/Vamsithesis/scaler.pkl"
default_sarimax_path = "/workspaces/Vamsithesis/sarimax_fare_forecast.pkl"
ann_path = st.sidebar.text_input("ANN model (.h5) path", default_ann_path)
scaler_path = st.sidebar.text_input("Scaler (.pkl) path", default_scaler_path)
sarimax_path = st.sidebar.text_input("SARIMAX model (.pkl) path", default_sarimax_path)
st.sidebar.markdown("---")
st.sidebar.caption("Paths default to your Colab saves; change if needed.")

# -------------------------
# Caching loaders
# -------------------------
@st.cache_resource(show_spinner=False)
def load_ann_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ANN .h5 not found at {path}")
    return load_model(path, compile = False)

@st.cache_resource(show_spinner=False)
def load_scaler(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler .pkl not found at {path}")
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_sarimax(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SARIMAX .pkl not found at {path}")
    return sm.load(path)

# Load local CSV file
@st.cache_data(show_spinner=False)
def load_flight_data_from_file(file_path):
    """Load flight data from local CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def process_dataframe(df):
    """Process the loaded dataframe"""
    if df is None or df.empty:
        return None
    
    # Convert date columns
    if 'searchDate' in df.columns:
        df['searchDate'] = pd.to_datetime(df['searchDate'], errors='coerce')
    if 'flightDate' in df.columns:
        df['flightDate'] = pd.to_datetime(df['flightDate'], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['baseFare', 'totalFare', 'totalTravelDistance', 'travelDuration', 
                   'elapsedDays', 'seatsRemaining', 'segmentsDurationInSeconds', 'segmentsDistance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean columns
    bool_cols = ['isBasicEconomy', 'isRefundable', 'isNonStop']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes'])
    
    return df

# -------------------------------------------
# Feature order (same as you trained the ANN)
# -------------------------------------------
FEATURE_ORDER = [
    "baseFare",
    "totalTravelDistance",
    "travelDuration",
    "elapsedDays",
    "seatsRemaining",
    "segmentsDistance",
    "segmentsDurationInSeconds",
    "isNonStop_bin",
    "isBasicEconomy_bin",
    "isRefundable_bin",
    "days_ahead",
    "flight_is_weekend",
]

# -------------------------
# Helpers
# -------------------------
def compute_days_ahead(search_date, flight_date):
    if search_date is None or flight_date is None:
        return None
    if isinstance(search_date, dt.date) and not isinstance(search_date, dt.datetime):
        search_date = dt.datetime.combine(search_date, dt.time())
    if isinstance(flight_date, dt.date) and not isinstance(flight_date, dt.datetime):
        flight_date = dt.datetime.combine(flight_date, dt.time())
    return max(0, (flight_date - search_date).days)

def weekend_flag(flight_date):
    if not flight_date:
        return 0
    if isinstance(flight_date, dt.datetime):
        d = flight_date.date()
    else:
        d = flight_date
    return 1 if d.weekday() >= 5 else 0

def batch_prepare_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a batch for ANN prediction."""
    if all(c in df.columns for c in FEATURE_ORDER):
        X = df[FEATURE_ORDER].copy()
    else:
        base_needed = {
            "baseFare","totalTravelDistance","travelDuration","elapsedDays",
            "seatsRemaining","segmentsDistance","segmentsDurationInSeconds",
            "isNonStop_bin","isBasicEconomy_bin","isRefundable_bin"
        }
        missing = base_needed - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required base columns: {missing}")

        # Derive features if dates present
        if "searchDate" in df.columns and "flightDate" in df.columns:
            sd = pd.to_datetime(df["searchDate"], errors="coerce")
            fd = pd.to_datetime(df["flightDate"], errors="coerce")
            days = (fd - sd).dt.days.clip(lower=0).fillna(0).astype(int)
            wk = fd.dt.weekday.fillna(0).astype(int).apply(lambda d: 1 if d >= 5 else 0)
            df["days_ahead"] = days
            df["flight_is_weekend"] = wk
        else:
            df["days_ahead"] = df.get("elapsedDays", 0).fillna(0).astype(int)
            df["flight_is_weekend"] = 0

        X = df[FEATURE_ORDER].copy()

    for c in FEATURE_ORDER:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X.astype(np.float32)

# -------------------------
# UI Tabs
# -------------------------
st.title("✈️ Flight Fare Intelligence")
tab_pred, tab_fc, tab_viz, tab_stats = st.tabs(
    ["🔮 Prediction", "📈 Forecasting", "📊 Visualizations", "📊 Stats"]
)

# ======================
# TAB 1: Prediction
# ======================
with tab_pred:
    st.subheader("🔮 High-Fare Probability (ANN, .h5)")
    defaults = {
        "baseFare": 150.0,
        "totalTravelDistance": 800.0,
        "travelDuration": 120.0,
        "elapsedDays": 14,
        "seatsRemaining": 5,
        "segmentsDistance": 800.0,
        "segmentsDurationInSeconds": 7200,
        "isNonStop_bin": True,
        "isBasicEconomy_bin": False,
        "isRefundable_bin": False,
        "searchDate": dt.date.today(),
        "flightDate": dt.date.today() + dt.timedelta(days=14),
    }
    for k,v in defaults.items():
        st.session_state.setdefault(k, v)

    colL, colR = st.columns([1,1])
    with colL:
        st.markdown("**Inputs** (same features used for training)")
        baseFare = st.number_input("baseFare", min_value=0.0, value=st.session_state["baseFare"], step=10.0, key="baseFare")
        totalTravelDistance = st.number_input("totalTravelDistance (miles)", min_value=0.0, value=st.session_state["totalTravelDistance"], step=10.0, key="totalTravelDistance")
        travelDuration = st.number_input("travelDuration (minutes)", min_value=0.0, value=st.session_state["travelDuration"], step=5.0, key="travelDuration")
        elapsedDays = st.number_input("elapsedDays (from data)", min_value=0, value=st.session_state["elapsedDays"], step=1, key="elapsedDays")
        seatsRemaining = st.number_input("seatsRemaining", min_value=0, value=st.session_state["seatsRemaining"], step=1, key="seatsRemaining")
        segmentsDistance = st.number_input("segmentsDistance (miles)", min_value=0.0, value=st.session_state["segmentsDistance"], step=10.0, key="segmentsDistance")
        segmentsDurationInSeconds = st.number_input("segmentsDurationInSeconds", min_value=0, value=st.session_state["segmentsDurationInSeconds"], step=60, key="segmentsDurationInSeconds")

        isNonStop_bin = st.checkbox("Non-stop", value=st.session_state["isNonStop_bin"], key="isNonStop_bin")
        isBasicEconomy_bin = st.checkbox("Basic Economy", value=st.session_state["isBasicEconomy_bin"], key="isBasicEconomy_bin")
        isRefundable_bin = st.checkbox("Refundable", value=st.session_state["isRefundable_bin"], key="isRefundable_bin")

        st.markdown("**Dates (for derived features)**")
        sd = st.date_input("searchDate", value=st.session_state["searchDate"], key="searchDate")
        fd = st.date_input("flightDate", value=st.session_state["flightDate"], key="flightDate")

        if st.button("🧪 Predict TestData", type="secondary"):
            try:
                # Load test data
                test_df = pd.read_csv("/workspaces/Vamsithesis/testdata.csv")
                
                # Process the test data
                test_df = process_dataframe(test_df)
                
                # Load models
                model = load_ann_model(ann_path)
                scaler = load_scaler(scaler_path)
                
                # Prepare features for prediction
                X_test = batch_prepare_from_df(test_df)
                X_test_scaled = scaler.transform(X_test)
                
                # Make predictions
                predictions = model.predict(X_test_scaled, verbose=0).ravel()
                
                # Create results dataframe
                results_df = test_df[['legId', 'totalFare', 'startingAirport', 'destinationAirport']].copy()
                results_df['predicted_high_fare_prob'] = predictions
                results_df['actual_fare'] = test_df['totalFare']
                
                st.success(f"✅ Predictions completed for {len(results_df)} test samples!")
                
                # Show detailed results
                st.markdown("### 📊 Detailed Results (First 20 rows)")
                display_cols = ['legId', 'startingAirport', 'destinationAirport', 'actual_fare', 
                               'predicted_high_fare_prob']
                st.dataframe(results_df[display_cols].head(20))
                
                # Download button
                st.download_button(
                    "📥 Download Full Results CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="testdata_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing test data: {str(e)}")
                st.info("Make sure testdata.csv exists in the correct path and has the required columns.")

    with colR:
        derived_days = compute_days_ahead(st.session_state["searchDate"], st.session_state["flightDate"])
        st.number_input("days_ahead", min_value=0, value=(derived_days or st.session_state["elapsedDays"]), step=1, key="days_ahead")
        st.selectbox("flight_is_weekend", options=[0,1],
                     index=weekend_flag(st.session_state["flightDate"]), key="flight_is_weekend")

        if st.button("🔎 Predict"):
            try:
                model = load_ann_model(ann_path)
                scaler = load_scaler(scaler_path)
                form_vals = {k: st.session_state[k] for k in [
                    "baseFare","totalTravelDistance","travelDuration","elapsedDays",
                    "seatsRemaining","segmentsDistance","segmentsDurationInSeconds",
                    "isNonStop_bin","isBasicEconomy_bin","isRefundable_bin",
                    "days_ahead","flight_is_weekend"
                ]}
                x = batch_prepare_from_df(pd.DataFrame([form_vals]))
                x_s = scaler.transform(x)
                prob = float(model.predict(x_s, verbose=0).ravel()[0])
                st.success(f"High-fare probability: **{prob:.3f}**")
            except Exception as e:
                st.error(str(e))


# ======================
# TAB 2: Forecasting
# ======================
with tab_fc:
    st.subheader("📈 Daily Fare Forecast (SARIMAX, .pkl)")
    st.markdown("""
    **How SARIMAX Forecasting Works:**
    
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) is a sophisticated time series forecasting model that:
    
    - **Captures Trends**: Identifies long-term upward or downward movements in fare prices
    - **Seasonal Patterns**: Detects recurring patterns (weekly, monthly cycles in flight pricing)
    - **Autocorrelation**: Uses historical fare values to predict future prices
    - **External Factors**: Can incorporate external variables like holidays, events, or market conditions
    
    **Model Components:**
    - **AR (AutoRegressive)**: Uses past fare values to predict future values
    - **I (Integrated)**: Accounts for trends by differencing the data
    - **MA (Moving Average)**: Incorporates past forecast errors to improve predictions
    - **Seasonal**: Handles repeating patterns in airline pricing
    """)
    
    if "fc_horizon" not in st.session_state:
        st.session_state["fc_horizon"] = 30

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### 🔧 Forecast Configuration")
        horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=90,
                                  value=st.session_state["fc_horizon"], step=1, key="fc_horizon")
        
        st.markdown("**📊 What You'll Get:**")
        st.markdown("""
        - Daily average fare predictions for the next N days
        - Confidence intervals showing prediction uncertainty
        - Interactive line chart for trend visualization
        - Downloadable CSV with all forecast values
        """)
        
        st.info("💡 **Tip**: Shorter horizons (7-14 days) typically provide more accurate forecasts than longer periods.")

    with c2:
        st.markdown("### 🚀 Generate Forecast")
        if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating forecast..."):
                    forecaster = load_sarimax(sarimax_path)
                    fc = forecaster.get_forecast(steps=int(st.session_state["fc_horizon"])).predicted_mean
                    fc = fc.rename("forecast").to_frame()
                    
                    st.success(f"✅ Generated {len(fc)} day forecast successfully!")
                    st.line_chart(fc, height=280, use_container_width=True)
                    
                    st.markdown("### 📋 Forecast Results (Last 10 Days)")
                    st.dataframe(fc.tail(10))
                    
                    st.download_button("📥 Download forecast CSV",
                                       data=fc.to_csv().encode("utf-8"),
                                       file_name="forecast.csv",
                                       mime="text/csv")
            except Exception as e:
                st.error(f"❌ Forecast Error: {str(e)}")
                st.markdown("**Possible solutions:**")
                st.markdown("- Check if the SARIMAX model file exists at the specified path")
                st.markdown("- Verify the model was trained and saved properly")
                st.markdown("- Ensure the forecast horizon is within reasonable limits")

# ======================
# TAB 3: Visualizations
# ======================
with tab_viz:
    st.subheader("📊 Interactive Flight Data Visualizations")
    
    # Load data from local file
    # Update this path to match your file location
    file_path = "/workspaces/Vamsithesis/itineraries_first_50000.csv"  # Change this to your actual file path
    
    # Load data automatically when tab is opened
    if 'viz_df' not in st.session_state:
        with st.spinner("Loading flight data..."):
            df = load_flight_data_from_file(file_path)
            if df is not None:
                df = process_dataframe(df)
                st.session_state['viz_df'] = df
                st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
            else:
                st.error(f"❌ Could not load file from {file_path}. Please check the file path.")
    
    df = st.session_state.get('viz_df', None)
    
    # Visualization section
    if df is not None and not df.empty:
        st.markdown("---")
        st.markdown("### 📈 Create Custom Visualizations")
        
        df_viz = df  # Use all 50k rows since it's already a manageable sample
        
        # Get column types
        numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_viz.select_dtypes(include=['object', 'bool']).columns.tolist()
        date_cols = df_viz.select_dtypes(include=['datetime64']).columns.tolist()
        all_cols = df_viz.columns.tolist()
        
        # Visualization controls
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
                 "Box Plot", "Heatmap", "3D Scatter", "Sunburst", "Pie Chart"]
            )
        
        with viz_col2:
            if chart_type == "Scatter Plot":
                x_col = st.selectbox("X-axis", numeric_cols, 
                                   index=numeric_cols.index('totalTravelDistance') if 'totalTravelDistance' in numeric_cols else 0)
                y_col = st.selectbox("Y-axis", numeric_cols,
                                   index=numeric_cols.index('totalFare') if 'totalFare' in numeric_cols else 0)
                color_col = st.selectbox("Color by", [None] + all_cols)
            
            elif chart_type == "Line Chart":
                x_col = st.selectbox("X-axis", date_cols + numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols)
                group_col = st.selectbox("Group by", [None] + categorical_cols)
            
            elif chart_type == "Bar Chart":
                x_col = st.selectbox("Category", categorical_cols + numeric_cols)
                y_col = st.selectbox("Value", numeric_cols)
                agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "median", "min", "max"])
            
            elif chart_type == "Histogram":
                hist_col = st.selectbox("Column", numeric_cols)
                bins = st.slider("Number of bins", 10, 100, 30)
            
            elif chart_type == "Box Plot":
                box_y = st.selectbox("Value", numeric_cols)
                box_x = st.selectbox("Group by", [None] + categorical_cols)
            
            elif chart_type == "Heatmap":
                available_numeric = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
                heatmap_cols = st.multiselect("Columns for correlation", available_numeric,
                                            default=available_numeric[:min(5, len(available_numeric))])
            
            elif chart_type == "3D Scatter":
                x_3d = st.selectbox("X-axis", numeric_cols)
                y_3d = st.selectbox("Y-axis", numeric_cols)
                z_3d = st.selectbox("Z-axis", numeric_cols)
                color_3d = st.selectbox("Color by", [None] + all_cols)
            
            elif chart_type == "Sunburst":
                path_cols = st.multiselect("Hierarchy levels", categorical_cols,
                                          default=categorical_cols[:min(2, len(categorical_cols))])
                value_col = st.selectbox("Value", [None] + numeric_cols)
            
            elif chart_type == "Pie Chart":
                pie_category = st.selectbox("Category", categorical_cols)
                pie_values = st.selectbox("Values", ["count"] + numeric_cols)
        
        with viz_col3:
            st.markdown("**Filters (optional)**")
            filter_col = st.selectbox("Filter by", [None] + all_cols)
            
            if filter_col and filter_col != None:
                if filter_col in numeric_cols:
                    min_val = float(df_viz[filter_col].min())
                    max_val = float(df_viz[filter_col].max())
                    filter_range = st.slider(f"{filter_col} range", min_val, max_val, (min_val, max_val))
                    df_viz = df_viz[(df_viz[filter_col] >= filter_range[0]) & 
                                   (df_viz[filter_col] <= filter_range[1])]
                elif filter_col in categorical_cols:
                    unique_vals = df_viz[filter_col].unique()
                    selected_vals = st.multiselect(f"{filter_col} values", unique_vals, 
                                                 default=list(unique_vals[:min(5, len(unique_vals))]))
                    if selected_vals:
                        df_viz = df_viz[df_viz[filter_col].isin(selected_vals)]
        
        # Generate visualization
        if st.button("📊 Generate Visualization", type="primary", use_container_width=True):
            try:
                fig = None
                
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df_viz, x=x_col, y=y_col, color=color_col,
                                   title=f"{y_col} vs {x_col}")
                
                elif chart_type == "Line Chart":
                    if group_col:
                        df_grouped = df_viz.groupby([x_col, group_col]).mean().reset_index()
                        fig = px.line(df_grouped, x=x_col, y=y_col, color=group_col,
                                    title=f"{y_col} over {x_col}")
                    else:
                        fig = px.line(df_viz.sort_values(x_col), x=x_col, y=y_col,
                                    title=f"{y_col} over {x_col}")
                
                elif chart_type == "Bar Chart":
                    if x_col in categorical_cols:
                        agg_data = df_viz.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        agg_data = agg_data.sort_values(y_col, ascending=False).head(20)
                    else:
                        df_viz['binned'] = pd.cut(df_viz[x_col], bins=20)
                        agg_data = df_viz.groupby('binned')[y_col].agg(agg_func).reset_index()
                        agg_data[x_col] = agg_data['binned'].astype(str)
                    
                    fig = px.bar(agg_data, x=x_col, y=y_col,
                               title=f"{agg_func} of {y_col} by {x_col}")
                
                elif chart_type == "Histogram":
                    fig = px.histogram(df_viz, x=hist_col, nbins=bins,
                                     title=f"Distribution of {hist_col}")
                
                elif chart_type == "Box Plot":
                    fig = px.box(df_viz, x=box_x, y=box_y,
                               title=f"Distribution of {box_y}" + (f" by {box_x}" if box_x else ""))
                
                elif chart_type == "Heatmap":
                    if heatmap_cols:
                        corr_matrix = df_viz[heatmap_cols].corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                      title="Correlation Heatmap",
                                      color_continuous_scale='RdBu_r')
                
                elif chart_type == "3D Scatter":
                    fig = px.scatter_3d(df_viz, x=x_3d, y=y_3d, z=z_3d, color=color_3d,
                                      title=f"3D: {x_3d} vs {y_3d} vs {z_3d}")
                
                elif chart_type == "Sunburst":
                    if path_cols:
                        fig = px.sunburst(df_viz, path=path_cols, values=value_col,
                                        title="Hierarchical View")
                
                elif chart_type == "Pie Chart":
                    if pie_values == "count":
                        pie_data = df_viz[pie_category].value_counts().head(10).reset_index()
                        pie_data.columns = [pie_category, 'count']
                        fig = px.pie(pie_data, names=pie_category, values='count',
                                   title=f"Distribution of {pie_category}")
                    else:
                        pie_data = df_viz.groupby(pie_category)[pie_values].sum().reset_index()
                        pie_data = pie_data.nlargest(10, pie_values)
                        fig = px.pie(pie_data, names=pie_category, values=pie_values,
                                   title=f"{pie_values} by {pie_category}")
                
                if fig:
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select appropriate columns for the visualization.")
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        
        # Quick insights
        st.markdown("---")
        st.markdown("### 🎯 Quick Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Fare Trends"):
                if 'totalFare' in numeric_cols and 'elapsedDays' in numeric_cols:
                    trend_data = df_viz.groupby('elapsedDays')['totalFare'].mean().reset_index()
                    fig = px.line(trend_data, x='elapsedDays', y='totalFare',
                                title="Average Fare by Days in Advance")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("✈️ Top Routes"):
                if 'startingAirport' in df_viz.columns and 'destinationAirport' in df_viz.columns:
                    routes = df_viz.groupby(['startingAirport', 'destinationAirport']).size().reset_index(name='count')
                    routes = routes.nlargest(10, 'count')
                    routes['route'] = routes['startingAirport'] + ' → ' + routes['destinationAirport']
                    fig = px.bar(routes, x='count', y='route', orientation='h',
                               title="Top 10 Most Frequent Routes")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if st.button("💰 Price Distribution"):
                if 'totalFare' in numeric_cols:
                    fig = px.histogram(df_viz, x='totalFare', nbins=50,
                                     title="Fare Distribution")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        with st.expander("📋 Preview Data"):
            st.dataframe(df_viz.head(100))
            st.caption(f"Showing first 100 rows of {len(df_viz):,} rows being visualized")
            
            # Basic statistics
            if st.checkbox("Show statistics"):
                st.write(df_viz.describe())
    
    else:
        st.warning("⚠️ No data loaded. Please check the file path in the code.")


# ======================
# TAB 4: Stats
# ======================
with tab_stats:
    st.subheader("📊 Flight Data Statistics")
    st.caption("Comprehensive statistical analysis of the flight dataset")
    
    # Daily Fare Trends
    st.markdown("### 📅 Daily Average Fare Trends")
    st.markdown("**Definition**: This table shows the average total fare and number of flights for each date, helping identify fare patterns over time.")
    
    daily_trends_data = {
        'flightDate': ['2022-04-17', '2022-04-18', '2022-04-19', '2022-04-20', '2022-04-21', 
                       '2022-04-22', '2022-04-23', '2022-04-24', '2022-04-25', '2022-04-26'],
        'avg_total_fare': [429.56, 397.35, 333.40, 330.48, 365.46, 389.78, 406.47, 524.25, 387.86, 303.55],
        'n': [8258, 16524, 30702, 41078, 45888, 53384, 53475, 51930, 70168, 107712]
    }
    daily_df = pd.DataFrame(daily_trends_data)
    st.dataframe(daily_df, use_container_width=True)
    
    st.markdown("---")
    
    # Overall Statistics
    st.markdown("### 📈 Overall Fare Statistics")
    st.markdown("**Definition**: Key statistical measures of total fare across all flights including central tendency, spread, and volume metrics.")
    
    overall_stats = {
        'Metric': ['Average Total Fare', 'Minimum Total Fare', 'Maximum Total Fare', 'Standard Deviation', 'Total Count'],
        'Value': ['$340.39', '$19.59', '$8,260.61', '$196.03', '82,138,753']
    }
    overall_df = pd.DataFrame(overall_stats)
    st.dataframe(overall_df, use_container_width=True)
    
    st.markdown("---")
    
    # Distribution Metrics
    st.markdown("### 📊 Distribution Characteristics")
    st.markdown("**Definition**: Skewness and kurtosis measures that describe the shape and symmetry of fare and distance distributions.")
    
    distribution_data = {
        'Metric': ['Fare Skewness', 'Fare Kurtosis', 'Distance Skewness', 'Distance Kurtosis'],
        'Value': [2.102, 16.202, 0.337, -0.898],
        'Interpretation': ['Right-skewed (higher fares are outliers)', 'Heavy-tailed distribution', 'Slightly right-skewed', 'Flatter than normal distribution']
    }
    dist_df = pd.DataFrame(distribution_data)
    st.dataframe(dist_df, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation
    st.markdown("### 🔗 Fare-Distance Correlation")
    st.markdown("**Definition**: Pearson correlation coefficient measuring the linear relationship between fare and flight distance.")
    
    correlation_data = {
        'Relationship': ['Fare vs Distance Correlation'],
        'Coefficient': [0.493],
        'Strength': ['Moderate positive correlation']
    }
    corr_df = pd.DataFrame(correlation_data)
    st.dataframe(corr_df, use_container_width=True)
    
    st.markdown("---")
    
    # Top Fare Basis Codes
    st.markdown("### 🎫 Top Fare Basis Codes")
    st.markdown("**Definition**: Most frequently used fare basis codes which determine pricing rules, restrictions, and booking classes.")
    
    fare_codes_data = {
        'fareBasisCode': ['QAA0OKEN', 'KAUOA0MQ', 'V7AWZNN1', 'QAA0OFEN', 'HAA0OKEN', 
                         'G7AWZNN1', 'KAVOA0MQ', 'KA0NA0MC', 'V0AHZNN1', 'L0AIZNN1'],
        'Count': [1386883, 937072, 797408, 570964, 537443, 470363, 438196, 398620, 393533, 369229]
    }
    codes_df = pd.DataFrame(fare_codes_data)
    st.dataframe(codes_df, use_container_width=True)
    
    st.markdown("---")
    
    # Flight Characteristics
    st.markdown("### ✈️ Flight Characteristics Distribution")
    st.markdown("**Definition**: Percentage breakdown of key flight attributes that affect pricing and passenger experience.")
    
    characteristics_data = {
        'Characteristic': ['Non-stop Flights', 'Refundable Tickets', 'Basic Economy'],
        'Percentage': ['26.87%', '0.0016%', '14.40%'],
        'Description': ['Direct flights without connections', 'Tickets allowing full refunds', 'Lowest fare class with restrictions']
    }
    char_df = pd.DataFrame(characteristics_data)
    st.dataframe(char_df, use_container_width=True)
    
    st.markdown("---")
    
    # Booking Window Analysis
    st.markdown("### 📅 Booking Window Analysis")
    st.markdown("**Definition**: Analysis of fare patterns based on how far in advance tickets are booked.")
    
    booking_data = {
        'Booking Window': ['<1 week'],
        'Average Fare': ['$340.39'],
        'Count': ['82,138,753']
    }
    booking_df = pd.DataFrame(booking_data)
    st.dataframe(booking_df, use_container_width=True)
    
    st.markdown("---")
    
    # Quartile Analysis
    st.markdown("### 📐 Quartile Analysis & Outlier Detection")
    st.markdown("**Definition**: Statistical quartiles used to identify fare distribution and detect outliers using the IQR method.")
    
    quartile_data = {
        'Metric': ['Q1 (25th percentile)', 'Q3 (75th percentile)', 'IQR', 'Lower Bound', 'Upper Bound'],
        'Value': ['$197.10', '$452.09', '$254.99', '-$185.38', '$834.57']
    }
    quart_df = pd.DataFrame(quartile_data)
    st.dataframe(quart_df, use_container_width=True)
    
    # Outlier Summary
    outlier_data = {
        'Metric': ['Total Records', 'Outliers Count', 'Outlier Rate'],
        'Value': ['82,138,753', '1,191,764', '1.45%']
    }
    outlier_df = pd.DataFrame(outlier_data)
    st.dataframe(outlier_df, use_container_width=True)
    
    st.markdown("---")
    
    # Percentile Distribution
    st.markdown("### 📊 Fare Percentile Distribution")
    st.markdown("**Definition**: Fare values at key percentiles showing the distribution of prices across the dataset.")
    
    percentile_data = {
        'Percentile': ['10th', '25th', '50th (Median)', '75th', '90th', '95th'],
        'Fare': ['$133.60', '$197.10', '$305.20', '$452.09', '$578.60', '$665.10']
    }
    perc_df = pd.DataFrame(percentile_data)
    st.dataframe(perc_df, use_container_width=True)
    
    st.markdown("---")
    
    # Airline Statistics
    st.markdown("### 🛫 Airline Fare Statistics")
    st.markdown("**Definition**: Fare distribution percentiles by airline, showing pricing patterns and flight volumes for major carriers.")
    
    airline_data = {
        'Airline': ['AA||AA', 'DL||DL', 'UA||UA', 'AA', 'DL', 'UA', 'NK||NK', 'B6', 'B6||B6', 'AS||AS',
                   'DL||UA', 'UA||DL', 'UA||UA||DL', 'DL||UA||UA', 'AA||AA||AA', 'DL||DL||DL', 'NK', 'F9||F9', 'AS||UA', 'F9'],
        'P10': [151.6, 171.6, 147.6, 118.6, 108.6, 103.6, 93.58, 93.6, 150.61, 227.2,
               446.59, 437.21, 500.1, 517.11, 216.71, 237.69, 43.59, 89.59, 391.59, 47.98],
        'P25': [199.1, 247.6, 222.6, 163.6, 153.6, 151.6, 132.58, 138.6, 198.11, 307.2,
               477.21, 479.21, 539.1, 547.1, 288.7, 344.7, 67.59, 139.59, 452.2, 63.99],
        'Median': [287.6, 357.6, 321.6, 233.61, 228.6, 224.6, 193.58, 198.6, 276.61, 427.21,
                  531.6, 531.6, 574.1, 582.7, 404.6, 487.1, 117.59, 217.99, 525.6, 93.99],
        'P75': [391.09, 487.6, 431.6, 338.61, 328.6, 333.6, 270.58, 259.6, 390.62, 557.21,
               585.6, 591.2, 652.71, 661.7, 511.2, 616.69, 178.59, 344.0, 618.6, 145.99],
        'P90': [490.6, 611.6, 529.19, 435.1, 458.6, 448.6, 339.58, 368.6, 522.21, 697.19,
               641.6, 647.2, 772.7, 772.71, 1071.7, 853.1, 249.59, 514.0, 721.6, 223.99],
        'P95': [564.61, 712.1, 591.6, 528.6, 518.6, 538.6, 380.58, 464.6, 616.72, 787.19,
               681.6, 681.61, 837.09, 837.1, 1251.71, 1085.6, 290.59, 564.6, 781.6, 273.99],
        'Count': [16296340, 11351669, 10217320, 7453245, 4855169, 4531426, 4215514, 3578393, 3129646, 1777996,
                 1400226, 1371847, 1232572, 1101129, 1096382, 1034931, 735728, 694424, 491504, 466926]
    }
    airline_df = pd.DataFrame(airline_data)
    st.dataframe(airline_df, use_container_width=True, height=600)
    
    st.markdown("**Airline Code Legend:**")
    st.markdown("""
    - **AA**: American Airlines
    - **DL**: Delta Air Lines  
    - **UA**: United Airlines
    - **NK**: Spirit Airlines
    - **B6**: JetBlue Airways
    - **AS**: Alaska Airlines
    - **F9**: Frontier Airlines
    - **||**: Indicates connecting flights (e.g., AA||DL means American to Delta connection)
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Key Insights")
    st.info("""
    **Summary of Key Findings:**
    
    1. **Fare Distribution**: Average fare is $340.39 with high variability (std: $196.03)
    2. **Temporal Patterns**: Fares vary significantly by date, with lowest average on 2022-04-26 ($303.55)
    3. **Airline Pricing**: Budget carriers (NK, F9) show consistently lower fares across all percentiles
    4. **Route Complexity**: Multi-segment flights (indicated by ||) generally command higher fares
    5. **Market Concentration**: Top 3 airlines (AA, DL, UA) represent majority of flights
    6. **Outliers**: 1.45% of fares are statistical outliers, indicating premium pricing segments
    7. **Booking Patterns**: Most bookings occur within 1 week of departure
    """)
    