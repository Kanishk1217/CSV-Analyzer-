import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="CSV Data Analyzer", layout="wide")

# Custom CSS to match the clean UI in images
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2d3436;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

## --- HEADER ---
st.title("CSV Data Analyzer")
st.write("Upload and analyze any CSV file")

## --- FILE UPLOAD (Image 1 & 3) ---
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], help="Drag and drop your CSV file here")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # File Info Summary (Image 3)
    st.info(f"**File: {uploaded_file.name}** \n{df.shape[0]} rows × {df.shape[1]} columns")

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Statistical Analysis", "Visualizations", "Machine Learning"])

    ## --- TAB 1: DATA PREVIEW (Image 3) ---
    with tab1:
        st.subheader("Data Preview")
        rows_per_page = st.selectbox("Rows per page:", [10, 20, 50, 100], index=0)
        st.dataframe(df.head(rows_per_page), use_container_width=True)

    ## --- TAB 2: STATISTICAL ANALYSIS (Image 6) ---
    with tab2:
        st.subheader("Statistical Analysis")
        stat_subtabs = st.tabs(["Summary Statistics", "Categorical Data", "Correlation", "Missing Values"])
        
        with stat_subtabs[0]:
            stats_list = []
            for col in df.columns:
                col_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
                count = df[col].count()
                missing = df[col].isnull().sum()
                
                if col_type == "numeric":
                    details = f"Mean: {df[col].mean():.2f} | Min: {df[col].min()} | Max: {df[col].max()}"
                else:
                    details = f"Unique values: {df[col].nunique()}"
                
                stats_list.append({
                    "Column": col, "Type": col_type, "Count": count, "Missing": missing, "Details": details
                })
            st.table(pd.DataFrame(stats_list))

    ## --- TAB 3: VISUALIZATIONS (Image 2 & 5) ---
    with tab3:
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            viz_column = st.selectbox("Column", df.columns, key="viz_col")
        with col2:
            chart_type = st.selectbox("Chart Type", ["Pie Chart", "Box Plot", "Scatter Plot", "Bar Chart"])

        if chart_type == "Pie Chart":
            # Image 2 style
            fig = px.pie(df, names=viz_column, hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Box Plot":
            # Image 5 style: Box Plot Analysis
            y_axis = st.selectbox("Select Y-Axis (Numeric)", df.select_dtypes(include=['number']).columns)
            x_axis = st.selectbox("Select X-Axis (Categorical)", df.columns)
            fig = px.box(df, x=x_axis, y=y_axis, points="all", color_discrete_sequence=['#87CEEB'])
            st.plotly_chart(fig, use_container_width=True)

    ## --- TAB 4: MACHINE LEARNING (Image 4) ---
    with tab4:
        st.subheader("Machine Learning Analysis")
        ml_mode = st.radio("Mode", ["Classification", "Regression"], horizontal=True)
        
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            model_type = st.selectbox("Model:", ["Random Forest", "Decision Tree"])
        with c2:
            target_col = st.selectbox("Target Column:", df.columns)
        with c3:
            st.write("##") # Spacer
            train_btn = st.button(f"Train {model_type}")

        if train_btn:
            with st.spinner("Training model..."):
                # Simple Preprocessing
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Handle categorical strings for ML
                X = pd.get_dummies(X)
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                if ml_mode == "Classification":
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()
                
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.success(f"Model trained! Accuracy/R² Score: {score:.2f}")

else:
    st.info("Please upload a CSV file to begin your analysis.")