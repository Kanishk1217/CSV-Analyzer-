# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="CSV Data Analyzer", layout="wide")

# --- UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #2d3436; color: white; width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("CSV Data Analyzer")
st.write("Comprehensive data exploration and predictive modeling tool.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # --- AUTOMATED DATA CLEANING (Based on Notebook Logic) ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Filling missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Statistical Analysis", "Visualizations", "Machine Learning"])

    ## --- TAB 1: DATA PREVIEW ---
    with tab1:
        st.subheader("Data Preview")
        rows = st.selectbox("Rows per page:", [10, 20, 50, 100])
        st.dataframe(df.head(rows), use_container_width=True)

    ## --- TAB 2: STATISTICAL ANALYSIS ---
    with tab2:
        s1, s2, s3, s4 = st.tabs(["Summary Statistics", "Categorical Data", "Correlation", "Missing Values"])
        
        with s1:
            st.table(df.describe().T)
        
        with s2:
            if categorical_cols:
                sel_cat = st.selectbox("Distribution of:", categorical_cols)
                st.plotly_chart(px.bar(df[sel_cat].value_counts().head(10)), use_container_width=True)
        
        with s3:
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
                st.pyplot(fig) # Correctly displays Matplotlib charts in Streamlit
        
        with s4:
            st.write(df.isnull().sum())

    ## --- TAB 3: VISUALIZATIONS (Fixed Axes) ---
    with tab3:
        st.subheader("Interactive Visualizations")
        v1, v2, v3 = st.columns(3)
        with v1:
            x_ax = st.selectbox("X-Axis (Primary)", df.columns)
        with v2:
            y_ax = st.selectbox("Y-Axis (Numeric)", numeric_cols)
        with v3:
            chart = st.selectbox("Chart Type", ["Scatter Plot", "Box Plot", "Pie Chart", "Bar Chart"])

        # Implementing chart logic for Scatter and Box plots
        if chart == "Scatter Plot":
            st.plotly_chart(px.scatter(df, x=x_ax, y=y_ax, color_discrete_sequence=['#ef553b']), use_container_width=True)
        elif chart == "Box Plot":
            st.plotly_chart(px.box(df, x=x_ax, y=y_ax), use_container_width=True)
        elif chart == "Pie Chart":
            st.plotly_chart(px.pie(df, names=x_ax), use_container_width=True)
        elif chart == "Bar Chart":
            st.plotly_chart(px.bar(df, x=x_ax, y=y_ax), use_container_width=True)

    ## --- TAB 4: MACHINE LEARNING (Fixed R2/Accuracy) ---
    with tab4:
        st.subheader("Predictive Analysis")
        target = st.selectbox("Select Target Label", df.columns)
        
        # Detection of task type
        is_num = pd.api.types.is_numeric_dtype(df[target])
        task = "Regression" if is_num else "Classification"
        st.write(f"Detected Task: **{task}**")

        if st.button("Train Model"):
            X = df.drop(columns=[target])
            y = df[target]
            
            # Simple encoding for ML
            X_encoded = pd.get_dummies(X, drop_first=True)
            if task == "Classification" and not is_num:
                y = LabelEncoder().fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            if task == "Classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                st.metric("Model Accuracy", f"{acc:.2%}")
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
                r2 = r2_score(y_test, model.predict(X_test))
                st.metric("R² Score", f"{r2:.4f}")

            # Feature Importance Section
            st.write("### Top 10 Feature Importance")
            importances = pd.Series(model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(10)
            st.plotly_chart(px.bar(importances, orientation='h'), use_container_width=True)

else:
    st.info("Awaiting CSV file upload.")
