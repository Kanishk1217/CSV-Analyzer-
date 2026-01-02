# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="CSV Data Analyzer", layout="wide")

st.title("CSV Data Analyzer")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Automated Data Cleaning
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Defining the main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Statistical Analysis", "Visualizations", "Machine Learning"])

    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

    with tab2:
        s1, s2, s3, s4 = st.tabs(["Summary Statistics", "Categorical Data", "Correlation", "Missing Values"])
        with s1: 
            st.table(df.describe().T)
        with s2:
            if categorical_cols:
                sel = st.selectbox("Categorical Column:", categorical_cols, key="cat_stat_sel")
                st.plotly_chart(px.bar(df[sel].value_counts().head(10)), use_container_width=True)
        with s3:
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        with s4: 
            st.write(df.isnull().sum())

    with tab3:
        st.subheader("Interactive Visualizations")
        v1, v2, v3 = st.columns(3)
        with v1: 
            x_ax = st.selectbox("X-Axis", df.columns, key="viz_x")
        with v2: 
            y_ax = st.selectbox("Y-Axis (Numeric)", numeric_cols, key="viz_y")
        with v3: 
            chart = st.selectbox("Chart Type", ["Scatter Plot", "Box Plot", "Pie Chart", "Bar Chart"], key="viz_type")
        
        if chart == "Scatter Plot":
            st.plotly_chart(px.scatter(df, x=x_ax, y=y_ax), use_container_width=True)
        elif chart == "Box Plot":
            st.plotly_chart(px.box(df, x=x_ax, y=y_ax), use_container_width=True)
        else:
            st.plotly_chart(px.pie(df, names=x_ax), use_container_width=True)

    with tab4:
        st.subheader("Machine Learning Analysis")
        # Manual Mode selection
        ml_mode = st.radio("Mode", ["Classification", "Regression"], horizontal=True)
        
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            model_type = st.selectbox("Model:", ["Random Forest", "Decision Tree"], key="ml_model")
        with c2:
            target_col = st.selectbox("Target Column:", df.columns, key="ml_target")
        with c3:
            st.write("##") 
            train_btn = st.button(f"Train {model_type}")

        if train_btn:
            with st.spinner(f"Training {model_type}..."):
                X = df.drop(columns=[target_col])
                y = df[target_col]
                X = pd.get_dummies(X)
                
                if ml_mode == "Classification" and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if ml_mode == "Classification":
                    model = RandomForestClassifier() if model_type == "Random Forest" else DecisionTreeClassifier()
                else:
                    model = RandomForestRegressor() if model_type == "Random Forest" else DecisionTreeRegressor()
                
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if ml_mode == "Classification":
                    st.success(f"Model trained! Accuracy: {score:.2%}")
                else:
                    st.success(f"Model trained! R² Score: {score:.4f}")

                if hasattr(model, 'feature_importances_'):
                    st.write("### Top Predictive Features")
                    feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
                    st.plotly_chart(px.bar(feat_imp, orientation='h'), use_container_width=True)
else:
    st.info("Please upload a CSV file to begin your analysis.")
