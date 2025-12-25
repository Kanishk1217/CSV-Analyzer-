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

# --- PAGE SETUP ---
st.set_page_config(page_title="CSV Data Analyzer", layout="wide")

st.title("CSV Data Analyzer")
st.write("Advanced Analytics & Predictive Modeling Interface")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- AUTOMATED CLEANING (From your Notebook) ---
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Statistical Analysis", "Visualizations", "Machine Learning"])

    ## --- TAB 2: STATISTICAL ANALYSIS (FIXED) ---
    with tab2:
        s1, s2, s3, s4 = st.tabs(["Summary Statistics", "Categorical Data", "Correlation", "Missing Values"])
        with s1:
            st.table(df.describe().T)
        with s2:
            if cat_cols:
                sel = st.selectbox("Distribution of:", cat_cols)
                st.plotly_chart(px.bar(df[sel].value_counts().head(10)), use_container_width=True)
        with s3:
            if len(num_cols) > 1:
                fig, ax = plt.subplots()
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
                st.pyplot(fig)
        with s4:
            st.write(df.isnull().sum())

    ## --- TAB 3: VISUALIZATIONS (FIXED SCATTER & BOX) ---
    with tab3:
        v1, v2, v3 = st.columns(3)
        with v1: x_axis = st.selectbox("X-Axis", df.columns)
        with v2: y_axis = st.selectbox("Y-Axis (Numeric)", num_cols)
        with v3: chart = st.selectbox("Chart Type", ["Scatter Plot", "Box Plot", "Pie Chart", "Bar Chart"])

        if chart == "Scatter Plot":
            st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis), use_container_width=True)
        elif chart == "Box Plot":
            st.plotly_chart(px.box(df, x=x_axis, y=y_axis), use_container_width=True)
        elif chart == "Pie Chart":
            st.plotly_chart(px.pie(df, names=x_axis), use_container_width=True)
        elif chart == "Bar Chart":
            st.plotly_chart(px.bar(df, x=x_axis, y=y_axis), use_container_width=True)

    ## --- TAB 4: MACHINE LEARNING (FIXED R2 & IMPORTANCE) ---
    with tab4:
        target = st.selectbox("Target Label", df.columns)
        is_num = pd.api.types.is_numeric_dtype(df[target])
        st.write(f"Task Detected: **{'Regression' if is_num else 'Classification'}**")

        if st.button("Train & Analyze"):
            X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
            y = LabelEncoder().fit_transform(df[target]) if not is_num else df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor() if is_num else RandomForestClassifier()
            model.fit(X_train, y_train)
            
            # Show Metrics
            if is_num:
                st.metric("R² Score", f"{r2_score(y_test, model.predict(X_test)):.4f}")
            else:
                st.metric("Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.2%}")

            # Feature Importance (From your Notebook logic)
            st.write("### Top 10 Feature Importance")
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
            st.plotly_chart(px.bar(feat_imp, orientation='h'))
