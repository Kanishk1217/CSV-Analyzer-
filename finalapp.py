x# -*- coding: utf-8 -*-
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
from sklearn.metrics import r2_score, accuracy_score

# ... (Previous code for Tab 1, 2, and 3 remains the same) ...

    ## --- TAB 4: MACHINE LEARNING ANALYSIS ---
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
