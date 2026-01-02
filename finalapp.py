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
        
        # Mode Selection: Manual choice between Classification and Regression
        ml_mode = st.radio("Mode", ["Classification", "Regression"], horizontal=True)
        
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            # Model selection dropdown
            model_type = st.selectbox("Model:", ["Random Forest", "Decision Tree"])
        with c2:
            # Target column selector
            target_col = st.selectbox("Target Column:", df.columns)
        with c3:
            st.write("##") # Spacer
            train_btn = st.button(f"Train {model_type}")

        if train_btn:
            with st.spinner(f"Training {model_type} model..."):
                # 1. Simple Preprocessing: Drop target and handle features
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # 2. Handle categorical strings for ML features
                X = pd.get_dummies(X)
                
                # 3. Handle target encoding if it's categorical
                if ml_mode == "Classification" and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # 4. Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # 5. Model Initialization based on Mode and Type
                if ml_mode == "Classification":
                    if model_type == "Random Forest":
                        model = RandomForestClassifier()
                    else:
                        model = DecisionTreeClassifier()
                else:
                    if model_type == "Random Forest":
                        model = RandomForestRegressor()
                    else:
                        model = DecisionTreeRegressor()
                
                # 6. Fit and Score
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                # 7. Display Metrics
                if ml_mode == "Classification":
                    st.success(f"Model trained! Accuracy Score: {score:.2%}")
                else:
                    st.success(f"Model trained! R² Score: {score:.4f}")

                # 8. Feature Importance Visualization
                if hasattr(model, 'feature_importances_'):
                    st.write("### Top Predictive Features")
                    feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
                    fig_imp = px.bar(feat_imp, orientation='h', labels={'value': 'Importance', 'index': 'Feature'})
                    st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin your analysis.")
