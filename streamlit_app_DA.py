import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from pandas.api.types import is_numeric_dtype
import pprint
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 20px 0 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("# 📊 Data Analysis Dashboard")
st.sidebar.markdown("---")

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Select a CSV file to begin analysis"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ File uploaded successfully!")
        
        # Main title
        st.markdown("<h1 class='main-title'>📊 Automated Data Analysis Dashboard</h1>", unsafe_allow_html=True)
        
        # ===== TAB NAVIGATION =====
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Overview",
            "📈 Visualization",
            "🔍 Correlation",
            "🧹 Data Quality",
            "🤖 ML Model",
            "📊 Feature Importance"
        ])
        
        # ===== TAB 1: OVERVIEW =====
        with tab1:
            st.markdown("<div class='section-header'><h3>Dataset Overview</h3></div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Rows", len(df))
            with col2:
                st.metric("📍 Total Columns", len(df.columns))
            with col3:
                st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=['int64', 'float64']).columns))
            with col4:
                st.metric("📝 Categorical Columns", len(df.select_dtypes(include=['object']).columns))
            
            st.markdown("---")
            
            # First few rows
            with st.expander("👀 View First 10 Rows", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            st.markdown("<div class='section-header'><h3>Column Information</h3></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                st.dataframe(df.dtypes, use_container_width=True)
            
            with col2:
                st.write("**Descriptive Statistics:**")
                st.dataframe(df.describe(), use_container_width=True)
        
        # ===== TAB 2: VISUALIZATION =====
        with tab2:
            st.markdown("<div class='section-header'><h3>Data Visualizations</h3></div>", unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Numeric distributions
            if numeric_cols:
                st.subheader("📊 Numeric Column Distributions")
                selected_numeric = st.multiselect(
                    "Select numeric columns to visualize:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
                )
                
                for col in selected_numeric:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.histplot(df[col], kde=True, bins=20, ax=ax)
                    ax.set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
                    st.pyplot(fig)
            
            # Categorical frequencies
            if categorical_cols:
                st.subheader("📊 Categorical Column Frequencies")
                selected_categorical = st.multiselect(
                    "Select categorical columns to visualize:",
                    categorical_cols,
                    default=categorical_cols[:3] if len(categorical_cols) > 3 else categorical_cols
                )
                
                for col in selected_categorical:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df[col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f"Frequency of {col}", fontsize=14, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        # ===== TAB 3: CORRELATION =====
        with tab3:
            st.markdown("<div class='section-header'><h3>Correlation Analysis</h3></div>", unsafe_allow_html=True)
            
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                correlation_matrix = numeric_df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax, cbar_kws={'label': 'Correlation'})
                ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, fontweight='bold')
                st.pyplot(fig)
                
                # Correlation insights
                st.markdown("**Top Positive Correlations:**")
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': correlation_matrix.columns[i],
                            'Feature 2': correlation_matrix.columns[j],
                            'Correlation': correlation_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
                st.dataframe(corr_df.head(10), use_container_width=True)
            else:
                st.warning("No numeric columns found for correlation analysis.")
        
        # ===== TAB 4: DATA QUALITY =====
        with tab4:
            st.markdown("<div class='section-header'><h3>Data Quality Assessment</h3></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Missing Values:**")
                missing = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Count': missing.values,
                    'Missing %': (missing.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            with col2:
                st.write("**Duplicates:**")
                duplicates = df.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
            
            # Data cleaning summary
            st.markdown("---")
            st.subheader("🧹 Data Cleaning Status")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                df_clean = df.copy()
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
                for col in categorical_cols:
                    if df_clean[col].isnull().sum() > 0:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown')
                
                st.success("✅ Data cleaned and ready for modeling!")
                st.info(f"**Cleaning applied:** Numeric columns filled with mean, Categorical columns filled with mode")
        
        # ===== TAB 5: ML MODEL =====
        with tab5:
            st.markdown("<div class='section-header'><h3>Machine Learning Model</h3></div>", unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Target column selection
            all_cols = df.columns.tolist()
            target_column = st.selectbox(
                "🎯 Select Target Column for Prediction:",
                all_cols,
                help="Choose the column you want to predict"
            )
            
            if target_column:
                # Determine problem type
                if is_numeric_dtype(df[target_column]):
                    problem_type = "Regression"
                else:
                    problem_type = "Classification"
                
                st.info(f"**Problem Type:** {problem_type}")
                
                # Data preprocessing
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                X_encoded = pd.get_dummies(X, drop_first=True)
                
                if problem_type == "Classification":
                    if y.dtype == 'object':
                        y = LabelEncoder().fit_transform(y)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.3, random_state=101
                )
                
                # Model training
                st.subheader("🤖 Model Training")
                progress_bar = st.progress(0)
                
                if problem_type == "Classification":
                    model = RandomForestClassifier(random_state=101, n_estimators=100)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    progress_bar.progress(100)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("📊 Train Size", len(X_train))
                    with col3:
                        st.metric("📊 Test Size", len(X_test))
                    
                else:  # Regression
                    model = RandomForestRegressor(random_state=101, n_estimators=100)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    progress_bar.progress(100)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("📈 RMSE", f"{rmse:.4f}")
                    with col3:
                        st.metric("📊 Train Size", len(X_train))
                    with col4:
                        st.metric("📊 Test Size", len(X_test))
                
                st.success("✅ Model trained successfully!")
        
        # ===== TAB 6: FEATURE IMPORTANCE =====
        with tab6:
            st.markdown("<div class='section-header'><h3>Feature Importance Analysis</h3></div>", unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            all_cols = df.columns.tolist()
            
            target_column = st.selectbox(
                "🎯 Select Target Column:",
                all_cols,
                key="feature_importance_target"
            )
            
            if target_column:
                # Determine problem type
                if is_numeric_dtype(df[target_column]):
                    problem_type = "Regression"
                else:
                    problem_type = "Classification"
                
                # Data preprocessing
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                X_encoded = pd.get_dummies(X, drop_first=True)
                
                if problem_type == "Classification":
                    if y.dtype == 'object':
                        y = LabelEncoder().fit_transform(y)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.3, random_state=101
                )
                
                # Model training
                if problem_type == "Classification":
                    model = RandomForestClassifier(random_state=101, n_estimators=100)
                else:
                    model = RandomForestRegressor(random_state=101, n_estimators=100)
                
                model.fit(X_train, y_train)
                
                # Feature importance
                importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
                importances = importances.sort_values(ascending=False)
                
                st.subheader("🌟 Top 15 Most Important Features")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_importances = importances.head(15)
                top_importances.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_ylabel('Features', fontsize=12)
                ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                
                # Feature importance table
                st.subheader("📋 Feature Importance Table")
                importance_df = pd.DataFrame({
                    'Feature': importances.index,
                    'Importance': importances.values,
                    'Importance %': (importances.values * 100).round(2)
                }).reset_index(drop=True)
                
                st.dataframe(importance_df, use_container_width=True)
        
        # ===== FOOTER =====
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #888; padding: 20px;'>"
            "<p>📊 Automated Data Analysis Dashboard | Built with Streamlit</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted.")

else:
    # Initial landing page
    st.markdown(
        """
        <div style='text-align: center; padding: 50px;'>
            <h1>📊 Data Analysis Dashboard</h1>
            <p style='font-size: 18px; color: #666;'>
                Upload a CSV file to get started with automated data analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>📋 Overview</h3>
                <p>Get comprehensive statistics and data summaries</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>📈 Visualizations</h3>
                <p>Interactive charts and distributions</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>🤖 ML Models</h3>
                <p>Automated model training & evaluation</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown(
        """
        ## Features:
        
        ✅ **Data Overview** - View statistics, data types, and quality metrics  
        ✅ **Visualizations** - Interactive plots for numeric and categorical data  
        ✅ **Correlation Analysis** - Heatmaps and correlation insights  
        ✅ **Data Quality** - Missing values, duplicates, and cleaning status  
        ✅ **ML Modeling** - Automatic classification and regression models  
        ✅ **Feature Importance** - Identify the most influential features  
        
        ## How to Use:
        
        1. Click "Browse files" in the sidebar
        2. Select your CSV file
        3. Explore different tabs for various analyses
        4. Select target column for ML model training
        
        """
    )
