import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pandas.api.types import is_numeric_dtype
import pprint
import warnings

warnings.filterwarnings('ignore')

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Data Analysis Baseline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM STYLING =====
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
        font-size: 2.5em;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 5px;
        margin: 20px 0 15px 0;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== SIDEBAR CONFIGURATION =====
st.sidebar.markdown("# 📊 Data Analysis Dashboard")
st.sidebar.markdown("Based on baseline.py analysis pipeline")
st.sidebar.markdown("---")

# ===== FILE UPLOAD =====
uploaded_file = st.sidebar.file_uploader(
    "📤 Upload your CSV file",
    type=['csv'],
    help="Select a CSV file to perform complete baseline analysis"
)

if uploaded_file is not None:
    try:
        # ===== LOAD DATA =====
        df = pd.read_csv(uploaded_file)
        df_original = df.copy()
        
        st.success("✅ File uploaded successfully!")
        
        st.markdown("<h1 class='main-header'>📊 Data Analysis Dashboard</h1>", unsafe_allow_html=True)
        
        # ===== CREATE TABS =====
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Exploration",
            "🔗 Correlation",
            "📊 Distributions",
            "📈 Visualizations",
            "🧹 Data Quality",
            "🎯 Preprocessing",
            "🤖 Model",
            "🌟 Features"
        ])
        
        # ===== TAB 1: BASIC DATA EXPLORATION =====
        with tab1:
            st.markdown("<div class='section-header'>📋 Basic Data Exploration</div>", unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Rows", len(df))
            with col2:
                st.metric("📍 Columns", len(df.columns))
            with col3:
                st.metric("🔢 Numeric", len(df.select_dtypes(include=['int64', 'float64']).columns))
            with col4:
                st.metric("📝 Categorical", len(df.select_dtypes(include=['object']).columns))
            
            st.markdown("---")
            
            # First few rows
            st.subheader("👀 First 5 Rows:")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("---")
            
            # Column Information
            st.subheader("🔍 Column Information:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=['Data Type']), use_container_width=True)
            
            with col2:
                st.write("**Memory Usage:**")
                memory_df = pd.DataFrame({
                    'Column': df.columns,
                    'Size (MB)': df.memory_usage(deep=True) / 1024 / 1024
                })
                st.dataframe(memory_df, use_container_width=True)
            
            st.markdown("---")
            
            # Descriptive Statistics
            st.subheader("📊 Descriptive Statistics:")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown("---")
            
            # DataFrame Info
            st.subheader("📋 DataFrame Info:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Shape:**", df.shape)
                st.write("**Total Cells:**", df.shape[0] * df.shape[1])
                st.write("**Duplicates:**", df.duplicated().sum())
            
            with col2:
                st.write("**Numeric Columns:**", len(df.select_dtypes(include=['int64', 'float64']).columns))
                st.write("**Categorical Columns:**", len(df.select_dtypes(include=['object']).columns))
                st.write("**Boolean Columns:**", len(df.select_dtypes(include=['bool']).columns))
        
        # ===== TAB 2: CORRELATION HEATMAP =====
        with tab2:
            st.markdown("<div class='section-header'>🔗 Correlation Analysis</div>", unsafe_allow_html=True)
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Before Cleaning:")
                    coorelation_heatmap = df_original.select_dtypes(include=['int64', 'float64'])
                    if not coorelation_heatmap.empty:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.heatmap(coorelation_heatmap.corr(), annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
                        ax.set_title("Correlation Matrix (Original Data)")
                        st.pyplot(fig)
                    else:
                        st.warning("No numeric columns found")
                
                with col2:
                    st.subheader("After Cleaning:")
                    coorelation_heatmap = df.select_dtypes(include=['int64', 'float64'])
                    if not coorelation_heatmap.empty:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.heatmap(coorelation_heatmap.corr(), annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
                        ax.set_title("Correlation Matrix (After Cleaning)")
                        st.pyplot(fig)
                    else:
                        st.warning("No numeric columns found")
                
                # Correlation insights
                st.markdown("---")
                st.subheader("📊 Correlation Insights:")
                
                correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': correlation_matrix.columns[i],
                            'Feature 2': correlation_matrix.columns[j],
                            'Correlation': correlation_matrix.iloc[i, j]
                        })
                
                if corr_pairs:
                    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
                    st.dataframe(corr_df.head(10), use_container_width=True)
            else:
                st.warning("No numeric columns found for correlation analysis")
        
        # ===== TAB 3: DISTRIBUTIONS =====
        with tab3:
            st.markdown("<div class='section-header'>📊 Distribution Analysis</div>", unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numeric_cols:
                st.subheader("📈 Numeric Column Distributions:")
                
                # Option to select columns
                selected_cols = st.multiselect(
                    "Select numeric columns to visualize:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
                )
                
                for col in selected_cols:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    sns.histplot(df[col], kde=True, bins=20, ax=ax)
                    ax.set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found")
        
        # ===== TAB 4: VISUALIZATIONS =====
        with tab4:
            st.markdown("<div class='section-header'>📈 Data Visualizations</div>", unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Frequency plots for categorical columns
            if categorical_cols:
                st.subheader("📊 Categorical Column Frequencies:")
                
                selected_cat_cols = st.multiselect(
                    "Select categorical columns:",
                    categorical_cols,
                    default=categorical_cols[:3] if len(categorical_cols) > 3 else categorical_cols
                )
                
                for col in selected_cat_cols:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df[col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f"Frequency of {col}", fontsize=14, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Box plots
            if numeric_cols and categorical_cols:
                st.markdown("---")
                st.subheader("📦 Box Plots (Numeric vs Categorical):")
                
                show_boxplots = st.checkbox("Generate box plots (may take time with large datasets)", value=False)
                
                if show_boxplots:
                    selected_numeric = st.multiselect(
                        "Select numeric columns for box plots:",
                        numeric_cols,
                        default=numeric_cols[:2] if len(numeric_cols) > 2 else numeric_cols
                    )
                    
                    selected_categorical = st.multiselect(
                        "Select categorical columns for box plots:",
                        categorical_cols,
                        default=categorical_cols[:2] if len(categorical_cols) > 2 else categorical_cols
                    )
                    
                    for cat_col in selected_categorical:
                        for num_col in selected_numeric:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
                            ax.set_title(f"{num_col} vs {cat_col}", fontsize=14, fontweight='bold')
                            st.pyplot(fig)
        
        # ===== TAB 5: DATA QUALITY =====
        with tab5:
            st.markdown("<div class='section-header'>🧹 Data Quality Assessment</div>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔴 Total Missing Values", df.isnull().sum().sum())
            with col2:
                st.metric("📋 Duplicate Rows", df.duplicated().sum())
            with col3:
                st.metric("✅ Complete Rows", len(df) - df.isnull().any(axis=1).sum())
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 Missing Values per Column:")
                missing = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Count': missing.values,
                    'Missing %': (missing.values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ No missing values found!")
            
            with col2:
                st.subheader("📊 Data Type Distribution:")
                dtype_counts = df.dtypes.value_counts()
                fig, ax = plt.subplots()
                dtype_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title("Data Type Distribution")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Column summaries
            st.subheader("📋 Numeric Columns Summary:")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found")
            
            st.markdown("---")
            
            st.subheader("📋 Categorical Columns Summary:")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                for col in categorical_cols:
                    with st.expander(f"**{col}** - {df[col].nunique()} unique values"):
                        st.dataframe(df[col].value_counts().head(10), use_container_width=True)
            else:
                st.info("No categorical columns found")
        
        # ===== TAB 6: PREPROCESSING =====
        with tab6:
            st.markdown("<div class='section-header'>🔧 Data Preprocessing</div>", unsafe_allow_html=True)
            
            # Get column info
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            st.subheader("🧹 Data Cleaning:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Cleaning:**")
                st.metric("Missing Values", df_original.isnull().sum().sum())
            
            with col2:
                st.write("**After Cleaning:**")
                
                # Create cleaned dataframe
                df_clean = df.copy()
                
                # Fill numeric columns with mean
                if numeric_cols:
                    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
                
                # Fill categorical columns with mode
                for col in categorical_cols:
                    if df_clean[col].isnull().sum() > 0:
                        mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                        df_clean[col] = df_clean[col].fillna(mode_val)
                
                st.metric("Missing Values", df_clean.isnull().sum().sum())
            
            st.success("✅ Data cleaned successfully!")
            
            st.markdown("---")
            
            st.subheader("📊 Cleaning Strategy Applied:")
            st.info(
                """
                • **Numeric Columns**: Filled with mean value
                • **Categorical Columns**: Filled with mode (most frequent value)
                • **Result**: All missing values handled
                """
            )
            
            # Update main dataframe
            df = df_clean
            
            st.markdown("---")
            
            st.subheader("✅ Cleaned Data Preview:")
            st.dataframe(df.head(), use_container_width=True)
        
        # ===== TAB 7: MODEL TRAINING =====
        with tab7:
            st.markdown("<div class='section-header'>🤖 Model Training & Evaluation</div>", unsafe_allow_html=True)
            
            # Ensure data is clean
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
            
            # Target column selection
            all_cols = df.columns.tolist()
            target_column = st.selectbox(
                "🎯 Select Target Column:",
                all_cols,
                help="Choose the column you want to predict"
            )
            
            if target_column:
                st.markdown("---")
                
                # Determine problem type
                if is_numeric_dtype(df[target_column]):
                    problem_type = "regression"
                else:
                    problem_type = "classification"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Target Column", target_column)
                with col2:
                    st.metric("📊 Problem Type", problem_type.capitalize())
                with col3:
                    st.metric("📍 Unique Values", df[target_column].nunique())
                
                st.markdown("---")
                
                # Data preprocessing
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                X_encoded = pd.get_dummies(X, drop_first=True)
                
                if problem_type == "classification":
                    if y.dtype == 'object':
                        y = LabelEncoder().fit_transform(y)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.3, random_state=101
                )
                
                st.subheader("📊 Train-Test Split:")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔵 Train Set", len(X_train))
                with col2:
                    st.metric("🔴 Test Set", len(X_test))
                with col3:
                    st.metric("📈 Train %", f"{len(X_train) / len(X_encoded) * 100:.1f}%")
                with col4:
                    st.metric("📊 Test %", f"{len(X_test) / len(X_encoded) * 100:.1f}%")
                
                st.markdown("---")
                
                # Model training
                st.subheader("🤖 Model Training:")
                
                with st.spinner("Training model..."):
                    if problem_type == "classification":
                        model = RandomForestClassifier(random_state=101)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🎯 Accuracy Score", f"{accuracy:.4f}")
                        with col2:
                            st.metric("📊 Correct Predictions", f"{int(accuracy * len(y_test))}/{len(y_test)}")
                        
                        st.success("✅ Classification model trained successfully!")
                    
                    else:  # Regression
                        model = RandomForestRegressor(random_state=101)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 R² Score", f"{r2:.4f}")
                        with col2:
                            st.metric("📈 RMSE", f"{rmse:.4f}")
                        with col3:
                            st.metric("🔴 MSE", f"{mse:.4f}")
                        
                        st.success("✅ Regression model trained successfully!")
                
                st.markdown("---")
                
                st.subheader("📋 Model Details:")
                st.info(
                    f"""
                    **Model Type**: Random Forest
                    **Problem Type**: {problem_type.capitalize()}
                    **Features**: {X_encoded.shape[1]}
                    **Train Samples**: {len(X_train)}
                    **Test Samples**: {len(X_test)}
                    **Random State**: 101
                    """
                )
        
        # ===== TAB 8: FEATURE IMPORTANCE =====
        with tab8:
            st.markdown("<div class='section-header'>🌟 Feature Importance Analysis</div>", unsafe_allow_html=True)
            
            # Ensure data is clean
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
            
            # Target column selection
            all_cols = df.columns.tolist()
            target_column = st.selectbox(
                "🎯 Select Target Column:",
                all_cols,
                key="feature_target"
            )
            
            if target_column:
                # Determine problem type
                if is_numeric_dtype(df[target_column]):
                    problem_type = "regression"
                else:
                    problem_type = "classification"
                
                # Data preprocessing
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                X_encoded = pd.get_dummies(X, drop_first=True)
                
                if problem_type == "classification":
                    if y.dtype == 'object':
                        y = LabelEncoder().fit_transform(y)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.3, random_state=101
                )
                
                # Model training
                with st.spinner("Calculating feature importance..."):
                    if problem_type == "classification":
                        model = RandomForestClassifier(random_state=101)
                    else:
                        model = RandomForestRegressor(random_state=101)
                    
                    model.fit(X_train, y_train)
                
                # Feature importance
                importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
                importances = importances.sort_values(ascending=False)
                
                st.markdown("---")
                
                st.subheader("🏆 Top 10 Most Important Features:")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_importances = importances.head(10)
                top_importances.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                ax.set_ylabel('Features', fontsize=12, fontweight='bold')
                ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                st.pyplot(fig)
                
                st.markdown("---")
                
                st.subheader("📊 Feature Importance Table:")
                importance_df = pd.DataFrame({
                    'Rank': range(1, len(importances) + 1),
                    'Feature': importances.index,
                    'Importance Score': importances.values,
                    'Importance %': (importances.values * 100).round(2)
                })
                
                st.dataframe(importance_df, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("📈 Cumulative Importance:")
                
                cumulative_importance = np.cumsum(importances.values)
                cumulative_pct = (cumulative_importance / cumulative_importance[-1]) * 100
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, marker='o', linestyle='-', color='steelblue')
                ax.axhline(y=80, color='r', linestyle='--', label='80% Threshold')
                ax.axhline(y=90, color='orange', linestyle='--', label='90% Threshold')
                ax.set_xlabel('Number of Features', fontweight='bold')
                ax.set_ylabel('Cumulative Importance %', fontweight='bold')
                ax.set_title('Cumulative Feature Importance', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                
                # Find threshold
                features_80 = np.argmax(cumulative_pct >= 80) + 1
                features_90 = np.argmax(cumulative_pct >= 90) + 1
                
                st.info(
                    f"""
                    **Insights:**
                    - Features for 80% importance: **{features_80}**
                    - Features for 90% importance: **{features_90}**
                    - Total features: **{len(importances)}**
                    """
                )
        
        # ===== FOOTER =====
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #888; padding: 20px;'>
            <p>📊 Data Analysis Baseline | Powered by Streamlit</p>
            <p style='font-size: 0.8em;'>All analysis based on baseline.py methodology</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted with valid data.")

else:
    # Landing page
    st.markdown(
        """
        <div style='text-align: center; padding: 50px;'>
            <h1>📊 Data Analysis Dashboard</h1>
            <h3 style='color: #666;'>Complete Baseline Analysis Pipeline</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; min-height: 150px;'>
                <h3>📋 Exploration</h3>
                <p>Comprehensive data overview and statistics</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; min-height: 150px;'>
                <h3>📈 Visualization</h3>
                <p>Interactive charts and distributions</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; min-height: 150px;'>
                <h3>🤖 ML Models</h3>
                <p>Automated model training & evaluation</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    st.markdown(
        """
        ## 🚀 Features
        
        ✅ **Data Exploration** - View statistics, dtypes, shape, and memory usage  
        ✅ **Correlation Analysis** - Before and after cleaning heatmaps  
        ✅ **Distribution Plots** - Histograms with KDE for numeric columns  
        ✅ **Visualizations** - Frequency plots and box plots  
        ✅ **Data Quality** - Missing values, duplicates, and data types  
        ✅ **Data Preprocessing** - Automatic cleaning and encoding  
        ✅ **ML Model Training** - Classification and Regression models  
        ✅ **Feature Importance** - Top features and cumulative importance  
        
        ## 📖 How to Use
        
        1. **Upload CSV** - Click "Browse files" in the sidebar
        2. **Explore** - Review data overview and statistics
        3. **Analyze** - Check correlations and visualizations
        4. **Preprocess** - Automatic data cleaning
        5. **Model** - Select target column and train model
        6. **Evaluate** - View feature importance and metrics
        
        ## 📊 Analysis Pipeline
        
        This dashboard implements the complete baseline.py analysis including:
        - Basic data exploration
        - Correlation heatmaps
        - Distribution analysis
        - Missing value handling
        - Data preprocessing
        - Model training
        - Feature importance ranking
        
        ## 🎯 Supported Problem Types
        
        - **Classification** - For categorical target variables
        - **Regression** - For numeric target variables
        
        Models automatically detect the problem type and train accordingly!
        """
    )
    
    st.markdown("---")
    
    st.info("👈 **Get started**: Upload a CSV file from the sidebar to begin analysis!")
