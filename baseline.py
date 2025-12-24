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


# ===== LOAD DATA =====
df = pd.read_csv('BMW sales data (2010-2024) (1).csv')

# ===== BASIC DATA EXPLORATION =====
print("First few rows:")
print(df.head())

print("\nColumn Information:")
print(df.dtypes)

print("\nDescriptive Statistics:")
print(df.describe())

print("\n🔹 Missing Values per Column:")
print(df.isnull().sum())

print("\nDataFrame Info:")
df.info()


# ===== CORRELATION HEATMAP =====
coorelation_heatmap = df.select_dtypes(include=['int64', 'float64'])
if not coorelation_heatmap.empty:
    plt.figure(figsize=(15, 10))
    sns.heatmap(coorelation_heatmap.corr(), annot=True, cmap="coolwarm", fmt='.2f')
    plt.title("Relation among features")
    plt.show()
else:
    print("No relation between the features")


# ===== SEPARATE NUMERIC AND CATEGORICAL COLUMNS =====
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)


# ===== NUMERIC AND CATEGORICAL SUMMARY =====
if numeric_cols:
    print("\nNumeric Summary")
    print(df[numeric_cols].describe())

if categorical_cols:
    print("\nCategorical Summary")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts().head(5))


# ===== MISSING VALUES CHECK =====
print("\nMissing Values:")
print(df.isnull().sum())


# ===== HANDLE MISSING VALUES =====
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ===== CORRELATION HEATMAP AFTER CLEANING =====
coorelation_heatmap = df.select_dtypes(include=['int64', 'float64'])
if not coorelation_heatmap.empty:
    plt.figure(figsize=(15, 10))
    sns.heatmap(coorelation_heatmap.corr(), annot=True, cmap="coolwarm", fmt='.2f')
    plt.title("Relation among features (After Cleaning)")
    plt.show()
else:
    print("No relation between the features")


# ===== DISTRIBUTION PLOTS FOR NUMERIC COLUMNS =====
for col in numeric_cols:
    plt.figure(figsize=(12, 6))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.show()


# ===== FREQUENCY PLOTS FOR CATEGORICAL COLUMNS =====
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Frequency of {col}")
    plt.show()


# ===== BOX PLOTS: NUMERIC VS CATEGORICAL =====
for cat_col in categorical_cols:
    for num_col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f"{num_col} vs {cat_col}")
        plt.show()


# ===== TARGET COLUMN SELECTION =====
# NOTE: In Jupyter notebook, this used an interactive dropdown widget.
# For this .py file, you can manually specify the target column below:
target_column = input(f"Enter target column name (available: {df.columns.tolist()}): ")
# Or set it directly:
# target_column = 'Sales_Classification'  # Example


# ===== DETERMINE PROBLEM TYPE =====
if is_numeric_dtype(df[target_column]):
    problem_type = "regression"
else:
    problem_type = "classification"

print(f"Problem Type: {problem_type}")


# ===== DATA PREPROCESSING =====
X = df.drop(columns=[target_column])
y = df[target_column]

X_encoded = pd.get_dummies(X, drop_first=True)

if problem_type == "classification":
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=101)


# ===== MODEL TRAINING AND EVALUATION =====
results = {}
if problem_type == "classification":
    model = RandomForestClassifier(random_state=101)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results["RandomForestClassifier"] = {"Accuracy": acc}

print("\nModel Performance:")
pprint.pprint(results)


# ===== FEATURE IMPORTANCE =====
if problem_type == "classification":
    model = RandomForestClassifier(random_state=101)

model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
importances.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.show()

print("\nAnalysis Complete!")
