#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('BMW sales data (2010-2024) (1).csv')


# In[3]:


display(df.head())


# In[68]:


print("Column Information:")
display(df.dtypes)

print("\nDescriptive Statistics:")
display(df.describe())

print("\n🔹 Missing Values per Column:")
missing=display(df.isnull().sum())

print("\n Information:")
df.info()


# In[5]:


coorelation_heatmap=df.select_dtypes(include=['int64','float64'])
if not coorelation_heatmap.empty:
    plt.figure(figsize=(15,10)) 
    sns.heatmap(coorelation_heatmap.corr(),annot=True,cmap="coolwarm",fmt='.2f')
    plt.title("Relation among features")
    plt.show()
else:
    print("no relation between the features")


# In[6]:


numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)


# In[7]:


if numeric_cols:
    print("\n Numeric Summary")
    display(df[numeric_cols].describe())
if categorical_cols:
    print("\n Categorical Summary")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        display(df[col].value_counts().head(5))


# In[62]:


print(df.info())
print("\n Missing Values:")
print(df.isnull().sum())
print(df.describe())


# In[9]:


df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# In[10]:


coorelation_heatmap=df.select_dtypes(include=['int64','float64'])
if not coorelation_heatmap.empty:
    plt.figure(figsize=(15,10)) 
    sns.heatmap(coorelation_heatmap.corr(),annot=True,cmap="coolwarm",fmt='.2f')
    plt.title("Relation among features")
    plt.show()
else:
    print("no relation between the features")


# In[11]:


for col in numeric_cols:
    plt.figure(figsize=(12,6))
    sns.histplot(df[col],kde=True,bins=20)
    plt.title(f"Distribuation of{col}")
    plt.show()


# In[12]:


for col in categorical_cols:
    plt.figure(figsize=(6,4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Frequency of {col}")
    plt.show()


# In[45]:


target_column = input("Enter the target column name: ")


# In[46]:


from pandas.api.types import is_numeric_dtype

if is_numeric_dtype(df[target_column]):
    problem_type = "regression"
else:
    problem_type = "classification"

print(f"Problem Type: {problem_type}")


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = df.drop(columns=[target_column],axis=1)
y = df[target_column]

X_encoded = pd.get_dummies(X, drop_first=True)

if problem_type == "classification":
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
elif problem_type == "classification":
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=101)


# In[64]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pprint

results = {}
if problem_type == "regression":
    model = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=101)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}
elif problem_type == "classification":
    model = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=101)
    }
    for name, model in model.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"Accuracy": acc}

print("\nModel Performance:")
pprint.pprint(results)


# In[49]:


if problem_type == "classification":
    model = RandomForestClassifier(random_state=101)

model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
importances.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(10,6))
importances.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.show()

