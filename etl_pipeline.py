# ============================================
# COMPLETE DATA PIPELINE WITH GRAPHS & TABLES
# (ParserError FIXED)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ============================================
# 1. EXTRACT (ROBUST CSV LOADER)
# ============================================
DATA_PATH = "/content/raw_data.csv"

df = pd.read_csv(
    DATA_PATH,
    sep=",",
    engine="python",
    skipinitialspace=True
)

# Handle missing values
df.replace("?", np.nan, inplace=True)

print("Dataset loaded successfully")
print("Raw dataset shape:", df.shape)

# ============================================
# 2. TABLE: DATASET SUMMARY
# ============================================
print("\n===== DATASET SUMMARY TABLE =====\n")
summary_table = df.describe(include="all")
display(summary_table)

# ============================================
# 3. GRAPH: INCOME DISTRIBUTION
# ============================================
income_counts = df["income"].value_counts()

plt.figure()
income_counts.plot(kind="bar")
plt.title("Income Class Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

# ============================================
# 4. GRAPH: AGE DISTRIBUTION
# ============================================
plt.figure()
df["age"].plot(kind="hist", bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ============================================
# 5. TABLE: EDUCATION VS INCOME
# ============================================
print("\n===== EDUCATION VS INCOME TABLE =====\n")
education_income_table = pd.crosstab(df["education"], df["income"])
display(education_income_table)

# ============================================
# 6. FEATURE / TARGET SPLIT
# ============================================
X = df.drop(columns=["income"])
y = df["income"].map({"<=50K": 0, ">50K": 1})

# ============================================
# 7. COLUMN IDENTIFICATION
# ============================================
numeric_features = X.select_dtypes(include=["int64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# ============================================
# 8. TRANSFORMATION PIPELINE
# ============================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ============================================
# 9. TRAIN / TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# 10. FIT & TRANSFORM
# ============================================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Train Shape:", X_train_processed.shape)
print("Processed Test Shape:", X_test_processed.shape)

# ============================================
# 11. LOAD (SAVE OUTPUT FILES)
# ============================================
np.save("X_train.npy", X_train_processed)
np.save("X_test.npy", X_test_processed)
np.save("y_train.npy", y_train.values)
np.save("y_test.npy", y_test.values)

print("\nETL PIPELINE + EDA COMPLETED SUCCESSFULLY")
