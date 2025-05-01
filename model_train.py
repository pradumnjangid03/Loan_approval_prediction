import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump

# Load dataset
df = pd.read_csv('C:/pawan/python/Machine Learning/LoanAPP/loan.csv')

# Drop Loan_ID and missing target
df.drop(columns=["Loan_ID"], inplace=True)
df.dropna(subset=["Loan_Status"], inplace=True)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Features and target
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Identify column types
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Custom Label Encoder
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.columns = None

    def fit(self, X, y=None):
        # Handle both DataFrame and ndarray
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        else:
            self.columns = [f"col_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.columns)

        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns)

        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy.values  # return as array for compatibility


# Pipelines
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", MultiColumnLabelEncoder())
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, validate=True))
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, cat_cols),
    ("num", num_pipeline, num_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model
dump(pipeline, "loan_pipeline_web.joblib")
print("✅ Model saved as 'loan_pipeline_web.joblib'") 