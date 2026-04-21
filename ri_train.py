import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

#  DATA LOADING 
df = pd.read_csv("insurance.csv")
print("First 5 rows:")
print(df.head())
print("Dataset Shape:", df.shape)

#  DATA PREPROCESSING 

# Step 1: Missing value check
print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Duplicate removal
before = df.shape[0]
df = df.drop_duplicates()
print(f"\nDuplicates removed: {before - df.shape[0]}")

# Step 3: Outlier detection (IQR method on BMI)
Q1 = df["bmi"].quantile(0.25)
Q3 = df["bmi"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["bmi"] < Q1 - 1.5 * IQR) | (df["bmi"] > Q3 + 1.5 * IQR)]
print(f"BMI Outliers detected: {len(outliers)}")

# Step 4: Feature Engineering — interaction term
df["age_bmi_interaction"] = df["age"] * df["bmi"]
print("\nNew feature 'age_bmi_interaction' added.")

# Step 5: Data type verification & summary stats
print("\nData Types:")
print(df.dtypes)
print("\nSummary Statistics:")
print(df.describe())

# FEATURE & TARGET SPLIT 
X = df.drop("charges", axis=1)
y = df["charges"]

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical features:", num_features)
print("Categorical features:", cat_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PIPELINE CREATION 
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])



ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge())
])

# MODEL TRAINING 
ridge_pipeline.fit(X_train, y_train)
print("\nModel trained successfully.")

# CROSS-VALIDATION 
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    ridge_pipeline, X_train, y_train, cv=kf, scoring="r2"
)

print("\nCross-Validation R² Scores:", np.round(cv_scores, 3))
print("Average R²  :", round(cv_scores.mean(), 3))
print("Std Dev     :", round(cv_scores.std(), 3))

#  HYPERPARAMETER TUNING
param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["auto", "svd", "cholesky", "lsqr"]
}

grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("\nBest Parameters :", grid_search.best_params_)
print("Best CV R² Score:", round(grid_search.best_score_, 3))

# সব tested combinations
results_df = pd.DataFrame(grid_search.cv_results_)[[
    "param_model__alpha", "param_model__solver",
    "mean_test_score", "std_test_score"
]]
print("\nAll Tested Combinations:")
print(results_df.sort_values("mean_test_score", ascending=False).to_string())

#  BEST MODEL SELECTION 
best_model = grid_search.best_estimator_
print("\nBest Model:", best_model.named_steps["model"])

#  MODEL PERFORMANCE EVALUATION 
y_pred = best_model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n══ Final Model Performance on Test Set ══")
print(f"R² Score : {round(r2, 3)}")
print(f"RMSE     : {round(rmse, 2)}")
print(f"MAE      : {round(mae, 2)}")
print(f"MAPE (%) : {round(mape, 2)}")

# SAVE MODEL
with open("insurance_ridge_pipeline.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nRidge pipeline saved as insurance_ridge_pipeline.pkl")