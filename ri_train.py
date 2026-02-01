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


df = pd.read_csv("insurance.csv")

print(df)



df["age_bmi_interaction"] = df["age"] * df["bmi"]

X = df.drop("charges", axis=1)
y = df["charges"]

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns




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



ridge_model = Ridge()

ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", ridge_model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ridge_pipeline.fit(X_train, y_train)




y_pred = ridge_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)


print("R² Score:", round(r2, 3))
print("RMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))

with open("insurance_ridge_pipeline.pkl","wb") as f:
    pickle.dump(ridge_pipeline, f)


print("Ridge pipeling saved as insurance_ridge_pipeline.pkl")