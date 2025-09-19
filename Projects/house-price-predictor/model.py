import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# --- Reproducibility ---
np.random.seed(42)

# --- 1. Create synthetic dataset ---
n = 500
data = {
    "area": np.random.randint(500, 3500, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
    "location": np.random.choice(["Urban", "Suburban", "Rural"], n)
}

df = pd.DataFrame(data)

# Price formula + randomness
df["price"] = (
    df["area"] * 300
    + df["bedrooms"] * 50000
    + df["bathrooms"] * 30000
    + df["location"].map({"Urban": 200000, "Suburban": 100000, "Rural": 50000})
    + np.random.randint(10000, 50000, n)
)

# --- 2. Features & target ---
X = df[["area", "bedrooms", "bathrooms", "location"]]
y = df["price"]

# --- 3. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Preprocessing ---
numeric_features = ["area", "bedrooms", "bathrooms"]
numeric_transformer = StandardScaler()

categorical_features = ["location"]
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# --- 5. Pipeline ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# --- 6. Train ---
model.fit(X_train, y_train)

# --- 7. Evaluate ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Evaluation on Test Set:")
print(f" MAE: {mae:.2f}")
print(f" RMSE: {np.sqrt(mse):.2f}")
print(f" R²: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("\nCV R² (5-fold):", np.round(cv_scores, 3))
print("Mean CV R²:", np.round(cv_scores.mean(), 3))

# --- 8. Save pipeline ---
joblib.dump(model, "model_pipeline.pkl")
print("\n✅ Pipeline saved as model_pipeline.pkl")
