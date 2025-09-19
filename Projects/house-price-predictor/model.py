import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# --- Create synthetic dataset ---
np.random.seed(42)
data = {
    "area": np.random.randint(500, 3500, 100),
    "bedrooms": np.random.randint(1, 5, 100),
    "bathrooms": np.random.randint(1, 4, 100),
    "location": np.random.choice(["Urban", "Suburban", "Rural"], 100)
}

df = pd.DataFrame(data)

# Assign price (basic formula + randomness)
df["price"] = (df["area"] * 300) + (df["bedrooms"] * 50000) + \
              (df["bathrooms"] * 30000) + \
              df["location"].map({"Urban": 200000, "Suburban": 100000, "Rural": 50000}) + \
              np.random.randint(10000, 50000, 100)

# One-hot encode categorical feature
df = pd.get_dummies(df, columns=["location"], drop_first=True)

# Features & target
X = df.drop("price", axis=1)
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained & saved as model.pkl")