
# 🏡 House Price Predictor – Project

---

## ✅ Step 1: Install Requirements

- Install **Python** (3.9 or later)
- Install **VS Code**
- Add the **Python extension** in VS Code (Microsoft official)

---

## ✅ Step 2: Create Project Folder

1. Open VS Code.
2. Create a folder, e.g., `house-price-predictor`.
3. Inside it, create this structure:

    ```plaintext
    house-price-predictor/
    ├── app.py
    ├── model.py
    ├── requirements.txt
    └── templates/
         └── index.html
    ```

---

## ✅ Step 3: Create Virtual Environment

Open the VS Code terminal (`Ctrl + ~`) and run:

```bash
python3 -m venv venv
```

Activate it:

- **Windows (PowerShell):**
    ```bash
    .\venv\Scripts\activate
    ```
- **macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

---

## ✅ Step 4: Install Dependencies

Inside the virtual environment, install Flask and ML libraries:

```bash
pip install flask scikit-learn pandas numpy
```

**Optional:** To create a freeze file:

```bash
cd AIMLOps/Projects/house-price-predictor
pip freeze > requirements.txt
```

> **Tip:**  
> To install all dependencies later, just run:
> ```bash
> pip install -r requirements.txt
> ```

---

## ✅ Step 5: Add Code Files


### `model.py` – Training Script (Pipeline Version)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

# --- Create synthetic dataset ---
np.random.seed(42)
n = 500
data = {
    "area": np.random.randint(500, 3500, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
    "location": np.random.choice(["Urban", "Suburban", "Rural"], n)
}
df = pd.DataFrame(data)
df["price"] = (
    df["area"] * 300
    + df["bedrooms"] * 50000
    + df["bathrooms"] * 30000
    + df["location"].map({"Urban": 200000, "Suburban": 100000, "Rural": 50000})
    + np.random.randint(10000, 50000, n)
)

# --- Features & target ---
X = df[["area", "bedrooms", "bathrooms", "location"]]
y = df["price"]

# --- Preprocessing and Pipeline ---
numeric_features = ["area", "bedrooms", "bathrooms"]
numeric_transformer = StandardScaler()
categorical_features = ["location"]
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
model.fit(X, y)
joblib.dump(model, "model_pipeline.pkl")
print("✅ Pipeline trained & saved as model_pipeline.pkl")
```

Run the script to generate `model_pipeline.pkl`:

```bash
python3 model.py
```

---


### `app.py` – Flask Backend (Pipeline Version)

```python
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("model_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    location = request.form["location"]

    # Create DataFrame for prediction
    features = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "location": [location]
    })
    prediction = model.predict(features)[0]
    return render_template("index.html", prediction_text=f"Predicted House Price: ₹{round(prediction, 2)}")

if __name__ == "__main__":
    app.run(debug=True)
```

---

### `templates/index.html` – Frontend

```html
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; text-align: center; }
        .form-container { background: white; padding: 20px; margin: 50px auto; width: 400px; border-radius: 10px; box-shadow: 0px 0px 10px gray; }
        input, select { width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc; }
        button { background: #4CAF50; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="form-container">
        <h2>🏡 House Price Predictor</h2>
        <form action="/predict" method="post">
            <input type="number" name="area" placeholder="Enter Area (sq ft)" required>
            <input type="number" name="bedrooms" placeholder="Enter Bedrooms" required>
            <input type="number" name="bathrooms" placeholder="Enter Bathrooms" required>
            <select name="location" required>
                <option value="Urban">Urban</option>
                <option value="Suburban">Suburban</option>
                <option value="Rural">Rural</option>
            </select>
            <button type="submit">Predict</button>
        </form>

        {% if prediction_text %}
            <h3 style="color:blue;">{{ prediction_text }}</h3>
        {% endif %}
    </div>
</body>
</html>
```

---

## ✅ Step 6: Train the Model

```bash
python3 model.py
```
This will generate `model_pipeline.pkl` inside your project folder. ✅

---

## ✅ Step 7: Run Flask App

```bash
python3 app.py
```

You’ll see something like:

```
 * Running on http://127.0.0.1:5000
```

Open that link in your browser → form appears → enter details → prediction shown. 🎉

---

## 🧹 Bonus: Virtual Environment Tips

- **Create:**  
    ```bash
    python3 -m venv venv
    ```
- **Activate:**  
    - Windows: `./venv/Scripts/activate`
    - macOS/Linux: `source venv/bin/activate`
- **Deactivate:**  
    ```bash
    deactivate
    ```
- **Delete:**  
    Just remove the `venv` folder:
    ```bash
    rm -rf venv
    ```

---
## 🧹 Bonus: Address or Port already in use Error
If you get below error, fix it using these commands

```bash
python3 app.py           
 * Serving Flask app 'app'
 * Debug mode: on
Address already in use
Port 5000 is in use by another program. Either identify and stop that program, or start the server with a different port.
```
Run to below command to list processes using port 5000.

```bash
lsof -i :5000
```

Look for the PID (process ID), then kill it:
```bash
kill -9 <PID>
```
---

Happy Coding! 🚀

