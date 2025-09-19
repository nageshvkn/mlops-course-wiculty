from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Pre-trained Pipeline ---
model = joblib.load("model_pipeline.pkl")


# ------------------------------
# ROUTES
# ------------------------------

@app.route("/")
def index():
    """Landing page (index.html)"""
    return render_template("index.html")


@app.route("/home")
def home_page():
    """Alternative home page (home.html)"""
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form submission for price prediction.
    Works for both normal form POST and AJAX requests.
    """
    try:

        # --- Collect Form Data ---
        area = float(request.form.get("area", 0))
        bedrooms = int(request.form.get("bedrooms", 0))
        bathrooms = int(request.form.get("bathrooms", 0))
        location = request.form.get("location", "Rural")

        # --- Create DataFrame for prediction ---
        features = pd.DataFrame({
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "location": [location]
        })

        # --- Make prediction ---
        prediction = model.predict(features)[0]
        prediction_text = f"Predicted Price: ₹{round(prediction, 2):,}"

        # --- AJAX Request Handling ---
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"prediction": prediction_text})
        # --- Normal form submit ---
        return render_template("index.html", prediction_text=prediction_text)

    except Exception as e:
        # Error handling (useful for debugging/student learning)
        error_msg = f"Error: {str(e)}"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"prediction": error_msg})
        return render_template("index.html", prediction_text=error_msg)


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
