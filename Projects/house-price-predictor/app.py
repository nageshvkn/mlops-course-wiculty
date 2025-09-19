from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/home")
def home_page():
    return render_template("home.html")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    location = request.form["location"]

    loc_urban = 1 if location == "Urban" else 0
    loc_suburban = 1 if location == "Suburban" else 0

    features = np.array([[area, bedrooms, bathrooms, loc_suburban, loc_urban]])
    prediction = model.predict(features)[0]
    prediction_text = f"Predicted Price: ₹{round(prediction, 2)}"

    # If AJAX request, return snippet
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or True:
        return f'<div id="prediction">{prediction_text}</div>'
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
