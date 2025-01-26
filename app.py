import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import redirect

app = Flask(__name__)
CORS(app)
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))

@app.route("/")
def home():
	return redirect("https://boston-house-prices-frontend.vercel.app/")

@app.route("/predict_api", methods = ["POST"])
def predict_api():
	data = request.json["data"]
	print(data)
	print(np.array(list(data.values())).reshape(1, -1))
	new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
	output = regmodel.predict(new_data)
	print(output[0])
	return jsonify(output[0])

@app.route("/predict", methods = ["POST"])
def predict():
	data = [float(x) for x in request.form.values()] #convert to float
	final_input = scalar.transform(np.array(data).reshape(1, -1))
	print(final_input)
	output = regmodel.predict(final_input)[0]
	return redirect(f"https://boston-house-prices-frontend.vercel.app/?prediction={output}")

if __name__ == "__main__":
	app.run()