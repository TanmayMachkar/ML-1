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
    column_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
    data = request.json["data"]
    print(data)
    data_df = pd.DataFrame([data], columns=column_names)
    new_data = scalar.transform(data_df)
    print(new_data)
    output = regmodel.predict(new_data)[0]
    return jsonify({"prediction": output})

if __name__ == "__main__":
	app.run(debug = True)