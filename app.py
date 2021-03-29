
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.cropsuggestion import crop_suggestion_dict
import joblib
import traceback
from flask_restful import reqparse
app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/croprec', methods=['POST','GET'])
def predict_crop():
	lr = joblib.load("Models/crop_rec.pkl")
	if lr:
		try:
			json = request.get_json()	 
			model_columns = joblib.load("Models/croprecommenation_model_cols.pkl")
			temp=list(json[0].values())
			vals=np.array(temp)
			prediction = lr.predict(temp)
			print("here:",prediction)        
			return jsonify({'prediction': str(prediction[0])})

		except:        
			return jsonify({'trace': traceback.format_exc()})
	else:
		return ('No model here to use')
    
@app.route('/fertrec', methods=['POST','GET'])
def predict_fertilizer():
	lr = joblib.load("Models/fertilizer_recommender.pkl")
	if lr:
		try:
			json = request.get_json()	 
			model_columns = joblib.load("Models/fertilizerrec_columns.pkl")
			temp=list(json[0].values())
			vals=np.array(temp)
			prediction = lr.predict(temp)
			print("here:",prediction)        
			return jsonify({'prediction': str(prediction[0])})

		except:        
			return jsonify({'trace': traceback.format_exc()})
	else:
		return ('No model here to use')
	
@app.route('/cropsuggest', methods=['POST','GET'])
def suggest_crop():
	try:
		json = request.get_json()	 
		state= json[0]["state"]
		cropsuggest =[]

		for crop in crop_suggestion_dict.keys():
		  varieties = list(crop_suggestion_dict[crop].keys())
		  for variety in varieties:
		    states = crop_suggestion_dict[crop][variety]
		    if state in states:
		      cropsuggest.append([crop,variety])

		cropsuggest = dict(cropsuggest)
		return jsonify({'suggestion': cropsuggest})

	except:        
		return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)
    
