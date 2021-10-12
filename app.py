from flask import Flask, request, jsonify, make_response
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import sys
from flatten_json import flatten
from flask_cors import cross_origin

flask_app = Flask(__name__)

api = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Radiotherapy replan predicter", 
		  description = "Predicts whether a patient is likely to need a replan or not")

name_space = api.namespace('prediction', description='Prediction APIs', decorators=[cross_origin()])

model = api.model('Prediction params', 
				  {'neckWidth': fields.Float(required = True),
				  'neckDepth': fields.Float(required = True),
                  'bodyEqSphDi': fields.Float(required = True),
				  'bodyVolume': fields.Float(required = True),
                  'weight': fields.Float(required = True),
                  'ctvVol': fields.Float(required = True),
                  'ctvEqdDi': fields.Float(required = True),
                  'ptvVol': fields.Float(required = True),
                  'ptvEqdDi': fields.Float(required = True),
                  'staging': fields.List(fields.Float, required = True)})

classifier = joblib.load('radioModel.sav')

@api.route("/api")
class MainClass(Resource):
    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    @api.expect(model)
    def post(self):
        api.logger.info("in post")
        try:
            api.logger.info("here")
            formData = flatten(request.json)
            data = [val for val in formData.values()]
            prediction = classifier.predict(np.array(data).reshape(1, -1))
            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "prediction": int(prediction[0])
                })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })

if __name__ == "__main__":
    flask_app.run(debug=True)