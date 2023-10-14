import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
app = FastAPI()
class PredictionInput(BaseModel):
    radius_mean: float
    texture_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float
    def to_dict(self):
        return self.dict()

loaded_model = joblib.load('random_forest_model.joblib')

@app.post("/predict")
async def predict_model(input:  PredictionInput ):
    try:
        #convert input data into numpy array
        myInput = input.to_dict()
        myInputNp = np.array(list(myInput.values()))
        # reshape numpy array
        myInputNp = np.array(myInputNp).reshape(1, -1)
        # predict
        prediction = loaded_model.predict(myInputNp)
        # convert the prediction numpy array to list
        prediction = prediction.tolist()
        # return the prediction as a json   
        return {"predicted_class": prediction[0], "predicted_class_name": "malignant" if prediction[0] == 1 else "benign"}
    except:
        print("error converting input to dict")
        return {"error": "error converting input to dict"}

   
