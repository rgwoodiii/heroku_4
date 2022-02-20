# Put the code for your API here.
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI
from typing import Union, List, Optional


from pydantic import BaseModel, Field
import uvicorn
from joblib import load
import os.path
import sys


# load model
model = load("starter/model/models/trainedmodel.pkl")
ohe = load("starter/model/models/one_hot_encoding.joblib")


# models with pydantic
class ClassifierFeatureIn(BaseModel):
    age: int = Field(..., example=50)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2500, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")     
        
# instantiate app with fastapi
app = FastAPI()

    
# greeting
@app.get("/")
async def greet_user():
    return {"Welcome!"}


@app.post("/predict/")
async def predict(data1: ClassifierFeatureIn):
    df = pd.DataFrame.from_dict([data1])
    
    # transform 
    x = ohe.transform(df.values)
    
    # predict
    pred = model.predict(x)
    pred = pred[0]
    pred = str(np.where(pred == 1, '>50K', '<=50K'))
    return {'pred': pred}

"""
# pydantic output of the model
class ClassifierOut(BaseModel):
    # The forecast output will be either >50K or <50K
    forecast: str = "Income <=50k"
"""    
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
