# load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, fbeta_score
import json
from joblib import load
import pytest
import os.path
import logging


# functions
def test_answer():
    assert 5 == 5
    
    
## load data
@pytest.fixture()
def data():
    data = pd.read_csv("starter/data/census_without_spaces.csv")
    return data

# data
def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."

def test_slice_averages(data):
    """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
    for cat_feat in data["workclass"].unique():
        avg_value = data[data["workclass"] == cat_feat]["hours-per-week"].mean()
        assert (
            49 > avg_value > 28
        ), f"For {cat_feat}, average of {avg_value} not between 40 and 28"    

# model
def test_perf(data):
    # model
    model = load("starter/model/models/trainedmodel.pkl")
    ohe = load("starter/model/models/one_hot_encoding.joblib")
    
    # prep data
    data['salary'] = np.where(data['salary'] == ' >50K', 1, 0)

    x = data.drop(['salary', 'fnlgt'], axis=1)
    y = data['salary']
    
    #transform
    x = ohe.transform(x.values)
    
    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    
    #score
    assert model.score(x_test, y_test) >= .6, f"score is lower than expected."
    
    
def test_inference(data):
    # model
    model = load("starter/model/models/trainedmodel.pkl")
    ohe = load("starter/model/models/one_hot_encoding.joblib")
    
    # prep data
    data['salary'] = np.where(data['salary'] == ' >50K', 1, 0)

    x = data.drop(['salary', 'fnlgt'], axis=1)
    y = data['salary']
    
    #transform
    x = ohe.transform(x.values)
    #predict
    pred = model.predict(x)
    print(model.score(x, y))
    assert pred.shape[0] == y.shape[0], f"number of predictions are different from expected."

# metric review
def test_compute_model_metrics(data):
    # model
    model = load("starter/model/models/trainedmodel.pkl")
    ohe = load("starter/model/models/one_hot_encoding.joblib")
    
    # prep data
    data['salary'] = np.where(data['salary'] == ' >50K', 1, 0)

    x = data.drop(['salary', 'fnlgt'], axis=1)
    y = data['salary']
    
    #transform
    x = ohe.transform(x.values)
    #predict
    pred = model.predict(x)
    
    fbeta = fbeta_score(y, pred, average='weighted', beta=0.5)
    precision = precision_score(y, pred, average = None)
    recall = recall_score(y, pred, average = None)
    assert fbeta >= .8, f"fbeta is lower than expected."
    assert precision.mean() >= .8, f"precision is lower than expected."
    assert recall.mean() >= .8, f"recall is lower than expected."
