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

