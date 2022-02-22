# load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import precision_score, recall_score, fbeta_score
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

df = pd.read_csv("starter/data/census_without_spaces.csv")

def train_test_model(data=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # load data
    if data is None:
        df = pd.read_csv("starter/data/census_without_spaces.csv")
    else:
        df = data
    # separate target
    df['salary'] = np.where(df['salary'] == ' >50K', 1, 0)
    x = df.drop(['salary', 'fnlgt'], axis=1)
    y = df['salary']
    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=84)
    
    # one hot encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    x_train = ohe.fit_transform(x_train.values)
    x_test = ohe.transform(x_test.values)
    
    # model
    model = RandomForestClassifier(n_estimators=5)
    # fit
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))


def save_model(data=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # load data
    if data is None:
        df = pd.read_csv("starter/data/census_without_spaces.csv")
    else:
        df = data
    # separate target
    df['salary'] = np.where(df['salary'] == ' >50K', 1, 0)
    x = df.drop(['salary', 'fnlgt'], axis=1)
    y = df['salary']
    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=84)
    
    # one hot encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    x_train = ohe.fit_transform(x_train.values)
    x_test = ohe.transform(x_test.values)
    
    # model
    model = RandomForestClassifier(n_estimators=5)
    # fit
    model.fit(x_train, y_train)

    # save model
    dump(ohe, "starter/model/models/one_hot_encoding.joblib")
    dump(model, "starter/model/models/trainedmodel.pkl")
    print('models saved')


def inference(df):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # load model
    ohe = load("starter/model/models/one_hot_encoding.joblib")
    model = load("starter/model/models/trainedmodel.pkl")

    # prep data
    df['salary'] = np.where(df['salary'] == ' >50K', 1, 0)

    x = df.drop(['salary', 'fnlgt'], axis=1)
    y = df['salary']
    #transform
    x = ohe.transform(x.values)
    # predict
    pred = model.predict(x)
    return pred, y
    


def compute_model_metrics(pred, y):
    """
    Validates the trained machine learning model using\
    precision, recall, and F1.

    Inputs
    ------
    pred
    y

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, pred, average='weighted', beta=0.5)
    precision = precision_score(y, pred, average=None)
    recall = recall_score(y, pred, average=None, zero_division=1)
    return precision, recall, fbeta


if __name__ == "__main__":
    train_test_model(df)
    save_model(df)
    pred, y = inference(df)
    compute_model_metrics(pred, y)


