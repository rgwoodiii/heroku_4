# load libraries
from training import train_test_model, inference, compute_model_metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os.path
import sys
import numpy as np

# load data
df = pd.read_csv("starter/data/census_without_spaces.csv")
model = load("starter/model/models/trainedmodel.pkl")
ohe = load("starter/model/models/one_hot_encoding.joblib")
df['salary'] = np.where(df['salary'] == ' >50K', 1, 0)

report = []
# slice

for value in df.sex.unique():
    data_slice = df[df["sex"] == value]
    # split
    x = data_slice.drop(['salary'], axis=1)
    y = data_slice['salary']
    # transform
    x = ohe.transform(x.values)
    
    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    
    # pred 
    pred = model.predict(x_test)

    precision, recall, fbeta = compute_model_metrics(pred, y_test)
    print("Value: {value}".format(value = value))
    print("Precision: {precision}".format(precision = precision))
    print("recall: {recall}".format(recall = recall))
    print("fbeta: {fbeta}".format(fbeta = fbeta))
    
    summ_dict = {"value": value, "precision": precision.mean(), "recall": recall.mean(), "fbeta": fbeta}
    report.append(summ_dict)
      
with open("slice_output.txt", "w") as report_file:
    for item in report:
        report_file.write(str(item))
