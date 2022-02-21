import requests
import json


# URL that resolves to workspace
#URL = "http://127.0.0.1:8000/predict/"
URL = "https://udacity-api.herokuapp.com/predict/"


# call APIs
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

payload = {
        "age": 38,
        "workclass": "Private",
        "fnlgt": 374524,
        "education": " Bachelors",
        "education-num": 13,
        "marital-status": " Married-civ-spouse",
        "occupation": " Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": " United-States"
}

response = requests.request("POST", URL, headers=headers, dta = payload)
print(response.text)