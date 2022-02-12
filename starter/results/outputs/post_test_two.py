import requests
import json


# URL that resolves to workspace
URL = "http://127.0.0.1:8000/predict/"

# call APIs
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

payload = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 202683,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 48,
        "native-country": "United-States"
    }

response = requests.request("POST", URL, headers=headers, dta = payload)
print(response.text)