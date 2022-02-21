import requests

data = {"age": 38,
        "workclass": " Private",
        "fnlgt": 374524,
        "education": " Bachelors",
        "education-num": 13,
        "marital-status": " Married-civ-spouse",
        "occupation": " Prof-specialty",
        "relationship": " Husband",
        "race": " White",
        "sex": " Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": " United-States"}

r = requests.post('https://udacity-api.herokuapp.com/predict/', json=data)

print("Response body: %s" % r.json())