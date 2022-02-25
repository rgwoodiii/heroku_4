import pytest
from fastapi.testclient import TestClient
from main import app
import requests

@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client


def test_get(client):
    r = requests.get("https://udacity-api.herokuapp.com/")
    assert r.status_code == 200
    assert r.json() == ['Welcome!']

def test_post_pos(client):
    r = requests.post("https://udacity-api.herokuapp.com/predict", json={
        "age": 50,
        "workclass": "State-gov",
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {'pred': '<=50K'} 


def test_post_neg(client):
    r = requests.post("https://udacity-api.herokuapp.com/predict", json={
        "age": 38,
        "workclass": "Private",
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
    })
    assert r.status_code == 200
    assert r.json() != {'pred': '>50K'} 
