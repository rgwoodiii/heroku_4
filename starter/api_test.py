import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome!"}

def test_post_pos(client):
    r = client.post("/", json={
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
    assert r.json() == {"prediction": " >50K"}


def test_post_neg(client):
    r = client.post("/", json={
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
    assert r.json() == {"prediction": "<=50K"}
