import requests
import json


# URL that resolves to workspace
#URL = "http://127.0.0.1:8000" 
URL = "https://udacity-api.herokuapp.com/"


# call APIs
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

response = requests.request("get", URL, headers=headers)
print(response.text)