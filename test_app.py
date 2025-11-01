import requests
import json

test_data = {
    "age": 37,
    "workclass": 4,
    "education_num": 9,
    "marital_status": 1,
    "occupation": 10,
    "relationship": 1,
    "race": 4,
    "sex" : 1,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": 39
}

response = requests.post("http://localhost:5010/predict", json=test_data,headers={"Content-Type": "application/json"})

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
print("Prediction:", response.json().get("prediction"))
print("Income Bracket:", response.json().get("income"))
