"""Send an example prediction request to the deployed API."""
import os
import requests


API_URL = os.environ.get("API_URL", "https://YOUR-RENDER-SERVICE.onrender.com")
EXAMPLE = {
    "age": 39, "workclass": "State-gov", "fnlwgt": 77516,
    "education": "Bachelors", "education-num": 13,
    "marital-status": "Never-married", "occupation": "Adm-clerical",
    "relationship": "Not-in-family", "race": "White", "sex": "Male",
    "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
    "native-country": "United-States",
}

if __name__ == "__main__":
    response = requests.post(f"{API_URL.rstrip('/')}/predict", json=EXAMPLE, timeout=30)
    print(f"Inference result: {response.json()}")
    print(f"HTTP status code: {response.status_code}")
