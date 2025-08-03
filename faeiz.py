import requests

# Step 1: API Key
API_KEY = "Yg9BNfMyyRiQVe0Cz77HCGIotXixI-k-bjbqCXUjMCtR"

# Step 2: Get token
token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
)
mltoken = token_response.json()["access_token"]

# Step 3: Headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + mltoken
}

# Step 4: Define the payload
payload_scoring = {
    "input_data": [{
        "fields": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "values": [
            [7, 152, 88, 44, 0, 50, 0.337, 36],
            [2, 99, 52, 15, 94, 24.6, 0.637, 21], [1, 109, 56, 21, 135, 25.2, 0.833, 23],
            [2, 88, 74, 19, 53, 29, 0.229, 22], [17, 163, 72, 41, 114, 40.9, 0.817, 47],
            [4, 151, 90, 38, 0, 29.7, 0.294, 36], [7, 102, 74, 40, 105, 37.2, 0.204, 45],
            [0, 114, 80, 34, 285, 44.2, 0.167, 27], [2, 100, 64, 23, 0, 29.7, 0.368, 21],
            [0, 131, 88, 0, 0, 31.6, 0.743, 32], [6, 104, 74, 18, 156, 29.9, 0.722, 41],
            [3, 148, 66, 25, 0, 32.5, 0.256, 22], [4, 120, 68, 0, 0, 29.6, 0.709, 34],
            [4, 110, 66, 0, 0, 31.9, 0.471, 29], [3, 111, 90, 12, 78, 28.4, 0.495, 29],
            [6, 102, 82, 0, 0, 30.8, 0.18, 36], [6, 134, 70, 23, 130, 35.4, 0.542, 29],
            [2, 87, 0, 23, 0, 28.9, 0.773, 25], [1, 79, 60, 42, 48, 43.5, 0.678, 23],
            [2, 75, 64, 24, 55, 29.7, 0.37, 33], [8, 179, 72, 42, 130, 32.7, 0.719, 36],
            [6, 85, 78, 0, 0, 31.2, 0.382, 42], [0, 129, 110, 46, 130, 67.1, 0.319, 26],
            [2, 81, 72, 15, 76, 30.1, 0.547, 25]
        ]
    }]
}

# Step 5: Send the request to IBM Watson
scoring_url = "https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/diabetes/predictions?version=2021-05-01"
response_scoring = requests.post(scoring_url, json=payload_scoring, headers=headers)

# Step 6: Print predictions
print("Scoring response:")
try:
    print(response_scoring.json())
except ValueError:
    print(response_scoring.text)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
