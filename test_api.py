# test_api.py
import requests
import json
import numpy as np
from create_realistic_ecg import generate_realistic_ecg

# Generate test ECG
ecg_signal = generate_realistic_ecg().astype(np.float32)
with open("test_ecg.bin", "wb") as f:
    f.write(ecg_signal.tobytes())

# Prepare EHR data
ehr_data = {
    "age": 62,
    "sex": 1,
    "height": 175,
    "weight": 80
}

# Send request
url = "http://localhost:8000/predict"
files = {"ecg_file": open("test_ecg.bin", "rb")}
data = {"ehr": json.dumps(ehr_data)}

try:
    response = requests.post(url, files=files, data=data)
    
    # Debug output
    print("Status Code:", response.status_code)
    print("Raw Content:", response.text)
    
    # Try to parse JSON
    try:
        json_response = response.json()
        print("JSON Response:", json_response)
    except:
        print("Failed to parse JSON. Raw response:")
        print(response.text)
        
except Exception as e:
    print("Request failed:", str(e))