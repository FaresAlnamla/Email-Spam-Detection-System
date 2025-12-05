import requests

payload = {"text": "Congratulations! You won a free prize, click here now"}
r = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(r.status_code)
print(r.json())
