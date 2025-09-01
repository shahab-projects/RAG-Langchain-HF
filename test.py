import requests

response = requests.post(
    "http://127.0.0.1:8000/rag",
    json={"question": "What is 5G?"}
)
print(response.json())