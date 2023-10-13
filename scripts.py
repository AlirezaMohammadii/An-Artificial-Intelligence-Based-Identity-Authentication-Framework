import requests

r = requests.get("https://youtube.com")
print(r.status_code)
print(r.ok)
