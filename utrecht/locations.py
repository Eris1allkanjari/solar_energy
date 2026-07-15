import os

import requests

API_KEY = os.environ.get("KNMI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the KNMI_API_KEY environment variable")

collection = "hourly-in-situ-meteorological-observations-validated"
base_url = f"https://api.dataplatform.knmi.nl/edr/v1/collections/{collection}"

headers = {"Authorization": API_KEY}

response = requests.get(f"{base_url}/locations", headers=headers)
response.raise_for_status()

data = response.json()

stations = [
    (f["id"], f["properties"]["name"])
    for f in data["features"]
]

for s in stations:
    if "BILT" in s[1].upper():
        print(s)

