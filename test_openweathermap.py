import requests

# Define variables
API_KEY = "35dd25fe6e4bb95083d716c76865b6fa"
CITY = "Chicago"
url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=imperial"

# Execute GET request
response = requests.get(url)

# Print response logic
if response.status_code == 200:
    data = response.json()
    print(f"Weather in {CITY}: {data['weather'][0]['description']}")
    print(f"Current Temperature: {data['main']['temp']}°F")
else:
    print(f"Failed to fetch data. Error Code: {response.status_code}")