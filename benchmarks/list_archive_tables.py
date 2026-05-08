import requests

def list_tables():
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/tables"
    response = requests.get(url)
    if response.status_code == 200:
        print(response.text)
    else:
        print("Failed to fetch tables")

if __name__ == "__main__":
    list_tables()
