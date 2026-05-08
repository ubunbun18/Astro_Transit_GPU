import requests

def find_tess_tables():
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE table_name LIKE '%tess%'"
    params = {
        "QUERY": query,
        "FORMAT": "csv"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print(response.text)
    else:
        print(f"Failed to fetch: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    find_tess_tables()
