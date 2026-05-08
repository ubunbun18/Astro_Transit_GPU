import requests
import pandas as pd
from io import StringIO

def fetch_eb_via_tap():
    # VizieR TAP service
    url = "https://vizier.cds.unistra.fr/viz-bin/votable-search"
    # Alternative TAP endpoint
    url = "http://tapvizier.u-strasbg.fr/TAP/sync"
    
    query = 'SELECT "TIC", "Period" FROM "J/ApJS/258/16/table2"'
    
    print(f"Querying VizieR TAP for EB catalog...")
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "QUERY": query,
        "FORMAT": "csv"
    }
    
    try:
        response = requests.post(url, data=params)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        df.to_csv("data/tess_eb_catalog.csv", index=False)
        print(f"Successfully saved {len(df)} EB entries to data/tess_eb_catalog.csv")
    except Exception as e:
        print(f"Failed to fetch EB catalog: {e}")
        if 'response' in locals():
            print(f"Response snippet: {response.text[:200]}")

if __name__ == "__main__":
    fetch_eb_via_tap()
