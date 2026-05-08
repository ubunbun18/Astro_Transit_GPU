import requests
import pandas as pd
from io import StringIO

def fetch_eb_from_vizier():
    # VizieR table J/ApJS/258/16/table2 (Prsa+ 2022)
    url = "https://vizier.cds.unistra.fr/viz-bin/votable-search"
    # Actually, a simpler URL for CSV:
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    params = {
        "-source": "J/ApJS/258/16/table2",
        "-out.max": "10000",
        "-out": "TIC,Period,Depth",
        "-mime": "csv"
    }
    print("Fetching EB catalog from VizieR...")
    response = requests.get(url, params=params)
    if response.status_code == 200:
        # Filter out comments (starting with #)
        lines = [line for line in response.text.splitlines() if not line.startswith('#')]
        csv_data = "\n".join(lines)
        df = pd.read_csv(StringIO(csv_data), sep=';')
        df.to_csv("data/tess_eb_catalog.csv", index=False)
        print(f"EB catalog saved with {len(df)} entries.")
    else:
        print(f"Failed: {response.status_code}")

if __name__ == "__main__":
    fetch_eb_from_vizier()
