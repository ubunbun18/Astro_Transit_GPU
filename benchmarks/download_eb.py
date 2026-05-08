import requests
import os

def download_eb_catalog():
    url = "https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
    dest = "data/tess_eb_catalog.csv"
    
    if os.path.exists(dest):
        print(f"Catalog already exists at {dest}")
        return

    print(f"Downloading TESS EB catalog from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_eb_catalog()
