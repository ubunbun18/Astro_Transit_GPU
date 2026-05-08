import requests
import xml.etree.ElementTree as ET

def find_eb_tables():
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/tables"
    response = requests.get(url)
    if response.status_code == 200:
        # The response is XML (VOSI tableset)
        try:
            root = ET.fromstring(response.content)
            for table in root.findall('.//table'):
                name = table.find('name').text
                desc = table.find('description').text if table.find('description') is not None else ""
                if "eclipsing" in desc.lower() or "eb" in name.lower() or "binary" in desc.lower():
                    print(f"Table: {name}")
                    print(f"Desc: {desc[:100]}...")
                    print("-" * 20)
        except Exception as e:
            print(f"Error parsing XML: {e}")
    else:
        print("Failed to fetch tables")

if __name__ == "__main__":
    find_eb_tables()
