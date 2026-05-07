from astroquery.mast import Observations
import pandas as pd
import os

def get_sector_1_targets(limit=50):
    print(f"Querying MAST for TESS Sector 1 targets (limit={limit})...")
    # Search for TESS observations in Sector 1
    obs_table = Observations.query_criteria(
        project="TESS",
        obs_collection="TESS",
        sequence_number=1,
        dataproduct_type="timeseries"
    )
    
    # Extract TIC IDs (target_name usually contains the ID)
    # Filter for SPOC (original mission data)
    spoc_obs = obs_table[obs_table['provenance_name'] == 'SPOC']
    
    # Get unique TIC IDs
    tic_ids = []
    for name in spoc_obs['target_name']:
        if str(name).startswith('TIC'):
            tic_ids.append(name)
        else:
            tic_ids.append(f"TIC {name}")
            
    unique_tics = sorted(list(set(tic_ids)))[:limit]
    
    df = pd.DataFrame({"tic_id": unique_tics})
    df.to_csv("sector1_full.csv", index=False)
    print(f"Successfully saved {len(unique_tics)} targets to sector1_full.csv")

if __name__ == "__main__":
    if not os.path.exists("scripts"):
        os.makedirs("scripts")
    get_sector_1_targets(20000)
