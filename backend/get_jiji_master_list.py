import requests
import json
import pandas as pd

# The exact endpoint base from your cURL capture
url = "https://jiji.ng/api_web/v1/region_adverts_count.json"

headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.5',
    'referer': 'https://jiji.ng/houses-apartments-for-rent',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36'
}

# 1218 is the base region ID for Lagos State on Jiji
params = {
    'category_slug': 'houses-apartments-for-rent',
    'region_ids': 1218  
}

print("Flipping the switch... Automated extraction of Lagos locations from Jiji has started.")

try:
    response = requests.get(url, headers=headers, params=params, timeout=10)
    data = response.json()
    
    master_locations = []
    
    # Check if the structure returns children regions dynamically
    if 'regions' in data:
        for region in data['regions']:
            name = region.get('name')
            parent = region.get('parent_name', 'Lagos')
            master_locations.append({'Sub_Location': name, 'Main_Area': parent})
            
    # Fallback to dictionary parsing if it's stored under a count mapping object
    elif isinstance(data, dict):
        # If the API only returns a flat mapping, we extract key elements
        print("API returned structural mapping object. Checking nested values...")
        
    # Let's save what we harvested
    if master_locations:
        df_master = pd.DataFrame(master_locations)
        df_master.to_csv("Jiji_Official_Lagos_Dictionary.csv", index=False)
        print(f"\n>>> SUCCESS! Collected {len(df_master)} official locations automatically.")
        print("Saved directly to 'Jiji_Official_Lagos_Dictionary.csv'")
    else:
        # Emergency backup: If the JSON response is deeply nested or encrypted, 
        # we dump the raw data to investigate its precise keys
        with open("raw_jiji_response.json", "w") as f:
            json.dump(data, f, indent=4)
        print("\nAPI data structure dumped to 'raw_jiji_response.json' for immediate extraction inspection.")

except Exception as e:
    print(f"Extraction halted due to an issue: {str(e)}")