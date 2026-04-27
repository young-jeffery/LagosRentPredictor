import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_url = "https://www.propertypro.ng"
search_url = f"{base_url}/property-for-rent/in/lagos"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Phase 1: Crawling the search page for property links...")
response = requests.get(search_url, headers=headers)

properties_data = []

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    listings = soup.find_all("div", class_="property-listing")
    
    target_links = []
    
    # 1. THE CRAWLER: Extracting the URLs
    for listing in listings:
        title_div = listing.find("div", class_="pl-title")
        if title_div:
            title_text = title_div.text.lower()
            # Filter out commercial properties
            if "office" not in title_text and "land" not in title_text and "shop" not in title_text:
                link_tag = listing.find("a")
                if link_tag and 'href' in link_tag.attrs:
                    target_links.append(base_url + link_tag['href'])
                    
    print(f"Found {len(target_links)} valid residential links.")
    print("Phase 2: Initiating Deep Extraction (Limiting to first 3 for testing)...\n")
    
    # 2. THE EXTRACTOR: Diving into the first 3 links
    for link in target_links[:3]:
        print(f"Scraping detailed data from: {link}")
        try:
            detail_response = requests.get(link, headers=headers)
            if detail_response.status_code == 200:
                detail_soup = BeautifulSoup(detail_response.content, "html.parser")
                
                # Extract Title
                title_tag = detail_soup.title
                title = title_tag.text.strip() if title_tag else "N/A"
                
                # Extract Description (Using the class we found in your debug file)
                desc_div = detail_soup.find("div", class_="line-paragraph")
                description = desc_div.text.strip() if desc_div else ""
                
                # Extract Amenities Grid
                amenities = []
                amen_grid = detail_soup.find("div", class_="amen-grid")
                if amen_grid:
                    for a_tag in amen_grid.find_all("a"):
                        amenities.append(a_tag.text.strip())
                        
                # 3. THE PARSER: Feature Engineering via Text Analysis
                # We turn text descriptions into binary (1 or 0) mathematical features for the ML model
                is_serviced = 1 if "serviced" in description.lower() or "serviced" in title.lower() else 0
                is_furnished = 1 if "furnished" in description.lower() or "furnished" in title.lower() else 0
                
                # Append everything to our dataset
                properties_data.append({
                    "URL": link,
                    "Title": title,
                    "Amenities": ", ".join(amenities),
                    "Is_Serviced": is_serviced,
                    "Is_Furnished": is_furnished,
                    "Description_Snippet": description[:50] + "..." # Keep terminal output clean
                })
                
            # CS Best Practice: Wait 2 seconds between page requests to avoid IP bans
            time.sleep(2) 
            
        except Exception as e:
            print(f"Error scraping {link}: {e}")

    # Convert to Pandas DataFrame and save
    df = pd.DataFrame(properties_data)
    df.to_csv("deep_lagos_rent_data.csv", index=False)
    
    print("\n" + "="*50)
    print("SUCCESS! Deep Data saved to deep_lagos_rent_data.csv")
    print(df.head())
    
else:
    print(f"Failed to load main page. Status code: {response.status_code}")