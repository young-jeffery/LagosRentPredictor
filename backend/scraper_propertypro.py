import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

# --- Configuration ---
base_url = "https://www.propertypro.ng"
csv_filename = "propertypro_lagos_rent_data.csv"
tracking_filename = "scraped_urls.txt"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --- State Management ---
# Load previously scraped URLs to avoid duplicates and allow resuming
scraped_urls = set()
if os.path.exists(tracking_filename):
    with open(tracking_filename, "r", encoding="utf-8") as file:
        scraped_urls = set(line.strip() for line in file)
print(f"Loaded {len(scraped_urls)} previously scraped properties from memory.")

# --- Tracking Counters ---
session_success = 0
session_failed = 0
current_page = 1

print("\n" + "="*50)
print("INITIALIZING STATEFUL CRAWLER...")
print("="*50 + "\n")

# --- Phase 1: Pagination Crawl ---
while True:
    print(f"\n[PAGE {current_page}] Scanning search index for properties...")
    
    search_url = f"{base_url}/property-for-rent/in/lagos?page={current_page}"
    
    try:
        response = requests.get(search_url, headers=headers, timeout=15)
    except Exception as e:
        print(f"Network error on Page {current_page}: {e}. Retrying in 5 seconds...")
        time.sleep(5)
        continue

    if response.status_code != 200:
        print(f"Failed to load Page {current_page}. Status code: {response.status_code}. Pausing...")
        time.sleep(5)
        continue 

    soup = BeautifulSoup(response.content, "html.parser")
    listings = soup.find_all("div", class_="property-listing")
    
    # Break loop if no listings found
    if not listings:
        print(f"No properties found on Page {current_page}. Reached the end of the index.")
        break
        
    target_links = []
    
    # Extract URLs, filter out commercial spaces
    for listing in listings:
        title_div = listing.find("div", class_="pl-title")
        if title_div:
            title_text = title_div.text.lower()
            if "office" not in title_text and "land" not in title_text and "shop" not in title_text:
                link_tag = listing.find("a")
                if link_tag and 'href' in link_tag.attrs:
                    full_link = base_url + link_tag['href']
                    if full_link not in scraped_urls:
                        target_links.append(full_link)
                    
    print(f"[PAGE {current_page}] Found {len(target_links)} NEW residential links. Initiating Deep Extraction...\n")
    
    # --- Phase 2: Deep Extraction ---
    for link in target_links:
        try:
            detail_response = requests.get(link, headers=headers, timeout=10) 
            
            if detail_response.status_code == 200:
                detail_soup = BeautifulSoup(detail_response.content, "html.parser")
                
                # 1. Price
                price_div = detail_soup.find("div", class_="pricing")
                price = price_div.text.strip() if price_div else "N/A"
                
                # 2. Location
                location_icon = detail_soup.find("i", class_="fa-location-dot")
                location = location_icon.parent.text.strip() if location_icon and location_icon.parent else "N/A"
                
                # 3. Title & Description
                title_tag = detail_soup.title
                title = title_tag.text.strip() if title_tag else "N/A"
                
                desc_div = detail_soup.find("div", class_="line-paragraph")
                description = desc_div.text.strip() if desc_div else ""
                
                # 4. Amenities Grid
                amenities = []
                amen_grid = detail_soup.find("div", class_="amen-grid")
                if amen_grid:
                    for a_tag in amen_grid.find_all("a"):
                        amenities.append(a_tag.text.strip())
                        
                # 5. Base Feature Engineering
                is_serviced = 1 if "serviced" in description.lower() or "serviced" in title.lower() else 0
                is_furnished = 1 if "furnished" in description.lower() or "furnished" in title.lower() else 0
                
                # Package data WITH Raw Description
                property_data = {
                    "Location": location,
                    "Price": price,
                    "Is_Serviced": is_serviced,
                    "Is_Furnished": is_furnished,
                    "Amenities": ", ".join(amenities),
                    "Raw_Description": description, # Stored for Jupyter extraction later
                    "URL": link # Always good to have the source URL for auditing
                }
                
                # --- DISK-LEVEL APPENDING ---
                df = pd.DataFrame([property_data])
                file_exists = os.path.isfile(csv_filename)
                df.to_csv(csv_filename, mode='a', header=not file_exists, index=False, encoding='utf-8')
                
                # Update memory
                with open(tracking_filename, "a", encoding="utf-8") as track_file:
                    track_file.write(link + "\n")
                scraped_urls.add(link)
                
                session_success += 1
                print(f"[SUCCESS {session_success} | FAILED {session_failed}] Scraped: {location} | {price}")
                
            else:
                session_failed += 1
                print(f"[FAILED] Bad Status Code ({detail_response.status_code}) for link: {link}")
                
            time.sleep(1.5) 
            
        except Exception as e:
            session_failed += 1
            print(f"[FAILED] Error extracting {link}. Reason: {str(e)[:50]}...")
            time.sleep(2) 
            pass

    current_page += 1

print("\n" + "="*50)
print(f"CRAWL COMPLETE OR STOPPED.")
print(f"Total Successful This Session: {session_success}")
print(f"Total Failed This Session: {session_failed}")
print(f"Total Records Now In Database: {len(scraped_urls)}")
print("="*50)






























# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time

# base_url = "https://www.propertypro.ng"
# search_url = f"{base_url}/property-for-rent/in/lagos"
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# }

# print("Phase 1: Crawling the search page for property links...")
# response = requests.get(search_url, headers=headers)

# properties_data = []

# if response.status_code == 200:
#     soup = BeautifulSoup(response.content, "html.parser")
#     listings = soup.find_all("div", class_="property-listing")
    
#     target_links = []
    
#     # Extract URLs, filtering out commercial spaces
#     for listing in listings:
#         title_div = listing.find("div", class_="pl-title")
#         if title_div:
#             title_text = title_div.text.lower()
#             if "office" not in title_text and "land" not in title_text and "shop" not in title_text:
#                 link_tag = listing.find("a")
#                 if link_tag and 'href' in link_tag.attrs:
#                     target_links.append(base_url + link_tag['href'])
                    
#     print(f"Found {len(target_links)} residential links. Initiating Deep Extraction...\n")
    
#     # THE EXTRACTOR: Diving into ALL links
#     for link in target_links:
#         try:
#             detail_response = requests.get(link, headers=headers)
#             if detail_response.status_code == 200:
#                 detail_soup = BeautifulSoup(detail_response.content, "html.parser")
                
#                 # 1. Extract Price (Now included!)
#                 price_div = detail_soup.find("div", class_="pricing")
#                 price = price_div.text.strip() if price_div else "N/A"
                
#                 # 2. Extract Location (Now included!)
#                 location_icon = detail_soup.find("i", class_="fa-location-dot")
#                 location = location_icon.parent.text.strip() if location_icon and location_icon.parent else "N/A"
                
#                 # 3. Extract Title & Description
#                 title_tag = detail_soup.title
#                 title = title_tag.text.strip() if title_tag else "N/A"
                
#                 desc_div = detail_soup.find("div", class_="line-paragraph")
#                 description = desc_div.text.strip() if desc_div else ""
                
#                 # 4. Extract Amenities Grid
#                 amenities = []
#                 amen_grid = detail_soup.find("div", class_="amen-grid")
#                 if amen_grid:
#                     for a_tag in amen_grid.find_all("a"):
#                         amenities.append(a_tag.text.strip())
                        
#                 # 5. Feature Engineering
#                 is_serviced = 1 if "serviced" in description.lower() or "serviced" in title.lower() else 0
#                 is_furnished = 1 if "furnished" in description.lower() or "furnished" in title.lower() else 0
                
#                 properties_data.append({
#                     "Location": location,
#                     "Price": price,
#                     "Is_Serviced": is_serviced,
#                     "Is_Furnished": is_furnished,
#                     "Amenities": ", ".join(amenities)
#                 })
            
#             # Print progress to terminal
#             print(f"Scraped: {location} | {price}")
#             time.sleep(1.5) # Polite delay
            
#         except Exception as e:
#             pass

#     # Save the complete dataset
#     df = pd.DataFrame(properties_data)
#     df.to_csv("deep_lagos_rent_data.csv", index=False)
    
#     print("\n" + "="*50)
#     print("SUCCESS! Full dataset saved to deep_lagos_rent_data.csv")
# else:
#     print("Failed to load main page.")