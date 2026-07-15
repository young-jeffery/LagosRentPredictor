import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# --- Configuration ---
base_url = "https://jiji.ng"
search_url = f"{base_url}/lagos/houses-apartments-for-rent"
csv_filename = "jiji_property_rent_data.csv"
tracking_filename = "jiji_scraped_urls.txt"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# --- State Management ---
scraped_urls = set()
if os.path.exists(tracking_filename):
    with open(tracking_filename, "r", encoding="utf-8") as file:
        scraped_urls = set(line.strip() for line in file)
print(f"Loaded {len(scraped_urls)} previously scraped Jiji properties from memory.")

session_success = 0
session_failed = 0
target_scroll_links = 12000 # Adjust this to grab more links per session

print("\n" + "="*50)
print("INITIALIZING JIJI HYBRID CRAWLER...")
print("="*50 + "\n")

# --- Phase 1: Selenium Infinite Scroll ---
print("Booting up browser to handle infinite scroll...")

# Set up Chrome options (Running non-headless first so you can see if Jiji throws a CAPTCHA)
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
# chrome_options.add_argument("--headless") # Uncomment this later if you want it to run invisibly

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get(search_url)
time.sleep(5) # Let the initial page load

target_links = set()
last_height = driver.execute_script("return document.body.scrollHeight")

print("Scrolling to harvest links...")
while len(target_links) < target_scroll_links:
    # Extract links currently on screen
    elements = driver.find_elements(By.CSS_SELECTOR, "a.qa-advert-list-item")
    for el in elements:
        link = el.get_attribute("href")
        if link and link not in scraped_urls:
            target_links.add(link)
            
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3) # Wait for new properties to load
    
    # Check if we've hit the absolute bottom of the database
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("Reached the end of the infinite scroll!")
        break
    last_height = new_height

driver.quit() # Close browser to free up RAM

target_links = list(target_links)
print(f"\n[PHASE 1 COMPLETE] Harvested {len(target_links)} NEW links. Initiating Deep Extraction...\n")

# --- Phase 2: BeautifulSoup Deep Extraction ---
for link in target_links:
    try:
        detail_response = requests.get(link, headers=headers, timeout=15) 
        
        if detail_response.status_code == 200:
            detail_soup = BeautifulSoup(detail_response.content, "html.parser")
            
            # 1. Extract Price
            price_span = detail_soup.find("span", class_="qa-advert-price-view-value")
            price = price_span.text.strip() if price_span else "N/A"
            
            # 2. Extract Location (Region)
            location_div = detail_soup.find("div", class_="b-advert-info-statistics--region")
            location = location_div.text.strip() if location_div else "N/A"
            
            # 3. Extract Top Icons (Beds, Baths, Type)
            top_icons = []
            icon_divs = detail_soup.find_all("div", class_="b-advert-icon-attribute")
            for div in icon_divs:
                span = div.find("span")
                if span:
                    top_icons.append(span.text.strip())
            
            # 4. Extract Detailed Attributes Grid (Key-Value pairs)
            attributes_dict = {}
            attr_divs = detail_soup.find_all("div", class_="b-advert-attribute")
            for div in attr_divs:
                key_div = div.find("div", class_="b-advert-attribute__key")
                val_div = div.find("div", class_="b-advert-attribute__value")
                if key_div and val_div:
                    attributes_dict[key_div.text.strip()] = val_div.text.strip()
                    
            # 5. Extract Amenities Tags
            amenities = []
            tag_divs = detail_soup.find_all("div", class_="b-advert-attributes__tag")
            for div in tag_divs:
                amenities.append(div.text.strip())
                
            # 6. Extract Raw Description
            desc_span = detail_soup.find("span", class_="qa-description-text")
            description = desc_span.text.strip() if desc_span else "N/A"
            
            # Package the data
            property_data = {
                "URL": link,
                "Price": price,
                "Location": location,
                "Top_Icons": ", ".join(top_icons),
                "Detailed_Attributes": str(attributes_dict), # Saved as a stringified dictionary
                "Amenities": ", ".join(amenities),
                "Raw_Description": description
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
            print(f"[SUCCESS {session_success} | FAILED {session_failed}] Scraped: {price} | {location[:30]}...")
            
        else:
            session_failed += 1
            print(f"[FAILED] Status {detail_response.status_code} on link: {link}")
            
        time.sleep(2) # Polite delay
        
    except Exception as e:
        session_failed += 1
        print(f"[FAILED] Error extracting {link}. Reason: {str(e)[:50]}...")
        time.sleep(3)
        pass

print("\n" + "="*50)
print(f"JIJI CRAWL COMPLETE OR STOPPED.")
print(f"Total Successful This Session: {session_success}")
print(f"Total Records Now In Database: {len(scraped_urls)}")
print("="*50)