import requests
from bs4 import BeautifulSoup
import time

url = "https://www.propertypro.ng/property-for-rent/in/lagos"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Searching for the first valid residential property...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    listings = soup.find_all("div", class_="property-listing")
    
    target_link = None
    
    for listing in listings:
        # Get the title to check if it's residential
        title_div = listing.find("div", class_="pl-title")
        if title_div:
            title_text = title_div.text.lower()
            
            # THE FILTER: Skip if it's an office, land, shop, or commercial space
            if "office" not in title_text and "land" not in title_text and "shop" not in title_text and "commercial" not in title_text:
                
                # Find the Anchor tag (<a>) that contains the link to the deep page
                link_tag = listing.find("a")
                if link_tag and 'href' in link_tag.attrs:
                    # PropertyPro uses relative links (e.g., /property/123), so we add the base website to it
                    target_link = "https://www.propertypro.ng" + link_tag['href']
                    print(f"Found residential property: {title_text.strip()}")
                    print(f"Target Link Extracted: {target_link}")
                    break # We only need one link for our recon!

    if target_link:
        print("\nWaiting 2 seconds to be polite to the server...")
        time.sleep(2)
        
        print("Diving into the detailed page...")
        detail_response = requests.get(target_link, headers=headers)
        
        if detail_response.status_code == 200:
            # Save the raw HTML of the detailed page
            with open("detailed_debug.html", "w", encoding="utf-8") as file:
                file.write(detail_response.text)
            print("\nSUCCESS! Saved the deep data to detailed_debug.html")
            print("Please upload this file so we can build the Deep Scraper.")
        else:
            print(f"Failed to load detailed page. Status code: {detail_response.status_code}")
            
else:
    print(f"Failed to load main page. Status code: {response.status_code}")



# import requests
# from bs4 import BeautifulSoup

# # The  URL to scrape
# url = "https://www.propertypro.ng/property-for-rent/in/lagos"

# # Websites often block automated scripts. We use 'headers' to pretend 
# # we are a regular human using a Google Chrome browser on a Windows computer.
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# }

# print(f"Attempting to connect to: {url}...")

# # We use the requests library to get the webpage
# response = requests.get(url, headers=headers)

# # Check if the connection was successful (Status code 200 means "OK")
# if response.status_code == 200:
#     print("Connection successful! Handing data to BeautifulSoup...")
    
#     # Pass the raw website content to BeautifulSoup to organize it
#     soup = BeautifulSoup(response.content, "html.parser")
    
#     # To prove it worked, let's find and print the title of the webpage
#     page_title = soup.title.text
#     print(f"\nSuccess! The title of the page is: {page_title}")
    
# else:
#     print(f"Failed to connect. The server responded with status code: {response.status_code}")