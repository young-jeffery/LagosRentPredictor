import requests
from bs4 import BeautifulSoup

# The  URL to scrape
url = "https://www.propertypro.ng/property-for-rent/in/lagos"

# Websites often block automated scripts. We use 'headers' to pretend 
# we are a regular human using a Google Chrome browser on a Windows computer.
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Attempting to connect to: {url}...")

# We use the requests library to get the webpage
response = requests.get(url, headers=headers)

# Check if the connection was successful (Status code 200 means "OK")
if response.status_code == 200:
    print("Connection successful! Handing data to BeautifulSoup...")
    
    # Pass the raw website content to BeautifulSoup to organize it
    soup = BeautifulSoup(response.content, "html.parser")
    
    # To prove it worked, let's find and print the title of the webpage
    page_title = soup.title.text
    print(f"\nSuccess! The title of the page is: {page_title}")
    
else:
    print(f"Failed to connect. The server responded with status code: {response.status_code}")