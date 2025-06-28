import requests
from bs4 import BeautifulSoup
import time
import asyncio
import json
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

async def search_products(query):
    """Search for products using Amazon search"""
    print(f"Searching for products: {query}")
    
    try:
        # Create search URL for Amazon
        search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}&ref=sr_pg_1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
        }
        
        # Make request in thread to avoid blocking
        def make_request():
            response = requests.get(search_url, headers=headers, timeout=30)
            return response.text
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        html_content = await loop.run_in_executor(None, make_request)
        
        print("Product search completed successfully!")
        return html_content
        
    except Exception as e:
        print(f"Error searching for products: {e}")
        return ""

# Keep backward compatibility
async def scrape_website(website_or_query):
    """Backward compatibility wrapper - now searches for products"""
    if website_or_query.startswith('http'):
        # If it's a URL, extract domain and treat as search query
        domain = urlparse(website_or_query).netloc
        query = domain.replace('www.', '').replace('.com', '').replace('.', ' ')
        return await search_products(query)
    else:
        # Treat as search query
        return await search_products(website_or_query)


def extract_body_content(html_content):
    """Extract body content from HTML"""
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def clean_body_content(body_content):
    """Clean body content by removing scripts and styles"""
    soup = BeautifulSoup(body_content, "html.parser")

    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text or further process the content
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content


def split_dom_content(dom_content, max_length=6000):
    """Split DOM content into chunks"""
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]
