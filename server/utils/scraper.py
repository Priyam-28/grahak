import requests
from bs4 import BeautifulSoup
import time
import logging

logger = logging.getLogger(__name__)

def scrape_website(website):
    """Scrape website using requests with better error handling"""
    logger.info(f"Scraping website: {website}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(
            website, 
            timeout=(10, 30),  # (connect timeout, read timeout)
            allow_redirects=True,
            verify=True
        )
        response.raise_for_status()
        
        logger.info("Website scraped successfully!")
        return response.text
        
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise Exception("Website took too long to respond")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        raise Exception("Could not connect to the website")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        raise Exception(f"Website returned error: {e}")
    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        raise Exception(f"Error accessing website: {e}")
    finally:
        session.close()


def extract_body_content(html_content):
    """Extract body content from HTML with error handling"""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        body_content = soup.body
        if body_content:
            return str(body_content)
        return html_content  # Return full content if no body tag
    except Exception as e:
        logger.error(f"Error extracting body content: {e}")
        return html_content


def clean_body_content(body_content):
    """Clean body content by removing scripts and styles"""
    try:
        soup = BeautifulSoup(body_content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.extract()

        # Get text content
        cleaned_content = soup.get_text(separator="\n")
        cleaned_content = "\n".join(
            line.strip() for line in cleaned_content.splitlines() 
            if line.strip() and len(line.strip()) > 3
        )

        return cleaned_content
    except Exception as e:
        logger.error(f"Error cleaning content: {e}")
        return body_content


def split_dom_content(dom_content, max_length=6000):
    """Split DOM content into chunks"""
    if not dom_content:
        return [""]
    
    return [
        dom_content[i : i + max_length] 
        for i in range(0, len(dom_content), max_length)
    ]
