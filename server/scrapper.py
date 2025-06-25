import requests
from bs4 import BeautifulSoup
import time
import asyncio
from crawl4ai import *

async def scrape_website(website):
    """Scrape website using crawl4ai"""
    print(f"Scraping website: {website}")
    
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(url=website)
            print("Website scraped successfully!")
            return result.markdown  # or result.html if you want HTML
        except Exception as e:
            print(f"Error scraping website: {e}")
            return ""


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
