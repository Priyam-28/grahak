from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from bs4 import BeautifulSoup
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class PlaywrightScraper:
    def __init__(self):
        self.async_browser = None
        self.toolkit = None
        self.tools_by_name = None
        self.playwright = None
    
    async def _initialize_browser(self):
        """Initialize Playwright browser if not already initialized"""
        if self.async_browser is None:
            # Create a new browser instance with proper context management
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()
            self.async_browser = await self.playwright.chromium.launch(headless=True)
            self.toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.async_browser)
            tools = self.toolkit.get_tools()
            self.tools_by_name = {tool.name: tool for tool in tools}
    
    async def _close_browser(self):
        """Close the browser"""
        if self.async_browser:
            await self.async_browser.close()
            self.async_browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        self.toolkit = None
        self.tools_by_name = None

async def scrape_website_async(website: str, timeout: int = 30000) -> Optional[str]:
    """Async version of the website scraper"""
    logger.info(f"Scraping website: {website}")
    scraper = PlaywrightScraper()
    try:
        await scraper._initialize_browser()
        
        # Get the navigate tool
        navigate_tool = scraper.tools_by_name.get("navigate_browser")
        if not navigate_tool:
            raise Exception("Navigate tool not found in Playwright toolkit")
        
        # Navigate to the website with timeout
        logger.info("Navigating to website...")
        result = await navigate_tool.arun({
            "url": website,
            "timeout": timeout
        })
        logger.info(f"Navigation result: {result}")
        
        # Check if navigation was successful
        if "status code 200" not in result and "error" in result.lower():
            raise Exception(f"Navigation failed: {result}")
        
        # Try to get page content using get_elements tool
        get_elements_tool = scraper.tools_by_name.get("get_elements")
        if get_elements_tool:
            try:
                # Get the entire page HTML
                page_content = await get_elements_tool.arun({
                    "selector": "html",
                    "attributes": ["outerHTML"]
                })
                
                if page_content and isinstance(page_content, str):
                    # Parse the JSON-like response
                    import json
                    try:
                        parsed_content = json.loads(page_content.replace("'", '"'))
                        if parsed_content and len(parsed_content) > 0:
                            html_content = parsed_content[0].get("outerHTML", "")
                            if html_content:
                                logger.info("Website scraped successfully!")
                                return html_content
                    except json.JSONDecodeError:
                        # If it's not JSON, maybe it's already HTML
                        if "<html" in page_content.lower():
                            return page_content
            except Exception as e:
                logger.warning(f"get_elements failed: {e}")
        
        # Fallback: try extract_text tool for content
        extract_text_tool = scraper.tools_by_name.get("extract_text")
        if extract_text_tool:
            try:
                text_content = await extract_text_tool.arun({})
                if text_content:
                    # Create minimal HTML structure with the text
                    html_content = f"<html><body>{text_content}</body></html>"
                    logger.info("Website text extracted successfully!")
                    return html_content
            except Exception as e:
                logger.warning(f"extract_text failed: {e}")
        
        # Final fallback: try to access browser page directly
        if hasattr(scraper.async_browser, 'contexts') and scraper.async_browser.contexts:
            context = scraper.async_browser.contexts[0]
            if context.pages:
                page = context.pages[0]
                html_content = await page.content()
                logger.info("Website scraped successfully via direct page access!")
                return html_content
        
        raise Exception("Could not retrieve page content using any method")
        
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        raise Exception(f"Error scraping website: {e}")
    finally:
        await scraper._close_browser()

def scrape_website(website: str, timeout: int = 30000) -> Optional[str]:
    """Synchronous wrapper for the async scraper"""
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(scrape_website_async(website, timeout))
        finally:
            try:
                # Cancel all running tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # Wait for all tasks to complete
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()
    else:
        return loop.run_until_complete(scrape_website_async(website, timeout))

def extract_body_content(html_content: str) -> str:
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

def clean_body_content(body_content: str) -> str:
    """Clean body content by removing scripts and styles"""
    try:
        soup = BeautifulSoup(body_content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
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

def split_dom_content(dom_content: str, max_length: int = 6000) -> list[str]:
    """Split DOM content into chunks"""
    if not dom_content:
        return [""]
    
    return [
        dom_content[i : i + max_length] 
        for i in range(0, len(dom_content), max_length)
    ]

