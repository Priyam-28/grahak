import requests
from bs4 import BeautifulSoup
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import fake_useragent
import os
from tavily import TavilyClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tavily API configuration
TAVILY_API_KEY = "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"  # Set your Tavily API key as environment variable
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY environment variable not found. Tavily API calls will fail.")

# Popular e-commerce sites to target
POPULAR_ECOMMERCE_SITES = [
    "amazon.in", "flipkart.com", "myntra.com", "ajio.com", "nykaa.com",
    "snapdeal.com", "shopclues.com", "tatacliq.com", "paytmmall.com", "reliancedigital.in"
]

@dataclass
class ScrapingConfig:
    """Configuration for scraping strategies"""
    site_name: str
    selectors: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    delay_range: tuple = (1, 3)
    use_proxies: bool = False  # Disabled for simplicity
    max_retries: int = 2
    timeout: int = 10

@dataclass
class ProductData:
    """Standardized product data structure"""
    title: str = ""
    price: str = ""
    rating: str = ""
    image_url: str = ""
    description: str = ""
    availability: str = ""
    reviews_count: str = ""
    seller: str = ""
    product_url: str = ""
    platform: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)

class BrowserFingerprintManager:
    """Manages browser fingerprint randomization"""
    
    def __init__(self):
        self.ua = fake_useragent.UserAgent()
    
    def get_headers(self, platform: str = "") -> Dict[str, str]:
        """Generate randomized headers"""
        base_headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'DNT': '1',
        }
        
        # Platform-specific headers
        if platform.lower() == 'amazon':
            base_headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            })
        
        return base_headers

class SiteScrapingStrategy:
    """Base class for site-specific scraping strategies"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract data using site-specific selectors"""
        raise NotImplementedError

class AmazonScrapingStrategy(SiteScrapingStrategy):
    """Amazon-specific scraping strategy"""
    
    def __init__(self):
        config = ScrapingConfig(
            site_name="amazon",
            selectors={
                'title': '#productTitle, .product-title, [data-automation-id="product-title"]',
                'price': '.a-price-whole, .a-price .a-offscreen, #price_inside_buybox, .a-price-range',
                'rating': '.a-icon-alt, .reviewCountTextLinkedHistogram, [data-hook="average-star-rating"]',
                'image': '#landingImage, .a-dynamic-image, #imgTagWrapperId img',
                'description': '#feature-bullets ul, .a-unordered-list, #productDescription',
                'availability': '#availability span, .a-size-medium',
                'reviews_count': '#acrCustomerReviewText, [data-hook="total-review-count"]',
                'seller': '#sellerProfileTriggerId, .tabular-buybox-text'
            },
            delay_range=(1, 3)
        )
        super().__init__(config)
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract Amazon product data"""
        product = ProductData(platform="Amazon", product_url=url)
        
        # Title
        title_elem = soup.select_one(self.config.selectors['title'])
        product.title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Price - try multiple selectors
        price_elem = soup.select_one(self.config.selectors['price'])
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            product.price = price_text
        
        # Rating
        rating_elem = soup.select_one(self.config.selectors['rating'])
        if rating_elem:
            rating_text = rating_elem.get('alt', '') or rating_elem.get_text(strip=True)
            if rating_text:
                # Extract number from rating text
                import re
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                product.rating = rating_match.group(1) if rating_match else ""
        
        # Image
        img_elem = soup.select_one(self.config.selectors['image'])
        if img_elem:
            product.image_url = img_elem.get('src') or img_elem.get('data-src', '')
        
        # Description
        desc_elem = soup.select_one(self.config.selectors['description'])
        if desc_elem:
            if desc_elem.find_all('li'):
                desc_items = desc_elem.find_all('li')
                product.description = ' '.join([li.get_text(strip=True) for li in desc_items[:3]])
            else:
                product.description = desc_elem.get_text(strip=True)[:200]
        
        return product

class FlipkartScrapingStrategy(SiteScrapingStrategy):
    """Flipkart-specific scraping strategy"""
    
    def __init__(self):
        config = ScrapingConfig(
            site_name="flipkart",
            selectors={
                'title': '.B_NuCI, .x-product-title-label, ._35KyD6',
                'price': '._30jeq3._16Jk6d, .CEmiEU .hl05eU, ._1_WHN1',
                'rating': '._3LWZlK, .XQDdHH, ._3nGUzT',
                'image': '._396cs4._2amPTt._3qGmMb, .q6DClP, ._2r_T1I',
                'description': '._1mXcCf.RmoJUa, .qnEqpe, ._1AN87F',
                'availability': '._16FRp0, .Bz_DlC',
                'reviews_count': '._2_R_DZ, .row._2afbiS',
                'seller': '.L56qGx, ._3LcAWX'
            },
            delay_range=(1, 3)
        )
        super().__init__(config)
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract Flipkart product data"""
        product = ProductData(platform="Flipkart", product_url=url)
        
        # Title
        title_elem = soup.select_one(self.config.selectors['title'])
        product.title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Price
        price_elem = soup.select_one(self.config.selectors['price'])
        product.price = price_elem.get_text(strip=True) if price_elem else ""
        
        # Rating
        rating_elem = soup.select_one(self.config.selectors['rating'])
        product.rating = rating_elem.get_text(strip=True) if rating_elem else ""
        
        # Image
        img_elem = soup.select_one(self.config.selectors['image'])
        if img_elem:
            product.image_url = img_elem.get('src') or img_elem.get('data-src', '')
        
        # Description
        desc_elem = soup.select_one(self.config.selectors['description'])
        product.description = desc_elem.get_text(strip=True)[:200] if desc_elem else ""
        
        return product

class GenericScrapingStrategy(SiteScrapingStrategy):
    """Generic scraping strategy for unknown sites"""
    
    def __init__(self):
        config = ScrapingConfig(
            site_name="generic",
            selectors={
                'title': 'h1, .title, .product-title, [class*="title"], [class*="name"]',
                'price': '[class*="price"], [class*="cost"], [class*="amount"]',
                'rating': '[class*="rating"], [class*="star"], [class*="review"]',
                'image': '.product-image img, .main-image, [class*="product"] img',
                'description': '.description, .product-description, [class*="desc"]',
            },
            delay_range=(1, 2)
        )
        super().__init__(config)
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract generic product data"""
        domain = urlparse(url).netloc
        product = ProductData(platform=domain, product_url=url)
        
        # Title
        title_elem = soup.select_one(self.config.selectors['title'])
        product.title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Price
        price_elem = soup.select_one(self.config.selectors['price'])
        product.price = price_elem.get_text(strip=True) if price_elem else ""
        
        # Rating
        rating_elem = soup.select_one(self.config.selectors['rating'])
        product.rating = rating_elem.get_text(strip=True) if rating_elem else ""
        
        # Image
        img_elem = soup.select_one(self.config.selectors['image'])
        if img_elem:
            product.image_url = img_elem.get('src') or img_elem.get('data-src', '')
        
        # Description
        desc_elem = soup.select_one(self.config.selectors['description'])
        product.description = desc_elem.get_text(strip=True)[:200] if desc_elem else ""
        
        return product

class TavilyIntegration:
    """Integration with Tavily API for search-based product discovery"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = TavilyClient(api_key=api_key)
    
    def get_ecommerce_links_from_query(self, query: str, num_results: int = 10) -> List[str]:
        """
        Search for e-commerce product links using Tavily API
        Enhanced to target popular Indian e-commerce sites
        """
        try:
            # Create search query that targets popular e-commerce sites
            site_queries = []
            for site in POPULAR_ECOMMERCE_SITES[:5]:  # Top 5 sites for better results
                site_queries.append(f"site:{site} {query}")
            
            all_urls = []
            
            # Search each site individually for better targeting
            for site_query in site_queries:
                try:
                    response = self.client.search(
                        query=site_query,
                        search_depth="basic",
                        max_results=3,  # 3 results per site
                        include_domains=None,
                        exclude_domains=None
                    )
                    
                    for result in response.get('results', []):
                        url = result.get('url')
                        if url and self._is_product_url(url):
                            all_urls.append(url)
                            
                except Exception as e:
                    logger.warning(f"Error searching site {site_query}: {e}")
                    continue
            
            # If we don't have enough results, do a general search
            if len(all_urls) < num_results // 2:
                try:
                    general_query = f"{query} price buy online india"
                    response = self.client.search(
                        query=general_query,
                        search_depth="basic",
                        max_results=num_results,
                        include_domains=POPULAR_ECOMMERCE_SITES
                    )
                    
                    for result in response.get('results', []):
                        url = result.get('url')
                        if url and self._is_product_url(url) and url not in all_urls:
                            all_urls.append(url)
                            
                except Exception as e:
                    logger.error(f"Error in general Tavily search: {e}")
            
            # Remove duplicates and limit results
            unique_urls = list(dict.fromkeys(all_urls))[:num_results]
            logger.info(f"Found {len(unique_urls)} unique e-commerce URLs for query: {query}")
            return unique_urls
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []
    
    def _is_product_url(self, url: str) -> bool:
        """Check if URL is likely a product page"""
        if not url:
            return False
            
        domain = urlparse(url).netloc.lower()
        
        # Check if it's from a known e-commerce site
        if not any(site in domain for site in POPULAR_ECOMMERCE_SITES):
            return False
        
        # Check for product page indicators
        product_indicators = [
            '/dp/', '/product/', '/p/', '/item/', '/buy/',
            'product-', 'item-', '/gp/product/', '/pd/'
        ]
        
        return any(indicator in url.lower() for indicator in product_indicators)

class ScraperPoolManager:
    """Main scraper pool manager with Tavily integration"""
    
    def __init__(self, tavily_api_key: Optional[str] = TAVILY_API_KEY, max_workers: int = 5):
        self.fingerprint_manager = BrowserFingerprintManager()
        if not tavily_api_key:
            logger.error("Tavily API key is not configured. ScraperPoolManager cannot use Tavily.")
            self.tavily = None
        else:
            self.tavily = TavilyIntegration(tavily_api_key)
        self.max_workers = max_workers
        
        # Initialize scraping strategies
        self.strategies = {
            'amazon': AmazonScrapingStrategy(),
            'flipkart': FlipkartScrapingStrategy(),
            'generic': GenericScrapingStrategy()
        }
    
    def _detect_platform(self, url: str) -> str:
        """Detect platform from URL"""
        domain = urlparse(url).netloc.lower()
        
        if 'amazon' in domain:
            return 'amazon'
        elif 'flipkart' in domain:
            return 'flipkart'
        else:
            return 'generic'
    
    def _scrape_single_url(self, url: str, strategy: SiteScrapingStrategy) -> Optional[ProductData]:
        """Scrape a single URL with retries and error handling"""
        headers = self.fingerprint_manager.get_headers(strategy.config.site_name)
        
        for attempt in range(strategy.config.max_retries):
            try:
                # Add random delay
                delay = random.uniform(*strategy.config.delay_range)
                time.sleep(delay)
                
                # Make request
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=strategy.config.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Parse and extract data
                soup = BeautifulSoup(response.text, 'html.parser')
                product_data = strategy.extract_data(soup, url)
                
                # Validate extracted data
                if product_data.title:  # At least title should be present
                    logger.info(f"Successfully scraped: {url}")
                    return product_data
                else:
                    logger.warning(f"No valid data extracted from: {url}")
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                
                if attempt == strategy.config.max_retries - 1:
                    logger.error(f"Failed to scrape {url} after {strategy.config.max_retries} attempts")
        
        return None
    
    def scrape_urls(self, urls: List[str]) -> List[ProductData]:
        """Scrape multiple URLs concurrently"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {}
            
            for url in urls:
                platform = self._detect_platform(url)
                strategy = self.strategies.get(platform, self.strategies['generic'])
                
                future = executor.submit(self._scrape_single_url, url, strategy)
                future_to_url[future] = url
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        
        return results
    
    def search_and_scrape_products(self, query: str, max_results: int = 10) -> List[ProductData]:
        """
        Main function to search for products and scrape them
        This replaces the old SerpAPI functionality
        """
        if not self.tavily:
            logger.error("Tavily API not configured")
            return []
        
        try:
            # Step 1: Get e-commerce URLs using Tavily
            logger.info(f"Searching for e-commerce URLs for query: '{query}'")
            ecommerce_urls = self.tavily.get_ecommerce_links_from_query(query, num_results=max_results)
            
            if not ecommerce_urls:
                logger.warning(f"No e-commerce URLs found for query: '{query}'")
                return []
            
            # Step 2: Scrape the URLs
            logger.info(f"Scraping {len(ecommerce_urls)} URLs")
            scraped_products = self.scrape_urls(ecommerce_urls)
            
            # Step 3: Filter and sort results
            valid_products = [p for p in scraped_products if p.title and p.product_url]
            
            # Sort by completeness of data (products with price and rating first)
            valid_products.sort(
                key=lambda p: (
                    bool(p.price and p.rating),  # Has both price and rating
                    bool(p.price),               # Has price
                    bool(p.image_url),           # Has image
                    len(p.description)           # Description length
                ), 
                reverse=True
            )
            
            logger.info(f"Successfully scraped {len(valid_products)} valid products")
            return valid_products[:max_results]
            
        except Exception as e:
            logger.error(f"Error in search_and_scrape_products: {e}")
            return []

# High-level functions for the API
def search_products(query: str, max_results: int = 10) -> str:
    """
    Search for products and return formatted results
    """
    if not TAVILY_API_KEY:
        return "Error: Tavily API key not configured."
    
    logger.info(f"Searching for products: '{query}'")
    
    manager = ScraperPoolManager(tavily_api_key=TAVILY_API_KEY)
    products = manager.search_and_scrape_products(query, max_results=max_results)
    
    return format_products_as_text(products)

def format_products_as_text(products: List[ProductData]) -> str:
    """
    Format product data as readable text for LLM processing
    """
    if not products:
        return "No products found."

    formatted_text = f"Found {len(products)} products:\n\n"
    
    for i, product in enumerate(products, 1):
        formatted_text += f"--- Product {i} ---\n"
        formatted_text += f"Title: {product.title}\n"
        
        if product.price:
            formatted_text += f"Price: {product.price}\n"
        
        if product.rating:
            formatted_text += f"Rating: {product.rating}"
            if product.reviews_count:
                formatted_text += f" ({product.reviews_count} reviews)"
            formatted_text += "\n"
        
        if product.platform:
            formatted_text += f"Platform: {product.platform}\n"
        
        if product.description:
            formatted_text += f"Description: {product.description}\n"
        
        if product.product_url:
            formatted_text += f"URL: {product.product_url}\n"
        
        formatted_text += "\n"
    
    return formatted_text

# Backward compatibility functions
def scrape_website(query: str) -> str:
    """Backward compatibility function"""
    return search_products(query)

def extract_body_content(html_content: str) -> str:
    """Extract body content from HTML (for compatibility)"""
    return html_content

def clean_body_content(content: str) -> str:
    """Clean content (for compatibility)"""
    return content

def split_dom_content(content: str, max_length: int = 6000) -> List[str]:
    """Split content into chunks (for compatibility)"""
    if len(content) <= max_length:
        return [content]
    
    chunks = []
    for i in range(0, len(content), max_length):
        chunks.append(content[i:i + max_length])
    
    return chunks