import requests
from bs4 import BeautifulSoup
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from langchain_tavily import TavilySearch
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import fake_useragent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load Tavily API key from environment variable with fallback
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY environment variable not found. Make sure to set it in your .env file.")

@dataclass
class ScrapingConfig:
    """Configuration for scraping strategies"""
    site_name: str
    selectors: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    delay_range: tuple = (1, 3)
    use_proxies: bool = True
    max_retries: int = 3
    timeout: int = 15


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


class ProxyManager:
    """Manages proxy rotation"""
    
    def __init__(self):
        # Free proxy list - in production, use paid proxy services
        self.proxies = [
            {"http": "http://proxy1:port", "https": "https://proxy1:port"},
            {"http": "http://proxy2:port", "https": "https://proxy2:port"},
            # Add more proxies here
        ]
        self.current_proxy_index = 0
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get next proxy in rotation"""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        return proxy
    
    def remove_proxy(self, proxy: Dict[str, str]):
        """Remove dead proxy from pool"""
        if proxy in self.proxies:
            self.proxies.remove(proxy)
            logger.warning(f"Removed dead proxy: {proxy}")


class BrowserFingerprintManager:
    """Manages browser fingerprint randomization"""
    
    def __init__(self):
        self.ua = fake_useragent.UserAgent()
        self.common_headers = [
            "Accept-Language",
            "Accept-Encoding", 
            "Connection",
            "Upgrade-Insecure-Requests"
        ]
    
    def get_headers(self, platform: str = "") -> Dict[str, str]:
        """Generate randomized headers"""
        base_headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.5', 'en-GB,en;q=0.5', 'en-US,en;q=0.9']),
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': random.choice(['no-cache', 'max-age=0']),
            'DNT': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }
        
        # Platform-specific headers
        if platform.lower() == 'amazon':
            base_headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Sec-Fetch-User': '?1'
            })
        elif platform.lower() == 'flipkart':
            base_headers.update({
                'X-User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            })
        
        return base_headers


class SiteScrapingStrategy:
    """Base class for site-specific scraping strategies"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract data using site-specific selectors"""
        raise NotImplementedError
    
    def preprocess_html(self, html: str) -> str:
        """Site-specific HTML preprocessing"""
        return html


class AmazonScrapingStrategy(SiteScrapingStrategy):
    """Amazon-specific scraping strategy"""
    
    def __init__(self):
        config = ScrapingConfig(
            site_name="amazon",
            selectors={
                'title': '#productTitle, .product-title',
                'price': '.a-price-whole, .a-price .a-offscreen, #price_inside_buybox',
                'rating': '.a-icon-alt, .reviewCountTextLinkedHistogram',
                'image': '#landingImage, .a-dynamic-image',
                'description': '#feature-bullets ul, .a-unordered-list',
                'availability': '#availability span',
                'reviews_count': '#acrCustomerReviewText',
                'seller': '#sellerProfileTriggerId, .tabular-buybox-text'
            },
            delay_range=(2, 5)
        )
        super().__init__(config)
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract Amazon product data"""
        product = ProductData(platform="Amazon", product_url=url)
        
        # Title
        title_elem = soup.select_one(self.config.selectors['title'])
        product.title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Price
        price_elem = soup.select_one(self.config.selectors['price'])
        product.price = price_elem.get_text(strip=True) if price_elem else ""
        
        # Rating
        rating_elem = soup.select_one(self.config.selectors['rating'])
        if rating_elem:
            rating_text = rating_elem.get('alt', '') or rating_elem.get_text(strip=True)
            product.rating = rating_text.split()[0] if rating_text else ""
        
        # Image
        img_elem = soup.select_one(self.config.selectors['image'])
        if img_elem:
            product.image_url = img_elem.get('src') or img_elem.get('data-src', '')
        
        # Description
        desc_elem = soup.select_one(self.config.selectors['description'])
        if desc_elem:
            desc_items = desc_elem.find_all('li')
            product.description = ' '.join([li.get_text(strip=True) for li in desc_items[:3]])
        
        # Availability
        avail_elem = soup.select_one(self.config.selectors['availability'])
        product.availability = avail_elem.get_text(strip=True) if avail_elem else ""
        
        # Reviews count
        reviews_elem = soup.select_one(self.config.selectors['reviews_count'])
        if reviews_elem:
            reviews_text = reviews_elem.get_text()
            product.reviews_count = ''.join(filter(str.isdigit, reviews_text))
        
        return product


class FlipkartScrapingStrategy(SiteScrapingStrategy):
    """Flipkart-specific scraping strategy"""
    
    def __init__(self):
        config = ScrapingConfig(
            site_name="flipkart",
            selectors={
                'title': '.B_NuCI, .x-product-title-label',
                'price': '._30jeq3._16Jk6d, .CEmiEU .hl05eU',
                'rating': '._3LWZlK, .XQDdHH',
                'image': '._396cs4._2amPTt._3qGmMb, .q6DClP',
                'description': '._1mXcCf.RmoJUa, .qnEqpe',
                'availability': '._16FRp0, .Bz_DlC',
                'reviews_count': '._2_R_DZ, .row._2afbiS',
                'seller': '.L56qGx, ._3LcAWX'
            },
            delay_range=(1, 4)
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
        product.description = desc_elem.get_text(strip=True) if desc_elem else ""
        
        # Availability
        avail_elem = soup.select_one(self.config.selectors['availability'])
        product.availability = avail_elem.get_text(strip=True) if avail_elem else ""
        
        return product


class MyntraScrapingStrategy(SiteScrapingStrategy):
    """Myntra-specific scraping strategy"""
    
    def __init__(self):
        config = ScrapingConfig(
            site_name="myntra",
            selectors={
                'title': '.pdp-name, .pdp-title',
                'price': '.pdp-price strong, .pdp-mrp',
                'rating': '.index-overallRating, .ratings-rating',
                'image': '.image-grid-image, .product-sliderContainer img',
                'description': '.pdp-product-description-content, .product-description',
                'availability': '.size-buttons-size-button, .product-actions',
                'reviews_count': '.ratings-count, .index-ratingsCount',
                'seller': '.pdp-seller-name, .product-brand'
            },
            delay_range=(2, 4)
        )
        super().__init__(config)
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract Myntra product data"""
        product = ProductData(platform="Myntra", product_url=url)
        
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
        
        return product


class TavilyIntegration:
    """Integration with Tavily for search-based scraping"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.search_tool = TavilySearch(tavily_api_key=api_key)  # Changed to use tavily_api_key parameter
    
    def search_products(self, query: str, max_results: int = 10, **kwargs) -> List[Dict]:
        """Search for products using Tavily"""
        try:
            # Build a comprehensive search query for products
            product_query = f"{query} price comparison review specifications buy"
            
            # Use Tavily to search with search_depth=2 for more detailed results
            results = self.search_tool.invoke({
                "query": product_query,
                "k": max_results,
                "search_depth": "advanced",  # Use advanced search for product queries
                "include_raw_content": True,  # Get full content when available
                "include_images": True,       # Try to get product images
                **kwargs
            })
            
            # Tavily returns a list of dictionaries with search results
            formatted_results = []
            for result in results.get("results", []):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0),
                    "raw_content": result.get("raw_content", ""),
                    "image_url": result.get("image_url", "")
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []
    
    def search_amazon(self, query: str, **kwargs) -> List[Dict]:
        """Search Amazon specifically"""
        amazon_query = f"{query} site:amazon.com OR site:amazon.in"
        return self.search_products(amazon_query, **kwargs)
    
    def search_flipkart(self, query: str, **kwargs) -> List[Dict]:
        """Search Flipkart specifically"""
        flipkart_query = f"{query} site:flipkart.com"
        return self.search_products(flipkart_query, **kwargs)
    
    def search_general_shopping(self, query: str, **kwargs) -> List[Dict]:
        """Search general shopping sites"""
        shopping_query = f"{query} buy online price comparison review"
        return self.search_products(shopping_query, **kwargs)


class ScraperPoolManager:
    """Main scraper pool manager"""
    
    def __init__(self, tavily_key: Optional[str] = TAVILY_API_KEY, max_workers: int = 5):
        self.proxy_manager = ProxyManager()
        self.fingerprint_manager = BrowserFingerprintManager()
        if not tavily_key:
            logger.error("Tavily API key is not configured. ScraperPoolManager cannot use Tavily.")
            self.tavily = None
        else:
            self.tavily = TavilyIntegration(tavily_key)
        self.max_workers = max_workers
        
        # Initialize scraping strategies
        self.strategies = {
            'amazon': AmazonScrapingStrategy(),
            'flipkart': FlipkartScrapingStrategy(),
            'myntra': MyntraScrapingStrategy()
        }
    
    def _detect_platform(self, url: str) -> str:
        """Detect platform from URL"""
        domain = urlparse(url).netloc.lower()
        
        if 'amazon' in domain:
            return 'amazon'
        elif 'flipkart' in domain:
            return 'flipkart'
        elif 'myntra' in domain:
            return 'myntra'
        else:
            return 'generic'
    
    def _scrape_single_url(self, url: str, strategy: SiteScrapingStrategy) -> Optional[ProductData]:
        """Scrape a single URL with retries and error handling"""
        headers = self.fingerprint_manager.get_headers(strategy.config.site_name)
        
        for attempt in range(strategy.config.max_retries):
            try:
                # Get proxy if enabled
                proxy = self.proxy_manager.get_proxy() if strategy.config.use_proxies else None
                
                # Add random delay
                delay = random.uniform(*strategy.config.delay_range)
                time.sleep(delay)
                
                # Make request
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxy,
                    timeout=strategy.config.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Parse and extract data
                soup = BeautifulSoup(response.text, 'html.parser')
                product_data = strategy.extract_data(soup, url)
                
                logger.info(f"Successfully scraped: {url}")
                return product_data
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                
                if proxy and "proxy" in str(e).lower():
                    self.proxy_manager.remove_proxy(proxy)
                
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
                strategy = self.strategies.get(platform)
                
                if strategy:
                    future = executor.submit(self._scrape_single_url, url, strategy)
                    future_to_url[future] = url
                else:
                    logger.warning(f"No strategy found for platform: {platform}")
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        
        return results
    
    def search_and_scrape(self, query: str, platforms: List[str] = None) -> List[ProductData]:
        """Search using Tavily and scrape the results"""
        if not self.tavily:
            logger.error("Tavily API key not provided")
            return []
        
        platforms = platforms or ['amazon', 'general_shopping']
        all_results = []
        
        for platform in platforms:
            try:
                if platform == 'amazon':
                    search_results = self.tavily.search_amazon(query, max_results=10)
                elif platform == 'flipkart':
                    search_results = self.tavily.search_flipkart(query, max_results=10)
                else:
                    search_results = self.tavily.search_general_shopping(query, max_results=10)
                
                # Extract URLs from search results
                urls = []
                for result in search_results[:10]:  # Limit to top 10 results
                    url = result.get('url')
                    if url:
                        urls.append(url)
                
                # Scrape the URLs
                scraped_data = self.scrape_urls(urls)
                all_results.extend(scraped_data)
                
            except Exception as e:
                logger.error(f"Error searching {platform}: {e}")
        
        return all_results

    def get_product_data_from_tavily_result(self, result: Dict, platform_hint: str) -> Optional[ProductData]:
        """
        Extracts ProductData from a single Tavily result item.
        This is useful for extracting structured data from Tavily search results.
        """
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        raw_content = result.get("raw_content", "")  # Get the raw content if available
        score = result.get("score", 0)
        image_url = result.get("image_url", "")

        if not title or not url:
            return None

        # Try to extract price from content using basic regex
        import re
        price_match = re.search(r'[\$₹]\s*[\d,]+(?:\.\d{2})?', content)
        price = price_match.group() if price_match else ""

        # Try to extract rating from content
        rating_match = re.search(r'(\d+\.?\d*)\s*(?:out of|/)\s*5|(\d+\.?\d*)\s*star', content, re.IGNORECASE)
        rating = rating_match.group(1) or rating_match.group(2) if rating_match else ""

        # Detect platform from URL
        detected_platform = self._detect_platform(url)
        if detected_platform != 'generic':
            platform_hint = detected_platform.title()

        return ProductData(
            title=str(title),
            price=str(price),
            rating=str(rating),
            image_url=str(image_url),  # Use image URL from Tavily when available
            description=str(raw_content or content)[:200] + "..." if len(raw_content or content) > 200 else str(raw_content or content),
            availability="",  # Not available in search results
            reviews_count="",  # Not available in search results
            seller="",  # Not available in search results
            product_url=str(url),
            platform=platform_hint,
            raw_data=result
        )

    def search_and_optionally_scrape(self, query: str, platforms: Optional[List[str]] = None, max_results_per_platform: int = 5) -> List[ProductData]:
        """
        Search using Tavily. Extract basic data from search results and optionally scrape for more detailed data.
        """
        if not self.tavily:
            logger.error("Tavily is not initialized in ScraperPoolManager.")
            return []

        platforms = platforms or ['general_shopping', 'amazon']
        all_product_data: List[ProductData] = []
        urls_to_scrape: List[str] = []

        for platform in platforms:
            try:
                search_results_raw = []
                platform_for_data = platform
                
                if platform.lower() == 'amazon':
                    search_results_raw = self.tavily.search_amazon(query, max_results=max_results_per_platform * 2)
                    platform_for_data = "Amazon"
                elif platform.lower() == 'flipkart':
                    search_results_raw = self.tavily.search_flipkart(query, max_results=max_results_per_platform * 2)
                    platform_for_data = "Flipkart"
                elif platform.lower() == 'general_shopping':
                    search_results_raw = self.tavily.search_general_shopping(query, max_results=max_results_per_platform)
                    platform_for_data = "GeneralShopping"
                else:
                    logger.warning(f"Unsupported platform for Tavily search: {platform}")
                    continue

                logger.info(f"Tavily found {len(search_results_raw)} results for '{query}' on {platform}")

                temp_platform_products = []
                for res in search_results_raw:
                    # For general shopping, try to extract structured data directly
                    if platform.lower() == 'general_shopping':
                        product = self.get_product_data_from_tavily_result(res, platform_for_data)
                        if product:
                            temp_platform_products.append(product)
                    else:
                        # For Amazon/Flipkart, collect URLs for scraping
                        url = res.get('url')
                        if url:
                            # Basic filter for platform-specific URLs
                            domain = urlparse(url).netloc.lower()
                            if ((platform.lower() == 'amazon' and 'amazon' in domain) or 
                                (platform.lower() == 'flipkart' and 'flipkart' in domain) or
                                (platform.lower() == 'general_shopping')):
                                
                                if url not in urls_to_scrape:
                                    urls_to_scrape.append(url)

                all_product_data.extend(temp_platform_products[:max_results_per_platform])

            except Exception as e:
                logger.error(f"Error during Tavily search for {platform}: {e}")

        # Scrape URLs collected (mostly for Amazon/Flipkart)
        unique_urls_to_scrape = list(set(urls_to_scrape))
        logger.info(f"Attempting to scrape {len(unique_urls_to_scrape)} unique URLs.")

        # Limit scraping to avoid excessive requests
        effective_scrape_limit = max(0, (max_results_per_platform * len(platforms)) - len(all_product_data))

        if unique_urls_to_scrape and effective_scrape_limit > 0:
            scraped_data = self.scrape_urls(unique_urls_to_scrape[:effective_scrape_limit])
            all_product_data.extend(scraped_data)
            # Ensure we don't exceed total limit
            all_product_data = all_product_data[:(max_results_per_platform * len(platforms))]

        logger.info(f"Collected {len(all_product_data)} product data entries in total for query '{query}'.")
        return all_product_data
    
    def export_data(self, data: List[ProductData], format: str = "json", filename: str = None):
        """Export scraped data to various formats"""
        if not filename:
            filename = f"scraped_data_{int(time.time())}"
        
        if format.lower() == "json":
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump([product.__dict__ for product in data], f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            import csv
            with open(f"{filename}.csv", 'w', newline='', encoding='utf-8') as f:
                if data:
                    fieldnames = data[0].__dict__.keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for product in data:
                        writer.writerow(product.__dict__)
        
        logger.info(f"Data exported to {filename}.{format}")


def format_products_as_text(products: List[ProductData]) -> str:
    """
    Format product data as readable text, ensuring essential fields are present.
    """
    if not products:
        return "No products found."

    valid_products = [p for p in products if p.title and p.product_url]
    if not valid_products:
        return "No valid product data to format."

    formatted_text = f"Found {len(valid_products)} products:\n\n"
    
    for i, product in enumerate(valid_products, 1):
        formatted_text += f"--- Product {i} ---\n"
        formatted_text += f"Title: {product.title}\n"
        
        if product.price:
            formatted_text += f"Price: {product.price}\n"
        
        if product.rating:
            formatted_text += f"Rating: {product.rating}"
            if product.reviews_count and product.reviews_count != "0":
                formatted_text += f" ({product.reviews_count} reviews)\n"
            else:
                formatted_text += "\n"
        
        if product.availability:
            formatted_text += f"Availability: {product.availability}\n"
        
        if product.seller:
            formatted_text += f"Seller: {product.seller}\n"

        if product.platform:
            formatted_text += f"Platform: {product.platform}\n"
        
        if product.product_url:
            formatted_text += f"URL: {product.product_url}\n"
        
        if product.description:
            formatted_text += f"Description: {product.description}\n"

        formatted_text += "\n"
    
    return formatted_text


def fetch_product_information_tavily(query: str, platforms: Optional[List[str]] = None, max_results_per_platform: int = 3) -> str:
    """
    High-level function to fetch product information using Tavily and format it.
    This is intended to be the primary function called by the router.
    """
    if not TAVILY_API_KEY:
        logger.error("Tavily API key is not available. Cannot fetch product information.")
        return "Error: Tavily API key not configured."

    logger.info(f"Fetching product information for query: '{query}' using Tavily.")
    manager = ScraperPoolManager(tavily_key=TAVILY_API_KEY)

    product_data_list = manager.search_and_optionally_scrape(
        query,
        platforms=platforms or ['general_shopping', 'amazon'],
        max_results_per_platform=max_results_per_platform
    )

    if not product_data_list:
        logger.info(f"No product data found by ScraperPoolManager for query: '{query}'.")
        return "No products found matching your query."

    formatted_text = format_products_as_text(product_data_list)
    logger.info(f"Formatted product information for query: '{query}'. Length: {len(formatted_text)}")
    return formatted_text


def search_products(query: str, platform: str = "general_shopping,amazon", max_results: int = 3) -> str:
    """
    Search for products using the query and return formatted results.
    'platform' can be a comma-separated list of platforms like 'general_shopping,amazon'.
    This function now primarily uses fetch_product_information_tavily.
    The 'max_results' here means max_results_per_platform.
    """
    logger.info(f"search_products called for query: '{query}', platform(s): '{platform}'")

    platform_list = [p.strip().lower() for p in platform.split(',') if p.strip()]
    if not platform_list:
        platform_list = ['general_shopping', 'amazon']

    return fetch_product_information_tavily(query, platforms=platform_list, max_results_per_platform=max_results)


def scrape_website(query_or_url: str) -> str:
    """
    Backward compatibility function.
    If it's a URL, it tries to detect platform and scrape.
    If it's a query, it uses the new search_products flow.
    """
    parsed_url = urlparse(query_or_url)
    if parsed_url.scheme and parsed_url.netloc:
        logger.info(f"scrape_website called with URL: {query_or_url}. Attempting direct scrape.")
        if not TAVILY_API_KEY:
            logger.error("Tavily API key is not available. Cannot initialize ScraperPoolManager for direct scraping.")
            return "Error: Tavily API key not configured for scraping."
        manager = ScraperPoolManager(tavily_key=TAVILY_API_KEY)
        # Note: scrape_urls expects a list of URLs.
        product_data = manager.scrape_urls([query_or_url])
        return format_products_as_text(product_data)
    else:
        logger.info(f"scrape_website called with query: {query_or_url}. Using new search_products.")
        return search_products(query_or_url)  # platform and max_results will use defaults


def extract_body_content(html_content: str) -> str:
    """
    Extract body content from HTML (for compatibility)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    return body.get_text() if body else ""


def clean_body_content(content: str) -> str:
    """
    Clean content (for compatibility)
    """
    # Remove excessive whitespace and newlines
    cleaned = ' '.join(content.split())
    return cleaned


def split_dom_content(content: str, max_length: int = 6000) -> List[str]:
    """
    Split content into chunks (for compatibility)
    """
    if len(content) <= max_length:
        return [content]
    
    chunks = []
    for i in range(0, len(content), max_length):
        chunks.append(content[i:i + max_length])
    
    return chunks


