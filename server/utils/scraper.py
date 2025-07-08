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
from firecrawl import FirecrawlApp # Added Firecrawl

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load API keys from environment variables with fallback
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY environment variable not found. Make sure to set it in your .env file.")
if not FIRECRAWL_API_KEY:
    logger.warning("FIRECRAWL_API_KEY environment variable not found. Make sure to set it in your .env file.")

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
    currency: str = "INR" # Assuming Indian Rupees
    rating: str = "" # e.g., "4.5 out of 5 stars"
    image_url: str = ""
    description: str = "" # Detailed description
    short_summary: str = "" # A brief summary
    availability: str = ""
    reviews_count: str = ""
    reviews: List[str] = field(default_factory=list) # List of review texts
    seller: str = ""
    product_url: str = ""
    platform: str = "" # e.g., Amazon, Flipkart, Myntra
    raw_data: Dict[str, Any] = field(default_factory=dict) # Raw data from Firecrawl or Tavily
    llm_parsed_data: Dict[str, Any] = field(default_factory=dict) # Data specifically from LLM parsing


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
        
        title_elem = soup.select_one(self.config.selectors['title'])
        product.title = title_elem.get_text(strip=True) if title_elem else ""
        
        price_elem = soup.select_one(self.config.selectors['price'])
        product.price = price_elem.get_text(strip=True) if price_elem else ""
        
        rating_elem = soup.select_one(self.config.selectors['rating'])
        if rating_elem:
            rating_text = rating_elem.get('alt', '') or rating_elem.get_text(strip=True)
            product.rating = rating_text.split()[0] if rating_text else ""
        
        img_elem = soup.select_one(self.config.selectors['image'])
        if img_elem:
            product.image_url = img_elem.get('src') or img_elem.get('data-src', '')
        
        desc_elem = soup.select_one(self.config.selectors['description'])
        if desc_elem:
            desc_items = desc_elem.find_all('li')
            product.description = ' '.join([li.get_text(strip=True) for li in desc_items[:3]])
        
        avail_elem = soup.select_one(self.config.selectors['availability'])
        product.availability = avail_elem.get_text(strip=True) if avail_elem else ""
        
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
        
        title_elem = soup.select_one(self.config.selectors['title'])
        product.title = title_elem.get_text(strip=True) if title_elem else ""
        
        price_elem = soup.select_one(self.config.selectors['price'])
        product.price = price_elem.get_text(strip=True) if price_elem else ""
        
        rating_elem = soup.select_one(self.config.selectors['rating'])
        product.rating = rating_elem.get_text(strip=True) if rating_elem else ""
        
        img_elem = soup.select_one(self.config.selectors['image'])
        if img_elem:
            product.image_url = img_elem.get('src') or img_elem.get('data-src', '')
        
        desc_elem = soup.select_one(self.config.selectors['description'])
        product.description = desc_elem.get_text(strip=True) if desc_elem else ""
        
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
        
        title_elem = soup.select_one(self.config.selectors['title'])
        product.title = title_elem.get_text(strip=True) if title_elem else ""
        
        price_elem = soup.select_one(self.config.selectors['price'])
        product.price = price_elem.get_text(strip=True) if price_elem else ""
        
        rating_elem = soup.select_one(self.config.selectors['rating'])
        product.rating = rating_elem.get_text(strip=True) if rating_elem else ""
        
        img_elem = soup.select_one(self.config.selectors['image'])
        if img_elem:
            product.image_url = img_elem.get('src') or img_elem.get('data-src', '')
        
        return product


class TavilyIntegration:
    """Integration with Tavily for search-based scraping"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.search_tool = TavilySearch(tavily_api_key=api_key)
    
    def search_products(self, query: str, max_results: int = 10, **kwargs) -> List[Dict]:
        """Search for products using Tavily"""
        try:
            product_query = f"{query} price comparison review specifications buy"
            results = self.search_tool.invoke({
                "query": product_query,
                "k": max_results,
                "search_depth": "advanced",
                "include_raw_content": True,
                "include_images": True,
                **kwargs
            })
            formatted_results = []
            if results and "results" in results:
                for result in results["results"]:
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
        amazon_query = f"{query} site:amazon.com OR site:amazon.in"
        return self.search_products(amazon_query, **kwargs)
    
    def search_flipkart(self, query: str, **kwargs) -> List[Dict]:
        flipkart_query = f"{query} site:flipkart.com"
        return self.search_products(flipkart_query, **kwargs)
    
    def search_general_shopping(self, query: str, **kwargs) -> List[Dict]:
        shopping_query = f"{query} buy online price comparison review"
        return self.search_products(shopping_query, **kwargs)


class ScraperPoolManager:
    """Main scraper pool manager"""
    
    def __init__(self, tavily_key: Optional[str] = TAVILY_API_KEY, firecrawl_key: Optional[str] = FIRECRAWL_API_KEY, max_workers: int = 5):
        self.proxy_manager = ProxyManager()
        self.fingerprint_manager = BrowserFingerprintManager()

        if not tavily_key:
            logger.error("Tavily API key is not configured. ScraperPoolManager cannot use Tavily.")
            self.tavily = None
        else:
            self.tavily = TavilyIntegration(tavily_key)

        if not firecrawl_key:
            logger.error("Firecrawl API key is not configured. ScraperPoolManager cannot use Firecrawl.")
            self.firecrawl_client = None
        else:
            self.firecrawl_client = FirecrawlApp(api_key=firecrawl_key)

        self.max_workers = max_workers
        
        self.strategies = {
            'amazon': AmazonScrapingStrategy(),
            'flipkart': FlipkartScrapingStrategy(),
            'myntra': MyntraScrapingStrategy()
        }
    
    def _detect_platform(self, url: str) -> str:
        domain = urlparse(url).netloc.lower()
        if 'amazon' in domain: return 'amazon'
        if 'flipkart' in domain: return 'flipkart'
        if 'myntra' in domain: return 'myntra'
        return 'generic'

    def _scrape_single_url_with_firecrawl(self, url: str) -> Optional[Dict[str, Any]]:
        if not self.firecrawl_client:
            logger.error(f"Firecrawl client not initialized. Cannot scrape {url}.")
            return None
        
        logger.info(f"Attempting to scrape with Firecrawl: {url}")
        try:
            params = {'pageOptions': {'onlyMainContent': True}}
            scraped_data = self.firecrawl_client.scrape_url(url, params=params)
            
            if scraped_data and ('markdown' in scraped_data or 'content' in scraped_data or 'llm_extraction' in scraped_data):
                logger.info(f"Successfully scraped with Firecrawl: {url}")
                scraped_data['product_url'] = url
                scraped_data['platform_detected'] = self._detect_platform(url)
                return scraped_data
            elif scraped_data and 'error' in scraped_data:
                 logger.warning(f"Firecrawl returned an error for {url}. Error: {scraped_data.get('error')}")
                 return None
            else:
                logger.warning(f"Firecrawl returned no significant content or known error for {url}. Data: {scraped_data}")
                return None
        except Exception as e:
            logger.error(f"Firecrawl scraping failed for {url}: {e}")
            return None

    def scrape_and_parse_urls(self, urls: List[str], user_query: str) -> List[ProductData]:
        raw_firecrawl_data_list: List[Dict[str, Any]] = []
        parsed_product_data_list: List[ProductData] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self._scrape_single_url_with_firecrawl, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        raw_firecrawl_data_list.append(result)
                except Exception as e:
                    logger.error(f"Error processing future for Firecrawl result of {url}: {e}")
        
        if not raw_firecrawl_data_list:
            logger.info("No data successfully scraped by Firecrawl.")
            return []

        try:
            from .parser import parse_firecrawl_data_with_ollama
            logger.info(f"Attempting to parse {len(raw_firecrawl_data_list)} Firecrawl results with Ollama.")
            for fc_data in raw_firecrawl_data_list:
                platform_hint = fc_data.get('platform_detected', self._detect_platform(fc_data.get('product_url', '')))
                parsed_product = parse_firecrawl_data_with_ollama(
                    firecrawl_data=fc_data,
                    user_query=user_query,
                    site_name_hint=platform_hint
                )
                if parsed_product:
                    if not parsed_product.platform:
                         parsed_product.platform = platform_hint
                    parsed_product.product_url = fc_data.get('product_url', '')
                    parsed_product_data_list.append(parsed_product)
                else:
                    logger.warning(f"LLM Parser returned None for URL: {fc_data.get('product_url')}")
        except ImportError:
            logger.error("Could not import 'parse_firecrawl_data_with_ollama' from .parser. Returning raw data.")
            for fc_data in raw_firecrawl_data_list:
                parsed_product_data_list.append(ProductData(
                    product_url=fc_data.get('product_url', ''),
                    platform=fc_data.get('platform_detected', 'generic'),
                    raw_data=fc_data,
                    title=fc_data.get('metadata', {}).get('title', 'Title not found in metadata'),
                    description=fc_data.get('markdown', fc_data.get('content', 'Content not found'))[:1000]
                ))
        except Exception as e:
            logger.error(f"Error during LLM parsing phase: {e}", exc_info=True)

        logger.info(f"scrape_and_parse_urls completed. Returning {len(parsed_product_data_list)} products.")
        return parsed_product_data_list

    def _construct_tavily_query(self, base_query: str, site: Optional[str] = None, min_price: Optional[Union[int, float]] = None, max_price: Optional[Union[int, float]] = None) -> str:
        query_parts = [base_query]
        price_parts = []
        if min_price is not None: price_parts.append(f"over {min_price}")
        if max_price is not None: price_parts.append(f"under {max_price}")
        if price_parts:
            price_query = " price " + " and ".join(price_parts)
            if " and " in price_query and len(price_parts) == 2:
                 price_query = f" price between {min_price} and {max_price}"
            query_parts.append(price_query)
        if min_price is not None or max_price is not None: query_parts.append("rupees")
        if site:
            if site.lower() == "amazon": query_parts.append("site:amazon.in")
            elif site.lower() == "flipkart": query_parts.append("site:flipkart.com")
            elif site.lower() == "myntra": query_parts.append("site:myntra.com")
        final_query = " ".join(query_parts)
        logger.info(f"Constructed Tavily query: {final_query}")
        return final_query

    def search_for_products_and_parse(self,
                                     user_query: str,
                                     target_sites: List[str],
                                     min_price: Optional[Union[int, float]] = None,
                                     max_price: Optional[Union[int, float]] = None,
                                     max_results_per_site: int = 3) -> List[ProductData]:
        if not self.tavily: logger.error("Tavily is not initialized."); return []
        if not self.firecrawl_client: logger.error("Firecrawl client is not initialized."); return []

        unique_urls = set()
        for site_key in target_sites:
            site_name = site_key.lower()
            tavily_search_query = self._construct_tavily_query(user_query, site_name, min_price, max_price)
            try:
                logger.info(f"Searching Tavily for '{tavily_search_query}' on site '{site_name}'")
                search_results_raw = self.tavily.search_products(query=tavily_search_query, max_results=max_results_per_site * 2)
                if search_results_raw:
                    logger.info(f"Tavily found {len(search_results_raw)} results for query targeting {site_name}.")
                    count = 0
                    for res in search_results_raw:
                        url = res.get('url')
                        if url:
                            domain = urlparse(url).netloc.lower()
                            if site_name == "amazon" and "amazon.in" not in domain and "amazon.com" not in domain : continue
                            if site_name == "flipkart" and "flipkart.com" not in domain: continue
                            if site_name == "myntra" and "myntra.com" not in domain: continue
                            if url not in unique_urls:
                                unique_urls.add(url)
                                count += 1
                                if count >= max_results_per_site: break
            except Exception as e:
                logger.error(f"Error during Tavily search for site {site_name} with query '{tavily_search_query}': {e}")

        final_urls_to_scrape = list(unique_urls)
        if not final_urls_to_scrape: logger.info("No relevant URLs found."); return []
        logger.info(f"Found {len(final_urls_to_scrape)} unique URLs to scrape: {final_urls_to_scrape}")
        parsed_products = self.scrape_and_parse_urls(final_urls_to_scrape, user_query)
        logger.info(f"Total parsed products: {len(parsed_products)}")
        return parsed_products
    
    def export_data(self, data: List[ProductData], format: str = "json", filename: str = None):
        if not filename: filename = f"scraped_data_{int(time.time())}"
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
                    for product in data: writer.writerow(product.__dict__)
        logger.info(f"Data exported to {filename}.{format}")

def format_products_as_text(products: List[ProductData]) -> str:
    if not products: return "No products found."
    valid_products = [p for p in products if p.title and p.product_url]
    if not valid_products: return "No valid product data to format."
    formatted_text = f"Found {len(valid_products)} products:\n\n"
    for i, product in enumerate(valid_products, 1):
        formatted_text += f"--- Product {i} ---\n"
        formatted_text += f"Title: {product.title}\n"
        if product.price: formatted_text += f"Price: {product.price}\n"
        if product.rating:
            formatted_text += f"Rating: {product.rating}"
            if product.reviews_count and product.reviews_count != "0": formatted_text += f" ({product.reviews_count} reviews)\n"
            else: formatted_text += "\n"
        if product.availability: formatted_text += f"Availability: {product.availability}\n"
        if product.seller: formatted_text += f"Seller: {product.seller}\n"
        if product.platform: formatted_text += f"Platform: {product.platform}\n"
        if product.product_url: formatted_text += f"URL: {product.product_url}\n"
        if product.description: formatted_text += f"Description: {product.description}\n"
        formatted_text += "\n"
    return formatted_text

def fetch_product_information_tavily(query: str, platforms: Optional[List[str]] = None, min_price: Optional[Union[int, float]] = None, max_price: Optional[Union[int, float]] = None, max_results_per_platform: int = 3) -> List[ProductData]:
    """
    High-level function to fetch product information using Tavily, Firecrawl, and LLM parsing.
    Returns a list of ProductData objects.
    """
    if not TAVILY_API_KEY:
        logger.error("Tavily API key is not available.")
        return []
    if not FIRECRAWL_API_KEY:
        logger.error("Firecrawl API key is not available.")
        return []

    logger.info(f"Fetching product information for query: '{query}', platforms: {platforms}, price_range: {min_price}-{max_price}")
    manager = ScraperPoolManager(tavily_key=TAVILY_API_KEY, firecrawl_key=FIRECRAWL_API_KEY)

    target_platforms = platforms or ['amazon', 'flipkart', 'myntra']

    product_data_list = manager.search_for_products_and_parse( # Corrected function name
        user_query=query,
        target_sites=target_platforms,
        min_price=min_price,
        max_price=max_price,
        max_results_per_site=max_results_per_platform
    )

    if not product_data_list:
        logger.info(f"No product data found by ScraperPoolManager for query: '{query}'.")
        return []

    logger.info(f"Returning {len(product_data_list)} products for query: '{query}'.")
    return product_data_list

def search_products( # This function seems to be the main entry point for general searches now
    query: str,
    platform_str: str = "amazon,flipkart,myntra", # Retaining for compatibility if used elsewhere
    min_price: Optional[Union[int, float]] = None,
    max_price: Optional[Union[int, float]] = None,
    max_results_per_platform: int = 3
    ) -> List[ProductData]: # Ensure return type consistency
    """
    Search for products using the query and return a list of ProductData objects.
    'platform_str' can be a comma-separated list of platforms.
    This function directly calls fetch_product_information_tavily which is the core logic.
    """
    logger.info(f"search_products called: query='{query}', platforms='{platform_str}', min_price={min_price}, max_price={max_price}")

    platform_list = [p.strip().lower() for p in platform_str.split(',') if p.strip()]
    if not platform_list: # Default to all supported sites if none specified or empty string
        platform_list = ['amazon', 'flipkart', 'myntra']

    # This function now directly calls the refined fetch_product_information_tavily
    return fetch_product_information_tavily(
        query=query,
        platforms=platform_list,
        min_price=min_price,
        max_price=max_price,
        max_results_per_platform=max_results_per_platform
    )

def scrape_website(query_or_url: str) -> str: # This returns formatted text string
    """
    If it's a URL, it tries to detect platform, scrape with Firecrawl, parse with LLM, and format as text.
    If it's a query, it uses the 'search_products' flow (which now returns List[ProductData]) and formats as text.
    """
    manager = ScraperPoolManager(tavily_key=TAVILY_API_KEY, firecrawl_key=FIRECRAWL_API_KEY)
    parsed_url = urlparse(query_or_url)

    if parsed_url.scheme and parsed_url.netloc: # It's a URL
        logger.info(f"scrape_website called with URL: {query_or_url}. Attempting direct scrape and parse.")
        if not manager.firecrawl_client:
            return "Error: Firecrawl client not configured for scraping."

        product_data_list = manager.scrape_and_parse_urls(
            [query_or_url],
            user_query=f"details for product at {query_or_url}" # Generic query for direct URL
        )
        return format_products_as_text(product_data_list)
    else: # It's a search query
        logger.info(f"scrape_website called with query: {query_or_url}. Using search_products flow.")
        # search_products now returns List[ProductData]
        # We need to pass platform_str, min_price, max_price if they are meant to be used here.
        # For simplicity, using defaults for search_products if not provided to scrape_website.
        product_data_list = search_products(query=query_or_url)
        return format_products_as_text(product_data_list)

# Compatibility functions - these might not be needed if frontend/routers directly use the new functions
def extract_body_content(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    return body.get_text() if body else ""

def clean_body_content(content: str) -> str:
    cleaned = ' '.join(content.split())
    return cleaned

def split_dom_content(content: str, max_length: int = 6000) -> List[str]:
    if len(content) <= max_length: return [content]
    chunks = []
    for i in range(0, len(content), max_length): chunks.append(content[i:i + max_length])
    return chunks
