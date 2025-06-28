import requests
from bs4 import BeautifulSoup
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from serpapi import GoogleSearch
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import fake_useragent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class SerpAPIIntegration:
    """Integration with SerpAPI for search-based scraping"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def search_products(self, query: str, engine: str = "google_shopping", **kwargs) -> List[Dict]:
        """Search for products using SerpAPI"""
        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.api_key,
                "engine": engine,
                **kwargs
            })
            
            results = search.get_dict()
            return results.get("shopping_results", [])
            
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []
    
    def search_amazon(self, query: str, **kwargs) -> List[Dict]:
        """Search Amazon specifically"""
        return self.search_products(query, engine="amazon", **kwargs)
    
    def search_google_shopping(self, query: str, **kwargs) -> List[Dict]:
        """Search Google Shopping"""
        return self.search_products(query, engine="google_shopping", **kwargs)


class ScraperPoolManager:
    """Main scraper pool manager"""
    
    def __init__(self, serpapi_key: Optional[str] = None, max_workers: int = 5):
        self.proxy_manager = ProxyManager()
        self.fingerprint_manager = BrowserFingerprintManager()
        self.serpapi = SerpAPIIntegration(serpapi_key) if serpapi_key else None
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
        """Search using SerpAPI and scrape the results"""
        if not self.serpapi:
            logger.error("SerpAPI key not provided")
            return []
        
        platforms = platforms or ['amazon', 'google_shopping']
        all_results = []
        
        for platform in platforms:
            try:
                if platform == 'amazon':
                    search_results = self.serpapi.search_amazon(query)
                else:
                    search_results = self.serpapi.search_google_shopping(query)
                
                # Extract URLs from search results
                urls = []
                for result in search_results[:10]:  # Limit to top 10 results
                    url = result.get('link') or result.get('product_link')
                    if url:
                        urls.append(url)
                
                # Scrape the URLs
                scraped_data = self.scrape_urls(urls)
                all_results.extend(scraped_data)
                
            except Exception as e:
                logger.error(f"Error searching {platform}: {e}")
        
        return all_results
    
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


# Example usage
def main():
    # Initialize the scraper pool manager
    manager = ScraperPoolManager(
        serpapi_key="your_serpapi_key_here",  # Replace with actual key
        max_workers=3
    )
    
    # Example 1: Scrape specific URLs
    urls = [
        "https://www.amazon.in/product-example",
        "https://www.flipkart.com/product-example",
        "https://www.myntra.com/product-example"
    ]
    
    print("Scraping specific URLs...")
    results = manager.scrape_urls(urls)
    
    for product in results:
        print(f"Title: {product.title}")
        print(f"Price: {product.price}")
        print(f"Platform: {product.platform}")
        print("-" * 50)
    
    # Example 2: Search and scrape
    print("\nSearching and scraping...")
    search_results = manager.search_and_scrape("wireless headphones", ["amazon", "google_shopping"])
    
    # Export results
    manager.export_data(search_results, format="json", filename="headphones_data")
    manager.export_data(search_results, format="csv", filename="headphones_data")


if __name__ == "__main__":
    main()