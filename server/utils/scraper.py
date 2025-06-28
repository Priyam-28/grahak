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
import os # Added for environment variables


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load SerpAPI key from environment variable
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    logger.warning("SERPAPI_API_KEY environment variable not found. SerpAPI calls will fail.")
    # You might want to raise an error here or handle it depending on your application's needs
    # For now, we'll let it proceed, but SerpAPIIntegration will likely fail if used.

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


class BasicScrapingStrategy(SiteScrapingStrategy):
    """Basic scraping strategy for generic e-commerce sites."""
    def __init__(self):
        config = ScrapingConfig(
            site_name="generic",
            # No specific selectors, will try generic tags
            delay_range=(1, 2)
        )
        super().__init__(config)

    def extract_data(self, soup: BeautifulSoup, url: str) -> ProductData:
        product = ProductData(platform="Unknown Ecommerce", product_url=url)

        # Try to get title
        if soup.title and soup.title.string:
            product.title = soup.title.string.strip()

        # Try to get meta description as a fallback for description
        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description and meta_description.get("content"):
            product.description = meta_description.get("content").strip()
        else: # Fallback to first few paragraphs if no meta description
            paragraphs = soup.find_all("p")
            text_content = ""
            for p in paragraphs[:3]: # Get first 3 paragraphs
                text_content += p.get_text(strip=True) + " "
            product.description = text_content.strip()[:500] # Limit length

        # Price and image_url are unlikely to be found reliably without specific selectors
        # product.price = ""
        # product.image_url = ""

        logger.info(f"Basic scraping for {url} extracted title: '{product.title}' and a description snippet.")
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

    def get_ecommerce_links_from_query(self, query: str, num_results: int = 10) -> List[str]:
        """
        Perform a general Google search and attempt to extract e-commerce product page URLs.
        """
        if not self.api_key:
            logger.error("SerpAPI key not provided for general search.")
            return []
        try:
            logger.info(f"Performing general Google search for query: {query} to find e-commerce links.")
            search_params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": num_results * 2,  # Fetch more results to have a better chance after filtering
            }
            search = GoogleSearch(search_params)
            results = search.get_dict()

            organic_results = results.get("organic_results", [])
            extracted_links = []

            # Keywords and domain parts that often indicate e-commerce product pages
            # This list can be expanded.
            ecommerce_indicators = ["product", "item", "detail", "/p/", "/dp/"]
            common_ecommerce_domains = [
                "amazon.", "ebay.", "walmart.", "target.", "bestbuy.",
                "etsy.", "flipkart.", "myntra.", "rakuten.", "newegg.",
                "homedepot.", "lowes.", "costco."
            ] # Add more regional ones if needed

            for res in organic_results:
                link = res.get("link")
                title = res.get("title", "").lower()
                snippet = res.get("snippet", "").lower()

                if not link:
                    continue

                # Check 1: Domain known for e-commerce
                is_known_ecommerce_domain = any(domain_part in link for domain_part in common_ecommerce_domains)

                # Check 2: URL path/query parameters indicating product page
                has_ecommerce_path = any(indicator in link for indicator in ecommerce_indicators)

                # Check 3: Title/snippet containing terms like "buy", "price", "shop", "product"
                # (More prone to false positives, use with caution or stricter logic)
                # For now, prioritize domain and path structure.

                if is_known_ecommerce_domain and has_ecommerce_path:
                    if link not in extracted_links: # Avoid duplicates
                        extracted_links.append(link)
                        logger.debug(f"Found potential e-commerce link: {link}")
                elif is_known_ecommerce_domain and len(extracted_links) < num_results: # If it's a known domain but path is unclear, still consider if we need more links
                    if link not in extracted_links:
                         extracted_links.append(link)
                         logger.debug(f"Found link from known e-commerce domain (less specific path): {link}")


                if len(extracted_links) >= num_results:
                    break

            logger.info(f"Extracted {len(extracted_links)} potential e-commerce links for query '{query}'.")
            return extracted_links

        except Exception as e:
            logger.error(f"Error in get_ecommerce_links_from_query: {e}")
            return []


class ScraperPoolManager:
    """Main scraper pool manager"""
    
    def __init__(self, serpapi_key: Optional[str] = SERPAPI_API_KEY, max_workers: int = 5):
        self.proxy_manager = ProxyManager()
        self.fingerprint_manager = BrowserFingerprintManager()
        if not serpapi_key:
            logger.error("SerpAPI key is not configured. ScraperPoolManager cannot use SerpAPI.")
            self.serpapi = None
        else:
            self.serpapi = SerpAPIIntegration(serpapi_key)
        self.max_workers = max_workers
        
        # Initialize scraping strategies
        self.strategies = {
            'amazon': AmazonScrapingStrategy(),
            'flipkart': FlipkartScrapingStrategy(),
            'myntra': MyntraScrapingStrategy(),
            'generic': BasicScrapingStrategy() # Added BasicScrapingStrategy
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

    def get_product_data_from_serpapi_result(self, result: Dict, platform_hint: str) -> Optional[ProductData]:
        """
        Extracts ProductData from a single SerpAPI result item.
        This is useful for engines like Google Shopping that return structured data.
        """
        title = result.get("title")
        price = result.get("price")
        product_url = result.get("link")
        image_url = result.get("thumbnail") # or result.get("image")
        source = result.get("source") # Retailer/Seller
        rating = result.get("rating")
        reviews_count = result.get("reviews")
        description = result.get("description") # Often a snippet

        if not title or not product_url:
            return None

        return ProductData(
            title=str(title),
            price=str(price) if price else "",
            rating=str(rating) if rating else "",
            image_url=str(image_url) if image_url else "",
            description=str(description) if description else "",
            availability="", # Typically not in shopping results directly, might need scraping
            reviews_count=str(reviews_count) if reviews_count else "",
            seller=str(source) if source else "",
            product_url=str(product_url),
            platform=source or platform_hint, # Use source if available, else the hint
            raw_data=result
        )

    def search_and_optionally_scrape(self, query: str, platforms: Optional[List[str]] = None, max_results_per_platform: int = 5) -> List[ProductData]:
        """
        Search using SerpAPI. If SerpAPI provides enough structured data (e.g., Google Shopping), use that.
        Otherwise, fall back to scraping URLs obtained from SerpAPI (e.g., Amazon search results).
        """
        if not self.serpapi:
            logger.error("SerpAPI is not initialized in ScraperPoolManager.")
            return []

        platforms = platforms or ['google_shopping', 'amazon']
        all_product_data: List[ProductData] = []
        urls_to_scrape: List[str] = []

        for platform in platforms:
            try:
                search_results_raw = []
                platform_for_data = platform
                if platform.lower() == 'amazon':
                    # SerpAPI's Amazon engine gives search results that usually need further scraping
                    search_results_raw = self.serpapi.search_amazon(query, num=max_results_per_platform * 2) # Get more results as some might be ads/irrelevant
                    platform_for_data = "Amazon"
                elif platform.lower() == 'google_shopping':
                    search_results_raw = self.serpapi.search_google_shopping(query, num=max_results_per_platform)
                    platform_for_data = "GoogleShopping" # Will be overridden by 'source' if present
                else:
                    logger.warning(f"Unsupported platform for SerpAPI search: {platform}")
                    continue

                logger.info(f"SerpAPI found {len(search_results_raw)} results for '{query}' on {platform}")

                temp_platform_products = []
                for res in search_results_raw:
                    # For Google Shopping, try to extract structured data directly
                    if platform.lower() == 'google_shopping':
                        product = self.get_product_data_from_serpapi_result(res, platform_for_data)
                        if product:
                            temp_platform_products.append(product)
                    # For Amazon (and others if direct data extraction fails), collect URLs
                    else: # e.g., Amazon
                        url = res.get('link') or res.get('product_link')
                        # Basic filter for Amazon: ensure it's a product URL
                        if url and 'amazon' in urlparse(url).netloc.lower() and ('/dp/' in url or '/gp/product/' in url):
                             # Check for duplicates before adding
                            if not any(existing_url == url for existing_url in urls_to_scrape):
                                urls_to_scrape.append(url)
                        elif url and platform.lower() != 'amazon': # For other platforms if we decide to scrape them
                             if not any(existing_url == url for existing_url in urls_to_scrape):
                                urls_to_scrape.append(url)

                all_product_data.extend(temp_platform_products[:max_results_per_platform])


            except Exception as e:
                logger.error(f"Error during SerpAPI search for {platform}: {e}")

        # Scrape URLs collected (mostly for Amazon or if Google Shopping results were just links)
        # Deduplicate URLs before scraping
        unique_urls_to_scrape = list(set(urls_to_scrape))
        logger.info(f"Attempting to scrape {len(unique_urls_to_scrape)} unique URLs.")

        # Limit the number of URLs to scrape to avoid excessive requests
        # Prioritize URLs based on some logic if necessary, here just taking the first N
        # This limit should be considered along with max_results_per_platform
        # For instance, if we want total 10 products, and got 3 from GShopping, we might scrape up to 7 URLs.
        # Here, we'll apply a simpler limit for now.
        effective_scrape_limit = max(0, (max_results_per_platform * len(platforms)) - len(all_product_data))

        if unique_urls_to_scrape and effective_scrape_limit > 0:
            scraped_data = self.scrape_urls(unique_urls_to_scrape[:effective_scrape_limit])
            all_product_data.extend(scraped_data)
            # Ensure we don't exceed total max_results_per_platform * num_platforms approximately
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


# Example usage
def main():
    # Initialize the scraper pool manager
    manager = ScraperPoolManager(
        serpapi_key="5aa1d8da7808e4a13373aa32b6e1a4474fe3202fc6529b57acf6604737fa14a4",  # Replace with actual key
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
    # Use the new method: search_and_optionally_scrape
    search_results = manager.search_and_optionally_scrape("wireless headphones", platforms=["google_shopping", "amazon"], max_results_per_platform=3)
    
    # Export results
    manager.export_data(search_results, format="json", filename="headphones_data")
    manager.export_data(search_results, format="csv", filename="headphones_data")

# This function is being replaced by the new fetch_product_information_serpapi and the updated search_products
# def format_products_as_text(products: List[ProductData]) -> str:
#     """
#     Format product data as readable text
#     """
#     if not products:
#         return "No products found."
    
#     formatted_text = f"Found {len(products)} products:\n\n"

#     for i, product in enumerate(products, 1):
#         formatted_text += f"--- Product {i} ---\n"
#         formatted_text += f"Title: {product.title}\n"

#         if product.price:
#             formatted_text += f"Price: {product.price}\n"
        
#         if product.rating:
#             formatted_text += f"Rating: {product.rating}"
#             if product.reviews_count:
#                 formatted_text += f" ({product.reviews_count})"
#             formatted_text += "\n"
        
#         if product.availability:
#             formatted_text += f"Availability: {product.availability}\n"
        
#         if product.platform:
#             formatted_text += f"Platform: {product.platform}\n"
        
#         if product.product_url:
#             formatted_text += f"URL: {product.product_url}\n"

#         formatted_text += "\n"
    
#     return formatted_text

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
            if product.reviews_count and product.reviews_count != "0": # Avoid " (0)"
                formatted_text += f" ({product.reviews_count} reviews)\n"
            else:
                formatted_text += "\n" # End line if no review count
        
        if product.availability:
            formatted_text += f"Availability: {product.availability}\n"
        
        if product.seller:
            formatted_text += f"Seller: {product.seller}\n"

        if product.platform:
            formatted_text += f"Platform: {product.platform}\n"
        
        if product.product_url:
            formatted_text += f"URL: {product.product_url}\n"
        
        if product.description: # Added description to output
            formatted_text += f"Description: {product.description}\n"

        formatted_text += "\n"
    
    return formatted_text

def fetch_product_information_serpapi(query: str, platforms: Optional[List[str]] = None, max_results_per_platform: int = 3) -> str:
    """
    High-level function to fetch product information using SerpAPI and format it.
    This is intended to be the primary function called by the router.
    """
    if not SERPAPI_API_KEY:
        logger.error("SerpAPI key is not available. Cannot fetch product information.")
        return "Error: SerpAPI key not configured."

    logger.info(f"Fetching product information for query: '{query}' using SerpAPI.")
    manager = ScraperPoolManager(serpapi_key=SERPAPI_API_KEY)

    product_data_list = manager.search_and_optionally_scrape(
        query,
        platforms=platforms or ['google_shopping', 'amazon'],
        max_results_per_platform=max_results_per_platform
    )

    if not product_data_list:
        logger.info(f"No product data found by ScraperPoolManager for query: '{query}'.")
        return "No products found matching your query."

    formatted_text = format_products_as_text(product_data_list)
    logger.info(f"Formatted product information for query: '{query}'. Length: {len(formatted_text)}")
    return formatted_text


# Updated search_products to use the new SerpAPI powered fetch function
def search_products(query: str, platform: str = "google_shopping,amazon", max_results: int = 3) -> str:
    """
    Search for products using the query and return formatted results.
    'platform' can be a comma-separated list of platforms like 'google_shopping,amazon'.
    This function now primarily uses fetch_product_information_serpapi.
    The 'max_results' here means max_results_per_platform.
    """
    logger.info(f"search_products (new) called for query: '{query}', platform(s): '{platform}'")

    platform_list = [p.strip().lower() for p in platform.split(',') if p.strip()]
    if not platform_list:
        platform_list = ['google_shopping', 'amazon']

    return fetch_product_information_serpapi(query, platforms=platform_list, max_results_per_platform=max_results)


# Legacy function names for backward compatibility - these might need review if they are still used elsewhere directly
def scrape_website(query_or_url: str) -> str:
    """
    Backward compatibility function.
    If it's a URL, it tries to detect platform and scrape.
    If it's a query, it uses the new search_products flow.
    """
    parsed_url = urlparse(query_or_url)
    if parsed_url.scheme and parsed_url.netloc:
        logger.info(f"scrape_website called with URL: {query_or_url}. Attempting direct scrape.")
        if not SERPAPI_API_KEY:
             logger.error("SerpAPI key is not available. Cannot initialize ScraperPoolManager for direct scraping.")
             return "Error: SerpAPI key not configured for scraping."
        manager = ScraperPoolManager(serpapi_key=SERPAPI_API_KEY)
        # Note: scrape_urls expects a list of URLs.
        product_data = manager.scrape_urls([query_or_url])
        return format_products_as_text(product_data)
    else:
        logger.info(f"scrape_website called with query: {query_or_url}. Using new search_products.")
        return search_products(query_or_url) # platform and max_results will use defaults


def extract_body_content(html_content: str) -> str:
    """
    Extract body content from HTML (for compatibility)
    """
    return html_content


def clean_body_content(content: str) -> str:
    """
    Clean content (for compatibility)
    """
    return content


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


if __name__ == "__main__":
    main()