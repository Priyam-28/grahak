import asyncio
from crawl4ai import *

# async def main():
#     async with AsyncWebCrawler() as crawler:
#         result = await crawler.arun(
#             url="https://www.amazon.in/s?k=t+shirt",
#         )
#         print(result.markdown)

# if __name__ == "__main__":
#     asyncio.run(main())
def scrape_website(website, output_format="markdown", wait_for_js=True):
    print("Scraping website with Crawl4AI...")
    
    async def _scrape():
        async with AsyncWebCrawler(verbose=True) as crawler:
            try:
                # Basic crawl with AI-optimized settings
                result = await crawler.arun(
                    url=website,
                    word_count_threshold=10,  # Minimum word count
                    extraction_strategy=None,  # Use default extraction
                    chunking_strategy="sentence",  # How to chunk content
                    bypass_cache=True,  # Always fetch fresh content
                    process_iframes=True,  # Process iframe content
                    remove_overlay_elements=True,  # Remove popups/overlays
                    simulate_user=True if wait_for_js else False,  # Simulate user behavior
                    override_navigator=True,  # Override navigator properties
                    wait_for="css:body",  # Wait for body to load
                    delay_before_return_html=2.0 if wait_for_js else 0,  # Wait for JS
                )
                
                if result.success:
                    print("Website scraped successfully with Crawl4AI!")
                    
                    # Return content based on requested format
                    if output_format == "markdown":
                        return result.markdown
                    elif output_format == "cleaned_html":
                        return result.cleaned_html
                    elif output_format == "html":
                        return result.html
                    elif output_format == "json":
                        return json.dumps({
                            "url": result.url,
                            "title": result.title,
                            "markdown": result.markdown,
                            "cleaned_html": result.cleaned_html,
                            "links": result.links,
                            "media": result.media,
                            "metadata": result.metadata
                        }, indent=2)
                    else:
                        return result.markdown  # Default to markdown
                else:
                    raise Exception(f"Crawl4AI failed: {result.error_message}")
                    
            except Exception as e:
                print(f"Error scraping website with Crawl4AI: {e}")
                raise