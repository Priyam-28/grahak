from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional, Union
import logging

# Assuming ProductData and the main search function are in scraper module
# Adjust import path if your structure is different or if you face circular dependency.
# One way to handle potential circularity if parser also needs ProductData from scraper:
# In scraper, ProductData is defined. In parser, it's imported. Router imports from scraper.
try:
    from server.utils.scraper import search_for_products_and_parse, ProductData
except ImportError:
    # Fallback for local testing if paths are tricky, or if structure changes
    try:
        from ..utils.scraper import search_for_products_and_parse, ProductData
    except ImportError:
        raise ImportError("Could not import search_for_products_and_parse or ProductData from scraper module.")


logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/super-search/products", response_model=List[ProductData])
async def super_search_products(
    query: str,
    sites: List[str] = Query(default=["amazon", "flipkart", "myntra"], description="List of sites to search (e.g., 'amazon', 'flipkart', 'myntra')"),
    min_price: Optional[float] = Query(default=None, description="Minimum price for the product"),
    max_price: Optional[float] = Query(default=None, description="Maximum price for the product"),
    max_results_per_site: int = Query(default=3, ge=1, le=10, description="Maximum number of results to fetch per site")
):
    """
    Performs a super search across specified e-commerce sites (Amazon.in, Flipkart, Myntra),
    scrapes product pages using Firecrawl, and parses data using an LLM.
    """
    logger.info(
        f"Received super search request: query='{query}', sites={sites}, "
        f"min_price={min_price}, max_price={max_price}, max_results_per_site={max_results_per_site}"
    )

    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    if not sites:
        raise HTTPException(status_code=400, detail="Sites list cannot be empty. Provide at least one site.")

    valid_sites = ["amazon", "flipkart", "myntra"] # Add more if supported by scraper
    for site in sites:
        if site.lower() not in valid_sites:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid site '{site}'. Supported sites are: {', '.join(valid_sites)}"
            )

    try:
        # Call the main function from the scraper utility
        # Note: The scraper's search_for_products_and_parse function handles TAVILY_API_KEY and FIRECRAWL_API_KEY internally
        product_list: List[ProductData] = search_for_products_and_parse(
            user_query=query,
            target_sites=sites,
            min_price=min_price,
            max_price=max_price,
            max_results_per_site=max_results_per_site
        )

        if not product_list:
            logger.info(f"No products found for query: '{query}' with specified criteria.")
            # Return empty list, which is valid for response_model=List[ProductData]
            # Or raise HTTPException(status_code=404, detail="No products found matching your criteria.")
            # For now, returning an empty list seems more API-friendly than a 404 for "no results"
            return []

        logger.info(f"Super search successful. Returning {len(product_list)} products.")
        return product_list

    except ImportError as ie:
        logger.error(f"Import error in super_search_products: {ie}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error: A component could not be loaded.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during super search: {e}", exc_info=True)
        # Provide a generic error message to the client
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Example of how to add this router to a FastAPI app in main.py:
# from server.routers import super_search
# app.include_router(super_search.router, prefix="/api/v1", tags=["Super Search"])
