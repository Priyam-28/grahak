from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import logging
# Updated imports for Tavily integration
from utils.scraper import TavilyIntegration, ScraperPoolManager, ProductData, TAVILY_API_KEY
from utils.parser import get_llm_summary, stream_llm_summary, parse_with_ollama

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/findproduct", tags=["findproduct"])

# --- Helper function to format ProductData for LLM ---
def format_product_data_for_llm(products: List[ProductData]) -> str:
    if not products:
        return "No products found."

    text_parts = ["Here is a list of products found:\n"]
    for i, p in enumerate(products):
        text_parts.append(f"\n--- Product {i+1} ---")
        text_parts.append(f"Title: {p.title if p.title else 'N/A'}")
        text_parts.append(f"Price: {p.price if p.price else 'N/A'}")
        if p.platform:
            text_parts.append(f"Source: {p.platform}")
        if p.description:
            # Keep description concise for the LLM context
            desc_snippet = (p.description[:200] + '...') if len(p.description) > 200 else p.description
            text_parts.append(f"Description: {desc_snippet}")
        # Not including image_url or raw_data in the text for LLM
    return "\n".join(text_parts)

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str

class ParseRequest(BaseModel): # Potentially deprecated if not used by new flow
    query: str
    description: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    query: str # The user's current query/message

# --- Core Logic Function (Updated for Tavily) ---
async def process_product_query(user_query: str, num_urls_to_find: int = 7, max_products_to_return: int = 10):
    """
    Processes a user's product query:
    1. Finds relevant e-commerce URLs using Tavily API.
    2. Scrapes these URLs for product data.
    3. Generates an LLM summary based on the scraped data and user query.
    4. Returns the LLM summary and a list of structured product data.
    """
    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY is not configured. Cannot process query.")
        return {"llm_summary": "Error: API key for searching is not configured.", "products": []}

    try:
        # Step A: URL Discovery using Tavily
        logger.info(f"Step A: Discovering e-commerce links for query: '{user_query}'")
        tavily_integration = TavilyIntegration(api_key=TAVILY_API_KEY)
        
        # Run blocking I/O in a separate thread
        ecommerce_urls = await asyncio.to_thread(
            tavily_integration.get_ecommerce_links_from_query,
            user_query,
            num_results=num_urls_to_find
        )

        if not ecommerce_urls:
            logger.warning(f"No e-commerce URLs found by Tavily search for query: '{user_query}'")
            return {"llm_summary": "No products found for your query. Please try a different search term.", "products": []}
        
        # Step B: Scraping URLs
        logger.info(f"Step B: Scraping {len(ecommerce_urls)} URLs: {ecommerce_urls}")
        scraper_manager = ScraperPoolManager(tavily_api_key=TAVILY_API_KEY)
        # Run blocking I/O in a separate thread
        scraped_products = await asyncio.to_thread(scraper_manager.scrape_urls, ecommerce_urls)

        if not scraped_products:
            logger.warning(f"No products successfully scraped from discovered URLs for query: '{user_query}'")
            return {"llm_summary": "Could not retrieve detailed product information from the web.", "products": []}

        logger.info(f"Successfully scraped {len(scraped_products)} products.")

        # Step C: Prepare Context for LLM
        logger.info("Step C: Preparing context for LLM.")
        llm_context_text = format_product_data_for_llm(scraped_products)

        # Step D: LLM Processing
        logger.info(f"Step D: Sending context to LLM for query: '{user_query}'")
        llm_summary = await asyncio.wait_for(
            asyncio.to_thread(get_llm_summary, llm_context_text, user_query),
            timeout=180.0
        )
        logger.info("LLM processing completed.")

        # Step E: Format API Response (products part)
        # Select up to max_products_to_return, prioritizing those with image_url and price
        formatted_products = []
        for p_data in sorted(scraped_products, key=lambda p: (bool(p.image_url and p.price), bool(p.price)), reverse=True)[:max_products_to_return]:
            formatted_products.append({
                "title": p_data.title,
                "price": p_data.price,
                "image_url": p_data.image_url,
                "product_url": p_data.product_url,
                "platform": p_data.platform,
                "description": p_data.description[:150] + '...' if p_data.description and len(p_data.description) > 150 else p_data.description, # Snippet
                "rating": p_data.rating # Added rating
            })
        
        return {"llm_summary": llm_summary if llm_summary else "No specific summary generated.", "products": formatted_products}

    except asyncio.TimeoutError:
        logger.error(f"Request timed out during processing query: {user_query}")
        return {"llm_summary": "Search process timed out. Please try a simpler query.", "products": []}
    except Exception as e:
        logger.error(f"Error in process_product_query for query '{user_query}': {e}", exc_info=True)
        return {"llm_summary": f"An error occurred: {str(e)}", "products": []}

@router.post("/chat")
async def chat_with_products(request: ChatRequest):
    """Chat interface for finding products. Returns LLM summary and a list of products."""
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Search query is required")
        
        logger.info(f"Chat request received for query: {request.query}")
        
        processed_data = await process_product_query(request.query)
        
        return {
            "success": True,
            "llm_summary": processed_data.get("llm_summary"),
            "products": processed_data.get("products", []),
            "message": "Products found successfully"
        }
        
    except asyncio.CancelledError:
        # Handle client disconnection gracefully
        logger.info("Client disconnected during processing")
        pass
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@router.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat response. Streams LLM summary first, then sends a final message with product list.
    """
    user_query = request.query
    if not user_query:
        async def error_stream():
            yield f"event: error\ndata: {json.dumps({'error': 'Search query is required'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def stream_generator():
        # Initial status update
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'message': 'Discovering product links...'})}\n\n"

        if not TAVILY_API_KEY:
            logger.error("TAVILY_API_KEY is not configured for streaming.")
            yield f"event: llm_token\ndata: {json.dumps({'token': 'Error: API key for searching is not configured.'})}\n\n"
            yield f"event: products\ndata: {json.dumps([])}\n\n"
            yield f"event: end\ndata: Stream ended due to configuration error.\n\n"
            return

        try:
            # Step A: URL Discovery using Tavily
            tavily_integration = TavilyIntegration(api_key=TAVILY_API_KEY)
            ecommerce_urls = await asyncio.to_thread(
                tavily_integration.get_ecommerce_links_from_query, user_query, num_results=7
            )

            scraped_products_data: List[ProductData] = []
            if not ecommerce_urls:
                logger.warning(f"No e-commerce URLs found by Tavily search for query: '{user_query}'")
                yield f"event: llm_token\ndata: {json.dumps({'token': 'No products found for your query. Please try a different search term.'})}\n\n"
                yield f"event: products\ndata: {json.dumps([])}\n\n"
                yield f"event: end\ndata: Stream ended, no products found.\n\n"
                return
            else:
                yield f"event: status\ndata: {json.dumps({'status': 'processing', 'message': f'Found {len(ecommerce_urls)} links, now scraping...'})}\n\n"
                # Step B: Parallel Scraping
                scraper_manager = ScraperPoolManager(tavily_api_key=TAVILY_API_KEY)
                scraped_products_data = await asyncio.to_thread(scraper_manager.scrape_urls, ecommerce_urls)

            if not scraped_products_data:
                logger.warning(f"No products successfully scraped for query: '{user_query}'")
                yield f"event: llm_token\ndata: {json.dumps({'token': 'Could not retrieve detailed product information from the web.'})}\n\n"
                yield f"event: products\ndata: {json.dumps([])}\n\n"
                yield f"event: end\ndata: Stream ended, no products found.\n\n"
                return

            yield f"event: status\ndata: {json.dumps({'status': 'processing', 'message': f'Scraped {len(scraped_products_data)} products. Generating summary...'})}\n\n"

            # Step C: Prepare Context for LLM
            llm_context_text = format_product_data_for_llm(scraped_products_data)
            
            # Step D: Stream LLM Processing
            async for token in stream_llm_summary(llm_context_text, user_query):
                yield f"event: llm_token\ndata: {json.dumps({'token': token})}\n\n"
            
            # Step E: Format and send product list
            formatted_products_for_client = []
            for p_data in sorted(scraped_products_data, key=lambda p: (bool(p.image_url and p.price), bool(p.price)), reverse=True)[:10]:
                 formatted_products_for_client.append({
                    "title": p_data.title, "price": p_data.price, "image_url": p_data.image_url,
                    "product_url": p_data.product_url, "platform": p_data.platform,
                    "description": p_data.description[:150] + '...' if p_data.description and len(p_data.description) > 150 else p_data.description,
                    "rating": p_data.rating
                })
            yield f"event: products\ndata: {json.dumps(formatted_products_for_client)}\n\n"
            yield f"event: end\ndata: Stream completed successfully.\n\n"

        except asyncio.TimeoutError:
            logger.error(f"Request timed out during streaming for query: {user_query}")
            yield f"event: error\ndata: {json.dumps({'error': 'Request timed out. Search might be too complex.'})}\n\n"
        except asyncio.CancelledError:
            logger.info(f"Client disconnected during streaming for query: {user_query}")
            # Don't yield anything further if client disconnected.
        except Exception as e:
            logger.error(f"Error in streaming for query '{user_query}': {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': f'Error: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )