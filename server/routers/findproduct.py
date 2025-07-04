from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import logging
from dotenv import load_dotenv
from utils.scraper import ScraperPoolManager, ProductData, search_products, scrape_website
from utils.parser import parse_with_ollama

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/findproduct", tags=["findproduct"])

def format_product_data_for_llm(products: List[ProductData]) -> str:
    """Format product data for LLM input"""
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
            desc_snippet = (p.description[:200] + '...') if len(p.description) > 200 else p.description
            text_parts.append(f"Description: {desc_snippet}")
    return "\n".join(text_parts)

class SearchRequest(BaseModel):
    query: str

class ParseRequest(BaseModel):
    query: str
    description: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    query: str

async def process_product_query(user_query: str, num_urls_to_find: int = 7, max_products_to_return: int = 10):
    """Process product query using the scraper module's functionality"""
    try:
        # Use the existing search_products function from scraper.py
        search_results = await asyncio.to_thread(
            search_products, 
            user_query,
            platform="general_shopping,amazon",
            max_results=num_urls_to_find
        )

        if not search_results:
            logger.warning(f"No results for: '{user_query}'")
            return {"llm_summary": "No products found", "products": []}

        # Get the actual ProductData objects from the search results
        manager = ScraperPoolManager()
        products = []
        if isinstance(search_results, str):
            # If search_results is a string, it's an error message
            return {"llm_summary": search_results, "products": []}
        elif isinstance(search_results, list):
            products = search_results[:max_products_to_return]

        # Generate LLM summary
        logger.info("Generating LLM summary")
        context = format_product_data_for_llm(products)
        llm_summary = await asyncio.wait_for(
            asyncio.to_thread(parse_with_ollama, [context], user_query),
            timeout=180.0
        )

        # Format response
        formatted_products = []
        for p in sorted(products, key=lambda x: (bool(x.image_url and x.price), bool(x.price)), reverse=True)[:max_products_to_return]:
            formatted_products.append({
                "title": p.title,
                "price": p.price,
                "image_url": p.image_url,
                "product_url": p.product_url,
                "platform": p.platform,
                "description": p.description[:150] + '...' if p.description and len(p.description) > 150 else p.description,
                "rating": p.rating
            })
        
        return {
            "llm_summary": llm_summary or "No summary generated",
            "products": formatted_products
        }

    except asyncio.TimeoutError:
        logger.error("Processing timed out")
        return {"llm_summary": "Request timed out", "products": []}
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"llm_summary": f"Error: {str(e)}", "products": []}

@router.post("/chat")
async def chat_with_products(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if not request.query:
            raise HTTPException(400, "Query is required")
        
        logger.info(f"Processing query: {request.query}")
        result = await process_product_query(request.query)
        
        return {
            "success": True,
            "llm_summary": result["llm_summary"],
            "products": result["products"],
            "message": "Products found"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error: {str(e)}")

@router.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    async def generate_events():
        try:
            if not request.query:
                yield f"event: error\ndata: {json.dumps({'error': 'Query required'})}\n\n"
                return

            # Use the existing search functionality from scraper.py
            yield f"event: status\ndata: {json.dumps({'status': 'searching'})}\n\n"
            results = await asyncio.to_thread(
                search_products,
                request.query,
                platform="general_shopping,amazon",
                max_results=7
            )

            if not results or isinstance(results, str):
                yield f"event: error\ndata: {json.dumps({'error': 'No results' if not results else results})}\n\n"
                return

            # Get ProductData objects
            products = results[:10]  # Limit to 10 products

            # LLM processing
            yield f"event: status\ndata: {json.dumps({'status': 'processing'})}\n\n"
            context = format_product_data_for_llm(products)
            
            async for token in stream_llm_summary(context, request.query):
                yield f"event: llm_token\ndata: {json.dumps({'token': token})}\n\n"

            # Final products
            product_list = []
            for p in products:
                product_list.append({
                    "title": p.title,
                    "price": p.price,
                    "image_url": p.image_url,
                    "product_url": p.product_url,
                    "platform": p.platform,
                    "description": p.description[:100] + '...' if p.description and len(p.description) > 100 else p.description,
                    "rating": p.rating
                })
            
            yield f"event: products\ndata: {json.dumps(product_list)}\n\n"
            yield f"event: end\ndata: done\n\n"

        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

async def stream_llm_summary(context: str, query: str):
    """Stream LLM response tokens"""
    # Implement your LLM streaming here
    # Example mock implementation:
    for word in f"Summary for {query}:".split():
        yield word + " "
        await asyncio.sleep(0.1)
    yield "\nFound products with details..."