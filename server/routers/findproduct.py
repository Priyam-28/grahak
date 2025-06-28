from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import logging
# Updated import: scrape_website, extract_body_content, clean_body_content, split_dom_content are removed
# as their functionality is either integrated into the new search_products or no longer directly used by this router.
from utils.scraper import search_products
from utils.parser import parse_with_ollama

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/findproduct", tags=["findproduct"])

class SearchRequest(BaseModel):
    query: str

class ParseRequest(BaseModel):
    query: str
    description: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    query: str

async def safe_search_and_parse(query: str, description: str):
    """Safely search and parse with improved error handling and longer timeouts"""
    try:
        logger.info(f"Starting search for query: {query}")
        
        # Use the updated search_products function from utils.scraper
        # It now takes platform as a comma-separated string and max_results is per platform.
        # Let's default to 'google_shopping,amazon' and 3 results per platform.
        search_results = await asyncio.wait_for(
            asyncio.to_thread(search_products, query, "google_shopping,amazon", 3),
            timeout=60.0  # Adjusted timeout, SerpAPI calls can take time
        )
        
        logger.info(f"Search completed. Raw results length: {len(search_results)}")
        
        if not search_results or search_results.strip() == "No products found.":
            return "No products found for your search."
        
        logger.info(f"Products found, starting AI parsing...")
        
        # Use the search results directly for AI parsing
        parsed_result = await asyncio.wait_for(
            asyncio.to_thread(parse_with_ollama, [search_results], description),
            timeout=180.0  # Increased from 60s to 3 minutes
        )
        
        logger.info("AI parsing completed")
        
        # Clean up the result
        if isinstance(parsed_result, str):
            cleaned_result = parsed_result.strip()
            if cleaned_result:
                return cleaned_result
            else:
                return "No relevant products found."
        else:
            return str(parsed_result) if parsed_result else "No relevant products found."
        
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        return "Request timed out. Please try with a more specific search query."
    except Exception as e:
        logger.error(f"Error in search_and_parse: {str(e)}")
        return f"Error processing request: {str(e)}"

@router.post("/search")
async def search_products_endpoint(request: SearchRequest):
    """Search for products and return cleaned content"""
    # This endpoint might be less relevant with the new chat flow,
    # but updating it to use the new search_products signature for consistency.
    try:
        search_results = await asyncio.wait_for(
            asyncio.to_thread(search_products, request.query, "google_shopping,amazon", 3), # Updated call
            timeout=60.0  # Adjusted timeout
        )
        
        return {
            "success": True,
            "content": search_results[:5000],  # Limit content size
            "message": "Product search completed successfully"
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout - search took too long to respond")
    except Exception as e:
        logger.error(f"Error searching for products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching for products: {str(e)}")

@router.post("/parse")
async def parse_content(request: ParseRequest):
    """Search for products and parse for specific criteria"""
    try:
        logger.info(f"Parse request received for query: {request.query}")
        result = await safe_search_and_parse(request.query, request.description)
        
        return {
            "success": True,
            "result": result,
            "message": "Content parsed successfully"
        }
    except Exception as e:
        logger.error(f"Error parsing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing content: {str(e)}")

@router.post("/chat")
async def chat_with_products(request: ChatRequest):
    """Chat interface for finding products with improved connection handling"""
    try:
        # Validate input
        if not request.query:
            raise HTTPException(status_code=400, detail="Search query is required")
        
        logger.info(f"Chat request received for query: {request.query}")
        
        # Process the request with longer timeout
        result = await safe_search_and_parse(request.query, request.query)
        
        return {
            "success": True,
            "response": result,
            "message": "Products found successfully"
        }
        
    except asyncio.CancelledError:
        # Handle client disconnection gracefully
        logger.info("Client disconnected during processing")
        return {
            "success": False,
            "response": "Request was cancelled",
            "message": "Client disconnected"
        }
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@router.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat response to handle long operations with progress updates"""
    async def generate_response():
        try:
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Starting product search...'})}\n\n"
            
            # Search for products with progress updates - updated call
            search_results = await asyncio.wait_for(
                asyncio.to_thread(search_products, request.query, "google_shopping,amazon", 3), # Updated call
                timeout=60.0 # Adjusted timeout
            )
            
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Products found successfully, preparing for AI analysis...'})}\n\n"
            
            if not search_results or search_results.strip() == "No products found.":
                yield f"data: {json.dumps({'status': 'complete', 'response': 'No products found for your search.'})}\n\n"
                return
            
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Starting AI analysis...'})}\n\n"
            
            # Process with AI
            parsed_result = await asyncio.wait_for(
                asyncio.to_thread(parse_with_ollama, [search_results], request.query),
                timeout=180.0
            )
            
            yield f"data: {json.dumps({'status': 'complete', 'response': parsed_result})}\n\n"
            
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Request timed out. Search might be too complex.'})}\n\n"
        except asyncio.CancelledError:
            yield f"data: {json.dumps({'status': 'cancelled', 'message': 'Request cancelled by client'})}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'Error: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )