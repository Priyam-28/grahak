from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import logging
import uuid
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel
from utils.session_manager import SessionManager, SessionData
from utils.web_search_handler import fetch_and_scrape_tavily_results
from utils.llm_formatter import generate_formatted_response
from utils.scraper import SerpAPIIntegration, ScraperPoolManager, ProductData, SERPAPI_API_KEY
from utils.parser import get_llm_summary, stream_llm_summary, parse_with_ollama


# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/findproduct", tags=["findproduct"])

# --- Global Session Manager ---
# This will keep session data in memory as long as the server process is alive.
# For production, a more persistent session backend or distributed cache might be needed.
session_manager = SessionManager()

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = [] # Full chat history if provided by client
    query: str                               # The user's current query/message
    session_id: Optional[str] = None         # Client can send a session_id to maintain context

# --- Helper to convert ProductData to Dict for API response ---
def product_data_to_dict(p_data: ProductData) -> Dict[str, Any]:
    return {
        "title": p_data.title,
        "price": p_data.price,
        "image_url": p_data.image_url,
        "product_url": p_data.product_url,
        "platform": p_data.platform,
        "description": (p_data.description[:150] + '...' if p_data.description and len(p_data.description) > 150 else p_data.description),
        "rating": p_data.rating,
        # "raw_data": p_data.raw_data # Usually not needed for frontend
    }

# --- New Core Logic Function using Session, ChromaDB, Tavily, and LLM Formatter ---
async def process_chat_query_with_rag(
    user_query: str,
    session_id_str: str,
    num_tavily_results_to_fetch: int = 7,
    num_urls_to_scrape_from_tavily: int = 3, # Scrape fewer for speed
    num_chroma_results_for_context: int = 5,
    num_final_products_to_feature: int = 6
) -> Dict[str, Any]:

    logger.info(f"Processing query '{user_query}' for session '{session_id_str}'")

    # 1. Get or create session
    session_data: SessionData = session_manager.get_or_create_session(session_id_str)
    current_chat_history = list(session_data.chat_history) # Make a copy

    # 2. RAG Step 1: Query existing session data in ChromaDB
    logger.info(f"Session {session_id_str}: Querying ChromaDB for relevant existing products.")
    products_from_chroma: List[ProductData] = await asyncio.to_thread(
        session_manager.query_session_chroma,
        session_data,
        user_query,
        k=num_chroma_results_for_context
    )
    logger.info(f"Session {session_id_str}: Found {len(products_from_chroma)} products in ChromaDB.")

    # 3. Decision for Web Search (Simplified: always search for now to get freshest info / augment)
    #    A more advanced agent would make this decision. Here, we combine.
    logger.info(f"Session {session_id_str}: Performing Tavily web search for query '{user_query}'.")
    newly_scraped_products: List[ProductData] = await asyncio.to_thread(
        fetch_and_scrape_tavily_results,
        user_query,
        num_tavily_results=num_tavily_results_to_fetch,
        num_urls_to_scrape=num_urls_to_scrape_from_tavily
    )
    logger.info(f"Session {session_id_str}: Scraped {len(newly_scraped_products)} new products from Tavily search.")

    # 4. Add newly scraped products to session (ChromaDB and cache)
    if newly_scraped_products:
        # This is a blocking call internally due to Chroma/embeddings, so run in thread
        await asyncio.to_thread(session_manager.add_products_to_session, session_data, newly_scraped_products)

    # 5. Combine products for LLM context
    #    Re-query Chroma to get the most relevant combined list after potential additions.
    #    This ensures products are ranked by relevance to the current query.
    logger.info(f"Session {session_id_str}: Re-querying ChromaDB for combined product context.")
    all_relevant_products_for_llm: List[ProductData] = await asyncio.to_thread(
        session_manager.query_session_chroma,
        session_data,
        user_query,
        k=num_chroma_results_for_context + len(newly_scraped_products) # Get a bit more
    )
    # Ensure we have a diverse set, limit to a reasonable number for the LLM prompt
    all_relevant_products_for_llm = all_relevant_products_for_llm[:10]
    logger.info(f"Session {session_id_str}: Using {len(all_relevant_products_for_llm)} products for LLM context.")


    # 6. Generate Formatted Response using LLM
    logger.info(f"Session {session_id_str}: Generating LLM response.")
    # This call is also blocking internally (LLM inference)
    llm_text, featured_product_objects = await asyncio.to_thread(
        generate_formatted_response,
        user_query,
        current_chat_history, # Pass current history
        all_relevant_products_for_llm # Pass combined list for LLM to select from
    )

    # 7. Update Chat History
    await asyncio.to_thread(session_manager.add_to_chat_history, session_data, user_query, llm_text)

    # 8. Prepare final list of products for API response (those featured by LLM)
    #    Limit the number of products actually sent to frontend.
    final_product_list_for_api = [
        product_data_to_dict(p) for p in featured_product_objects[:num_final_products_to_feature]
    ]

    return {
        "llm_summary": llm_text,
        "products": final_product_list_for_api,
        "session_id": session_id_str # Return session_id so client can maintain it
    }

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

@router.post("/chat")
async def chat_with_products_rag(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(None) # Allow session ID via header
):
    """
    Chat interface for finding products using RAG with Tavily and ChromaDB.
    Returns LLM summary and a list of products.
    Manages session context and chat history.
    """
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Search query is required")
        
        # Determine session ID: Use from request body, then header, or create new
        session_id_to_use = request.session_id or x_session_id
        if not session_id_to_use:
            session_id_to_use = f"session_{uuid.uuid4().hex}" # Generate a new one if none provided
            logger.info(f"No session_id provided, generated new one: {session_id_to_use}")
        
        logger.info(f"Chat request for query: '{request.query}', Session ID: {session_id_to_use}")
        
        processed_data = await process_chat_query_with_rag(request.query, session_id_to_use)
        
        return {
            "success": True,
            "llm_summary": processed_data.get("llm_summary"),
            "products": processed_data.get("products", []),
            "session_id": processed_data.get("session_id"), # Send back the session_id used
            "message": "Products found successfully"
        }
        
    except asyncio.CancelledError:
        # Handle client disconnection gracefully
        logger.info("Client disconnected during processing")
        # For FastAPI, if the client disconnects, the request is cancelled,
        # and FastAPI handles sending an appropriate response or just closing the connection.
        # We don't need to return a JSON response here if the connection is already gone.
        # The 'except asyncio.CancelledError' handles this.
        pass # Pass here, error is already logged.
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
        # This case should ideally be caught by client-side validation too.
        # For streaming, we can send an error event and then close.
        async def error_stream():
            yield f"event: error\ndata: {json.dumps({'error': 'Search query is required'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def stream_generator():
        # Initial status update
        yield f"event: status\ndata: {json.dumps({'status': 'processing', 'message': 'Discovering product links...'})}\n\n"

        if not SERPAPI_API_KEY:
            logger.error("SERPAPI_API_KEY is not configured for streaming.")
            yield f"event: llm_token\ndata: {json.dumps({'token': 'Error: API key for searching is not configured.'})}\n\n"
            yield f"event: products\ndata: {json.dumps([])}\n\n" # Send empty products list
            yield f"event: end\ndata: Stream ended due to configuration error.\n\n"
            return

        try:
            # Step A: URL Discovery (Non-streamed part, happens first)
            serp_integration = SerpAPIIntegration(api_key=SERPAPI_API_KEY)
            ecommerce_urls = await asyncio.to_thread(
                serp_integration.get_ecommerce_links_from_query, user_query, num_results=7
            )

            scraped_products_data: List[ProductData] = []
            if not ecommerce_urls:
                logger.warning(f"No e-commerce URLs found by SerpAPI general search for query: '{user_query}'")
                yield f"event: status\ndata: {json.dumps({'status': 'processing', 'message': 'No direct e-commerce links found, trying fallback shopping search...'})}\n\n"
                shopping_results_raw = await asyncio.to_thread(serp_integration.search_google_shopping, user_query, num=10)
                raw_product_objects = [
                    serp_integration.get_product_data_from_serpapi_result(res, "Google Shopping")
                    for res in shopping_results_raw if res
                ]
                scraped_products_data = [p for p in raw_product_objects if p]
            else:
                yield f"event: status\ndata: {json.dumps({'status': 'processing', 'message': f'Found {len(ecommerce_urls)} links, now scraping...'})}\n\n"
                # Step B: Parallel Scraping (Non-streamed part)
                scraper_manager = ScraperPoolManager(serpapi_key=SERPAPI_API_KEY)
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