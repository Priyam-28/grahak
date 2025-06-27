from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import logging
from utils.scraper import scrape_website, extract_body_content, clean_body_content, split_dom_content
from utils.parser import parse_with_ollama

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/findproduct", tags=["findproduct"])

class ScrapeRequest(BaseModel):
    url: str

class ParseRequest(BaseModel):
    url: str
    query: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    url: str
    query: str

async def safe_scrape_and_parse(url: str, query: str):
    """Safely scrape and parse with improved error handling and longer timeouts"""
    try:
        logger.info(f"Starting scrape for URL: {url}")
        
        # Increased timeout for scraping
        html_content = await asyncio.wait_for(
            asyncio.to_thread(scrape_website, url), 
            timeout=45.0  # Increased from 30s
        )
        
        logger.info("Scraping completed, extracting content...")
        body_content = extract_body_content(html_content)
        cleaned_content = clean_body_content(body_content)
        
        if not cleaned_content.strip():
            return "No content found on the website."
        
        logger.info(f"Content extracted, splitting into chunks...")
        dom_chunks = split_dom_content(cleaned_content)
        logger.info(f"Split into {len(dom_chunks)} chunks, starting AI parsing...")
        
        # Significantly increased timeout for AI parsing
        parsed_result = await asyncio.wait_for(
            asyncio.to_thread(parse_with_ollama, dom_chunks, query),
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
        return "Request timed out. The website might be too large or slow. Please try with a more specific query."
    except Exception as e:
        logger.error(f"Error in scrape_and_parse: {str(e)}")
        return f"Error processing request: {str(e)}"

@router.post("/scrape")
async def scrape_url(request: ScrapeRequest):
    """Scrape a website and return cleaned content"""
    try:
        html_content = await asyncio.wait_for(
            asyncio.to_thread(scrape_website, request.url),
            timeout=45.0  # Increased timeout
        )
        body_content = extract_body_content(html_content)
        cleaned_content = clean_body_content(body_content)
        
        return {
            "success": True,
            "content": cleaned_content[:5000],  # Limit content size
            "message": "Website scraped successfully"
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout - website took too long to respond")
    except Exception as e:
        logger.error(f"Error scraping website: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scraping website: {str(e)}")

@router.post("/parse")
async def parse_content(request: ParseRequest):
    """Scrape website and parse for specific products"""
    try:
        logger.info(f"Parse request received for URL: {request.url}")
        result = await safe_scrape_and_parse(request.url, request.query)
        
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
        if not request.url or not request.query:
            raise HTTPException(status_code=400, detail="URL and query are required")
        
        logger.info(f"Chat request received for URL: {request.url} with query: {request.query}")
        
        # Process the request with longer timeout
        result = await safe_scrape_and_parse(request.url, request.query)
        
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
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Starting website scrape...'})}\n\n"
            
            # Scrape website with progress updates
            html_content = await asyncio.wait_for(
                asyncio.to_thread(scrape_website, request.url),
                timeout=45.0
            )
            
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Website scraped successfully, extracting content...'})}\n\n"
            
            body_content = extract_body_content(html_content)
            cleaned_content = clean_body_content(body_content)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Content extracted, preparing for AI analysis...'})}\n\n"
            
            dom_chunks = split_dom_content(cleaned_content)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': f'Content split into {len(dom_chunks)} chunks, starting AI analysis...'})}\n\n"
            
            # Process with AI
            parsed_result = await asyncio.wait_for(
                asyncio.to_thread(parse_with_ollama, dom_chunks, request.query),
                timeout=180.0
            )
            
            yield f"data: {json.dumps({'status': 'complete', 'response': parsed_result})}\n\n"
            
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Request timed out. Website might be too large or slow.'})}\n\n"
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