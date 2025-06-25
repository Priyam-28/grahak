from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import logging
import base64
from utils.scraper import scrape_website, extract_body_content, clean_body_content, split_dom_content
from utils.parser import parse_with_ollama
from utils.tts import tts_service

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
    voice_enabled: Optional[bool] = False
    voice_id: Optional[str] = "en-US-terrell"

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "en-US-terrell"

async def safe_scrape_and_parse(url: str, query: str):
    """Safely scrape and parse with error handling"""
    try:
        # Add timeout to prevent hanging
        html_content = await asyncio.wait_for(
            asyncio.to_thread(scrape_website, url), 
            timeout=30.0
        )
        
        body_content = extract_body_content(html_content)
        cleaned_content = clean_body_content(body_content)
        
        if not cleaned_content.strip():
            return "No content found on the website."
        
        dom_chunks = split_dom_content(cleaned_content)
        parsed_result = await asyncio.wait_for(
            asyncio.to_thread(parse_with_ollama, dom_chunks, query),
            timeout=60.0
        )
        
        return parsed_result if parsed_result.strip() else "No relevant products found."
        
    except asyncio.TimeoutError:
        return "Request timed out. Please try again with a simpler query."
    except Exception as e:
        logger.error(f"Error in scrape_and_parse: {str(e)}")
        return f"Error processing request: {str(e)}"

@router.post("/scrape")
async def scrape_url(request: ScrapeRequest):
    """Scrape a website and return cleaned content"""
    try:
        html_content = await asyncio.wait_for(
            asyncio.to_thread(scrape_website, request.url),
            timeout=30.0
        )
        body_content = extract_body_content(html_content)
        cleaned_content = clean_body_content(body_content)
        
        return {
            "success": True,
            "content": cleaned_content[:5000],  # Limit content size
            "message": "Website scraped successfully"
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error scraping website: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scraping website: {str(e)}")

@router.post("/parse")
async def parse_content(request: ParseRequest):
    """Scrape website and parse for specific products"""
    try:
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
    """Chat interface for finding products with optional TTS"""
    try:
        # Validate input
        if not request.url or not request.query:
            raise HTTPException(status_code=400, detail="URL and query are required")
        
        # Process the request with timeout
        result = await safe_scrape_and_parse(request.url, request.query)
        
        response_data = {
            "success": True,
            "response": result,
            "message": "Products found successfully"
        }
        
        # Generate TTS if requested
        if request.voice_enabled and result:
            try:
                audio_base64 = tts_service.generate_speech(result, request.voice_id)
                if audio_base64:
                    response_data["audio"] = audio_base64
                    response_data["audio_format"] = "mp3"
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                # Don't fail the entire request if TTS fails
        
        return response_data
        
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

@router.post("/tts")
async def generate_tts(request: TTSRequest):
    """Generate text-to-speech audio"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        audio_base64 = tts_service.generate_speech(request.text, request.voice_id)
        
        if not audio_base64:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        return {
            "success": True,
            "audio": audio_base64,
            "audio_format": "mp3",
            "message": "Audio generated successfully"
        }
    except Exception as e:
        logger.error(f"Error generating TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

@router.get("/voices")
async def get_voices():
    """Get available TTS voices"""
    try:
        voices = tts_service.get_available_voices()
        return {
            "success": True,
            "voices": voices,
            "message": "Voices retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting voices: {str(e)}")

@router.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat response to handle long operations"""
    async def generate_response():
        try:
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Scraping website...'})}\n\n"
            
            # Scrape website
            html_content = await asyncio.wait_for(
                asyncio.to_thread(scrape_website, request.url),
                timeout=30.0
            )
            
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Extracting content...'})}\n\n"
            
            body_content = extract_body_content(html_content)
            cleaned_content = clean_body_content(body_content)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Analyzing products...'})}\n\n"
            
            dom_chunks = split_dom_content(cleaned_content)
            parsed_result = await asyncio.wait_for(
                asyncio.to_thread(parse_with_ollama, dom_chunks, request.query),
                timeout=60.0
            )
            
            response_data = {'status': 'complete', 'response': parsed_result}
            
            # Generate TTS if requested
            if request.voice_enabled and parsed_result:
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating audio...'})}\n\n"
                try:
                    audio_base64 = tts_service.generate_speech(parsed_result, request.voice_id)
                    if audio_base64:
                        response_data["audio"] = audio_base64
                        response_data["audio_format"] = "mp3"
                except Exception as e:
                    logger.error(f"TTS generation failed: {e}")
            
            yield f"data: {json.dumps(response_data)}\n\n"
            
        except asyncio.CancelledError:
            yield f"data: {json.dumps({'status': 'cancelled', 'message': 'Request cancelled'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )
