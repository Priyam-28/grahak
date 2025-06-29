from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
import asyncio
from dotenv import load_dotenv # Added for .env loading
from routers.findproduct import router as findproduct_router

# Load environment variables from .env file
load_dotenv()

# Configure logging to suppress the connection error
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

app = FastAPI(title="Product Finder API", version="1.0.0")

# CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(findproduct_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Product Finder API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=False  # Disable access logs to reduce noise
    )
