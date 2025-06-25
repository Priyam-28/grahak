import os
from murf import Murf
import logging
from typing import Optional
import tempfile
import base64

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Murf client with API key"""
        try:
            api_key = os.getenv("MURF_API_KEY")
            if not api_key:
                logger.warning("MURF_API_KEY not found in environment variables")
                return
            
            self.client = Murf(api_key=api_key)
            logger.info("Murf TTS client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Murf client: {e}")
    
    def generate_speech(self, text: str, voice_id: str = "en-US-terrell") -> Optional[str]:
        """
        Generate speech from text using Murf API
        Returns base64 encoded audio data
        """
        if not self.client:
            logger.error("Murf client not initialized")
            return None
        
        try:
            # Generate speech
            res = self.client.text_to_speech.generate(
                text=text,
                voice_id=voice_id,
            )
            
            # Read the audio file and convert to base64
            with open(res.audio_file, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(res.audio_file)
            
            return audio_base64
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        # Common Murf voices - you can expand this list
        return [
            {"id": "en-US-terrell", "name": "Terrell (US English)", "language": "en-US"},
            {"id": "en-US-natalie", "name": "Natalie (US English)", "language": "en-US"},
            {"id": "en-US-clint", "name": "Clint (US English)", "language": "en-US"},
            {"id": "en-US-ruby", "name": "Ruby (US English)", "language": "en-US"},
            {"id": "en-GB-charles", "name": "Charles (UK English)", "language": "en-GB"},
            {"id": "en-GB-lily", "name": "Lily (UK English)", "language": "en-GB"},
        ]

# Global TTS service instance
tts_service = TTSService()
