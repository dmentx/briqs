from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime
import io
import os
import json
from pydub import AudioSegment
from groq import Groq
import logging
from typing import List

# Import simplified models
from src.models.core import Result, Excavator, AluminumSheet, Item

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Briqs - Audio to Text Product Matcher",
    version="1.0.0",
    description="Simple audio-to-text transcription with basic product matching"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    GROQ_AVAILABLE = True
except Exception as e:
    print(f"Groq client initialization failed: {e}")
    GROQ_AVAILABLE = False

try:
    openai_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
except Exception as e:
    print(f"OpenAI Groq client initialization failed: {e}")
    OPENAI_AVAILABLE = False

def compress_audio(audio_content: bytes) -> bytes:
    """
    Compress audio if it's over 100MB using pydub
    """
    try:
        max_size = 100 * 1024 * 1024  # 100MB
        
        if len(audio_content) <= max_size:
            return audio_content
            
        # Load audio with pydub
        audio = AudioSegment.from_wav(io.BytesIO(audio_content))
        
        # Compress by reducing sample rate and converting to mono
        compressed_audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Export back to bytes
        output_buffer = io.BytesIO()
        compressed_audio.export(output_buffer, format="wav")
        compressed_bytes = output_buffer.getvalue()
        
        # If still too large, reduce bit depth
        if len(compressed_bytes) > max_size:
            compressed_audio = compressed_audio.set_sample_width(2)  # 16-bit
            output_buffer = io.BytesIO()
            compressed_audio.export(output_buffer, format="wav")
            compressed_bytes = output_buffer.getvalue()
        
        return compressed_bytes
        
    except Exception as e:
        # Fallback to simple truncation if pydub fails
        print(f"Audio compression failed, using fallback: {str(e)}")
        if len(audio_content) > max_size:
            return audio_content[:max_size // 2]
        return audio_content

def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    """
    Transcribe audio using Groq API
    """
    try:
        if not GROQ_AVAILABLE:
            return {
                "text": f"Mock transcription from {filename} (Groq not available)",
                "language": "en",
                "duration": None,
                "segments": []
            }
        
        # Create a BytesIO object from audio bytes
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename  # Set filename for the API
        
        # Create transcription using Groq API
        transcription = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text",
            timestamp_granularities=["word", "segment"],
            language="en",
            temperature=0.0
        )
        
        return {
            "text": transcription.text,
            "language": "en",
            "duration": None,
            "segments": []
        }
        
    except Exception as e:
        print(f"Groq transcription failed: {str(e)}")
        return {
            "text": f"Transcription failed: {str(e)}",
            "language": "en",
            "duration": None,
            "segments": []
        }


@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """
    Transcribe audio file and return matched products
    """
    try:
        # Read file content
        audio_content = await file.read()
        
        # Compress audio if needed
        compressed_audio = compress_audio(audio_content)

        # Transcribe audio
        transcription_result = transcribe_audio(compressed_audio, file.filename)
        text_from_audio = transcription_result["text"]
        return text_from_audio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/offer")
async def offer_endpoint(text_input: str):
    """
    Process text input and return matched products
    """
    try:

        response = openai_client.responses.parse(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            input=[
                {"role": "system","content":"you are a helpful assistant find the following things, if you do not find these things set them to null"},
                {"role": "user","content":text_input},
            ],
            text_format=Item
        )
        result = response.output_parsed
        item = get_item(result)

        if item:
            return get_filtered_items(item)
        
        return []

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_excavators() -> List[Excavator]:
    """Loads excavators from mock data."""
    try:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'src/mock_data/excavator.json')
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.warning(f"Excavator data file not found or empty at {file_path}")
            return []
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [Excavator(**item) for item in data]
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Could not load or parse excavator.json: {e}")
        return []

def load_aluminum_sheets() -> List[AluminumSheet]:
    """Loads aluminum sheets from mock data."""
    try:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'src/mock_data/aluminium.json')
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.warning(f"Aluminium data file not found or empty at {file_path}")
            return []
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [AluminumSheet(**item) for item in data]
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Could not load or parse aluminium.json: {e}")
        return []


def get_filtered_items(item):
    """
    Filters items from mock data based on the attributes of the input item.
    """
    if isinstance(item, Excavator):
        all_items = load_excavators()
    elif isinstance(item, AluminumSheet):
        all_items = load_aluminum_sheets()
    else:
        return []

    # Get the fields from the input item that are not None
    filter_criteria = item.model_dump(exclude_none=True)
    
    # If no criteria are provided, return nothing
    if not filter_criteria:
        return []

    matched_items = []
    for db_item in all_items:
        is_match = True
        for key, value in filter_criteria.items():
            db_value = getattr(db_item, key, None)
            if db_value != value:
                is_match = False
                break
        if is_match:
            matched_items.append(db_item)
    
    return matched_items


def get_item(item:Item):
    if item.excavator:
        return item.excavator
    if item.aluminum_sheet:
        return item.aluminum_sheet
    return None

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "groq_available": GROQ_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 