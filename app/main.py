from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime
import io
import os
import json
from pydub import AudioSegment
from groq import Groq

app = FastAPI(
    title="Audio Processing API",
    description="Process .wav audio files with Groq transcription",
    version="1.0.0"
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
    client = Groq()
    GROQ_AVAILABLE = True
except Exception as e:
    print(f"Groq client initialization failed: {e}")
    GROQ_AVAILABLE = False

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
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            prompt="Transcribe this audio file accurately",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",
            temperature=0.0
        )
        
        return {
            "text": transcription.text,
            "language": transcription.language,
            "duration": transcription.duration,
            "segments": transcription.segments if hasattr(transcription, 'segments') else []
        }
        
    except Exception as e:
        print(f"Groq transcription failed: {str(e)}")
        return {
            "text": f"Transcription failed: {str(e)}",
            "language": "en",
            "duration": None,
            "segments": []
        }

@app.post("/api/audio/process")
async def process_audio(audio_file: UploadFile = File(...)):
    """
    Process a .wav audio file with compression and Groq transcription
    """
    try:
        # Validate file type
        if not audio_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only .wav files are supported")
        
        # Read audio file
        audio_content = await audio_file.read()
        original_size = len(audio_content)
        
        # Compress if over 100MB
        compressed_audio = compress_audio(audio_content)
        final_size = len(compressed_audio)
        
        # Check if compression was applied
        was_compressed = final_size < original_size
        
        # Transcribe audio using Groq
        transcription_result = transcribe_audio(compressed_audio, audio_file.filename)
        
        return {
            "session_id": str(uuid.uuid4()),
            "filename": audio_file.filename,
            "original_size": original_size,
            "final_size": final_size,
            "was_compressed": was_compressed,
            "compression_ratio": round(final_size / original_size, 2) if original_size > 0 else 1.0,
            "transcription": transcription_result,
            "groq_available": GROQ_AVAILABLE,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "audio-processing",
        "groq_available": GROQ_AVAILABLE
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Audio Processing API with Groq Transcription",
        "version": "1.0.0",
        "supported_formats": [".wav"],
        "max_size": "100MB (auto-compressed if larger)",
        "transcription": "Groq Whisper API" if GROQ_AVAILABLE else "Mock (Groq not available)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 