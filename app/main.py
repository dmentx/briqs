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
from dotenv import load_dotenv
import instructor

# Optional imports
try:
    import openai
    OPENAI_IMPORT_AVAILABLE = True
except ImportError:
    OPENAI_IMPORT_AVAILABLE = False

# Import simplified models
from src.models.core import (
    Result, Excavator, AluminumSheet, Item, RequestNegotiate, Playbook, Buyer, ResultToAgent,
    ResultData, ProductDetails, BuyerProfile, SellerPlaybookDetails, BuyerPlaybookDetails, ExcavatorOrAluminumSheet
)

# Import negotiation engine
from src.crew_ai.crew import NegotiationEngine

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
    load_dotenv()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    GROQ_AVAILABLE = True
except Exception as e:
    print(f"Groq client initialization failed: {e}")
    GROQ_AVAILABLE = False

try:
    if OPENAI_IMPORT_AVAILABLE:
        openai_client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        OPENAI_AVAILABLE = True
    else:
        openai_client = None
        OPENAI_AVAILABLE = False
except Exception as e:
    print(f"OpenAI Groq client initialization failed: {e}")
    openai_client = None
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
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",
            temperature=0.0
        )
        
        return {
            "text": transcription,
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


@app.get("/api/featuredProducts")
async def featured_products_endpoint(buyer_id: int):
    """
    Get featured products excluding already purchased items.
    
    Args:
        buyer_id: The ID of the buyer
        
    Returns:
        List of recommended products (excavators and aluminum sheets)
        that the buyer hasn't purchased yet
    """
    try:
        # Get purchased items for this buyer
        purchased_item_ids = get_purchased_items(buyer_id)
        logger.info(f"Buyer {buyer_id} has purchased {len(purchased_item_ids)} items")
        
        # Load all available products
        all_excavators = load_excavators()
        all_aluminum_sheets = load_aluminum_sheets()
        
        # Filter out already purchased excavators
        recommended_excavators = []
        for excavator in all_excavators:
            if str(excavator.id) not in purchased_item_ids:
                recommended_excavators.append(excavator)
        
        # Filter out already purchased aluminum sheets
        recommended_aluminum_sheets = []
        for aluminum_sheet in all_aluminum_sheets:
            if str(aluminum_sheet.id) not in purchased_item_ids:
                recommended_aluminum_sheets.append(aluminum_sheet)
        
        logger.info(f"Recommended {len(recommended_excavators)} excavators and {len(recommended_aluminum_sheets)} aluminum sheets for buyer {buyer_id}")
        
        return {
            "buyer_id": buyer_id,
            "recommended_excavators": recommended_excavators,
            "recommended_aluminum_sheets": recommended_aluminum_sheets,
            "total_recommendations": len(recommended_excavators) + len(recommended_aluminum_sheets)
        }
        
    except Exception as e:
        logger.error(f"Error in featured_products_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

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

@app.post("/api/negotiate")
async def negotiate_endpoint(request: RequestNegotiate):
    """
    Process text input and return matched products
    """
    try:

        # Use instructor with Groq for structured output
        instructor_client = instructor.from_groq(client)


        result = instructor_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system","content":"you are a helpful assistant. Does the user want to buy an excavator or an aluminum sheet?"},
                {"role": "user","content":f"User input: {request.text_input} and buyer id: {request.buyer_id} playbook should be null "},
            ],
            response_model=Item,
            temperature=0.1
        )
        item = get_item(result)
        filtered_items = get_filtered_items(item)
        return filtered_items
    
        #integrate playbook 
        buyer_profile = get_buyer_profile(request.buyer_id)

        # Create Result object based on the detected product type
        if item and isinstance(item, Excavator):
            # Handle excavator
            simple_result = Result(
                product_type="excavator",
                text_input=request.text_input,
                list_excavator=filtered_items if filtered_items else [],
                list_alu=[],
                buyer_playbook=str(playbook) if playbook else "",
                buyer_profile=str(buyer_profile) if buyer_profile else "",
            )
        elif item and isinstance(item, AluminumSheet):
            # Handle aluminum sheet
            simple_result = Result(
                product_type="aluminum_sheet",
                text_input=request.text_input,
                list_excavator=[],
                list_alu=filtered_items if filtered_items else [],
                buyer_playbook=str(playbook) if playbook else "",
                buyer_profile=str(buyer_profile) if buyer_profile else "",
            )
        else:
            # Default case - no specific product detected
            simple_result = Result(
                product_type="unknown",
                text_input=request.text_input,
                list_excavator=[],
                list_alu=[],
                buyer_playbook=str(playbook) if playbook else "",
                buyer_profile=str(buyer_profile) if buyer_profile else "",
            )

        # Convert to ResultToAgent format
        result_to_agent = convert_result_to_agent(simple_result, request.buyer_id)
        
        negotiation_engine = NegotiationEngine()
        output_agent =negotiation_engine.start()
        return output_agent

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_buyer_playbook(buyer_id):
    #Load playbooks from json file as list of Playbook
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'src/playbooks/excavator/briqs_buyer_playbook.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    playbooks = [Playbook(**item) for item in data["playbooks"]]
    # Find playbook by buyer_id
    for playbook in playbooks:
        if playbook.buyer_id == buyer_id:
            return playbook
    return None

def get_seller_playbook(seller_id):
    #Load playbooks from json file as list of Playbook
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'src/playbooks/excavator/briqs_seller_playbook.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    playbooks = [Playbook(**item) for item in data["playbooks"]]
    # Find playbook by seller_id
    for playbook in playbooks:
        if playbook.seller_id == seller_id:
            return playbook
    return None

def get_buyer_profile(buyer_id):
    #Load buyer profile from json file as list of Buyer
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'src/playbooks/excavator/briqs_buyer_profile_1.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    buyer_profiles = [Buyer(**item) for item in data]
    # Find buyer profile by buyer_id
    for profile in buyer_profiles:
        if profile.buyer_id == buyer_id:
            return profile
    return None

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
    Filters items from mock data based on brand name only.
    For excavators: uses 'brand' field
    For aluminum sheets: uses 'brand' field, falls back to 'seller_name'
    Supports both exact and partial matching.
    """
    if isinstance(item, Excavator):
        all_items = load_excavators()
    elif isinstance(item, AluminumSheet):
        all_items = load_aluminum_sheets()
    else:
        return []

    # Extract only the brand from the filter item
    filter_brand = getattr(item, 'brand', None)
    
    # If no brand criteria provided, return nothing
    if not filter_brand:
        return []

    matched_items = []
    for db_item in all_items:
        db_brand = getattr(db_item, 'brand', None)
        
        # For aluminum sheets, fall back to seller_name if no brand field
        if not db_brand and isinstance(item, AluminumSheet):
            db_brand = getattr(db_item, 'seller_name', None)
        
        # Check if brands match
        if db_brand and _matches_brand(str(db_brand), str(filter_brand)):
            matched_items.append(db_item)

    for item in matched_items:
        seller_playbook = get_seller_playbook(item.seller_playbook)
        item.seller_playbook = seller_playbook
    
    return matched_items

def get_seller_playbook(seller_playbook_name:str):
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, f'src/playbooks/excavator/{seller_playbook_name}')
    with open(file_path, 'r') as f:
        data = json.load(f)
    playbooks = str(data)
    return playbooks


def _matches_brand(db_brand: str, filter_brand: str) -> bool:
    """
    Helper function to match brand names with flexible matching.
    Supports both exact and partial case-insensitive matching.
    """
    db_brand_lower = db_brand.lower().strip()
    filter_brand_lower = filter_brand.lower().strip()
    
    # Exact match
    if db_brand_lower == filter_brand_lower:
        return True
    
    # Partial match - check if filter is contained in db brand
    if filter_brand_lower in db_brand_lower:
        return True
    
    # Partial match - check if db brand is contained in filter
    if db_brand_lower in filter_brand_lower:
        return True
    
    return False


def get_purchased_items(buyer_id: int) -> set[str]:
    """
    Extract all purchased item IDs for a given buyer.
    
    Args:
        buyer_id: The ID of the buyer
        
    Returns:
        Set of purchased item IDs (both excavators and aluminum sheets)
    """
    try:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'src/mock_data/deals.json')
        
        if not os.path.exists(file_path):
            logger.warning(f"Deals data file not found at {file_path}")
            return set()
            
        with open(file_path, 'r') as f:
            deals_data = json.load(f)
        
        purchased_items = set()
        
        for deal in deals_data:
            if deal.get('buyer_id') == buyer_id:
                # Extract product IDs from the deal
                products = deal.get('product', [])
                for product_item in products:
                    # Check for excavator
                    if product_item.get('excavator'):
                        excavator_id = product_item['excavator'].get('id')
                        if excavator_id:
                            purchased_items.add(excavator_id)
                    
                    # Check for aluminum sheet
                    if product_item.get('aluminum_sheet'):
                        aluminum_id = product_item['aluminum_sheet'].get('id')
                        if aluminum_id:
                            purchased_items.add(aluminum_id)
        
        logger.info(f"Found {len(purchased_items)} purchased items for buyer {buyer_id}")
        return purchased_items
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Could not load or parse deals.json: {e}")
        return set()
    except Exception as e:
        logger.error(f"Error extracting purchased items: {e}")
        return set()


def get_item(item:Item):
    if item.excavator:
        return item.excavator
    if item.aluminum_sheet:
        return item.aluminum_sheet
    return None


def create_result_to_agent(product_type: str, text_input: str, buyer_id: int, 
                          excavators: List[Excavator] = None, 
                          aluminum_sheets: List[AluminumSheet] = None) -> ResultToAgent:
    """
    Helper function to create a ResultToAgent object directly from parameters.
    
    Args:
        product_type: Type of product ("excavator", "aluminum_sheet", etc.)
        text_input: Original text input from user
        buyer_id: ID of the buyer
        excavators: List of matched excavators (optional)
        aluminum_sheets: List of matched aluminum sheets (optional)
    
    Returns:
        ResultToAgent object with detailed playbook structure
    """
    # Create a Result object first
    result = Result(
        product_type=product_type,
        text_input=text_input,
        list_excavator=excavators or [],
        list_alu=aluminum_sheets or [],
        buyer_playbook="",
        buyer_profile=""
    )
    
    # Convert to ResultToAgent
    return convert_result_to_agent(result, buyer_id)


def convert_result_to_agent(result: Result, buyer_id: int) -> ResultToAgent:
    """
    Convert a simple Result object to a complex ResultToAgent object
    by loading detailed playbook and buyer profile data.
    """
    try:
        # Load detailed playbook data from knowledge base
        base_path = os.path.dirname(__file__)
        
        # Load the combined playbook (seller + buyer details)
        if result.product_type == "excavator":
            combined_playbook_path = os.path.join(base_path, 'src/knowledge_base/excavator_seller1.json')
        else:
            # Default to excavator for now, can be extended for other product types
            combined_playbook_path = os.path.join(base_path, 'src/knowledge_base/excavator_seller1.json')
        
        detailed_playbook = None
        if os.path.exists(combined_playbook_path):
            with open(combined_playbook_path, 'r') as f:
                detailed_playbook = json.load(f)
        
        # Extract detailed structures from the loaded playbook
        if detailed_playbook and "result" in detailed_playbook:
            playbook_data = detailed_playbook["result"]
            
            # Extract product details
            product_details = None
            if "product_details" in playbook_data:
                product_details = ProductDetails(**playbook_data["product_details"])
            
            # Extract buyer profile
            buyer_profile = None
            if "buyer_profile" in playbook_data:
                buyer_profile = BuyerProfile(**playbook_data["buyer_profile"])
            
            # Create ResultData
            result_data = ResultData(
                product_type=result.product_type,
                product_details=product_details,
                buyer_profile=buyer_profile
            )
            
            # Create ResultToAgent
            return ResultToAgent(result=result_data)
        
        # Fallback: create a basic structure if detailed data is not available
        logger.warning(f"Detailed playbook data not found, creating basic structure for buyer {buyer_id}")
        
        # Create basic buyer profile
        buyer_profile = BuyerProfile(
            credit_worthiness=8,  # Default value
            recurring_customer=True  # Default value
        )
        
        # Create basic result data
        result_data = ResultData(
            product_type=result.product_type,
            product_details=None,  # No detailed product info available
            buyer_profile=buyer_profile
        )
        
        return ResultToAgent(result=result_data)
        
    except Exception as e:
        logger.error(f"Error converting Result to ResultToAgent: {e}")
        # Return a minimal structure on error
        return ResultToAgent(
            result=ResultData(
                product_type=result.product_type,
                product_details=None,
                buyer_profile=BuyerProfile(credit_worthiness=5, recurring_customer=False)
            )
        )



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