# Models will be built as needed from endpoint requirements 

from pydantic import BaseModel, Field, model_validator
from uuid import UUID, uuid4
from typing import Optional, List

class Excavator(BaseModel):
    id: Optional[UUID] = Field(default_factory=uuid4)
    name: Optional[str] = None
    seller_name: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    price: Optional[float] = None
    condition: Optional[str] = None
    lifting_capacity_tons: Optional[float] = Field(default=None, description="Maximum lifting capacity in tons")
    operating_weight_tons: Optional[float] = Field(default=None, description="Operating weight of the excavator in tons")
    max_digging_depth_m: Optional[float] = Field(default=None, description="Maximum digging depth in meters")
    bucket_capacity_m3: Optional[float] = Field(default=None, description="Bucket capacity in cubic meters")
    seller_playbook: Optional[str] = None

class ExcavatorDTO(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    price: Optional[float] = None
    condition: Optional[str] = None
    lifting_capacity_tons: Optional[float] = Field(default=None, description="Maximum lifting capacity in tons")
    operating_weight_tons: Optional[float] = Field(default=None, description="Operating weight of the excavator in tons")
    max_digging_depth_m: Optional[float] = Field(default=None, description="Maximum digging depth in meters")
    bucket_capacity_m3: Optional[float] = Field(default=None, description="Bucket capacity in cubic meters")
    seller_playbook: Optional[str] = None


class AluminumSheet(BaseModel):
    id: Optional[UUID] = Field(default_factory=uuid4)
    name: Optional[str] = None
    seller_name: Optional[str] = None
    price: Optional[float] = None
    availability: Optional[int] = None
    thickness_mm: Optional[float] = Field(default=None, description="Thickness in millimeters")
    total_weight_kg: Optional[float] = Field(default=None, description="Total sheet weight in kg")
    seller_playbook: Optional[str] = None

class AluminumSheetDTO(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    availability: Optional[int] = None
    thickness_mm: Optional[float] = Field(default=None, description="Thickness in millimeters")
    total_weight_kg: Optional[float] = Field(default=None, description="Total sheet weight in kg")
    playbook: Optional[str] = None

class Item(BaseModel):
    excavator: Optional[Excavator] = None
    aluminum_sheet: Optional[AluminumSheet] = None

class RequestNegotiate(BaseModel):
    text_input: str = Field(..., description="Transcribed text from audio input")
    buyer_id: int = Field(..., description="Buyer ID")

class Deal(BaseModel):
    deal_id: int = Field(..., description="Deal ID")
    buyer_id: int = Field(..., description="Buyer ID")
    product: List[Item] = Field(..., description="Product")
    contract_terms: str = Field(..., description="Contract terms")

class DealHistory(BaseModel):
    list_deals: List[Deal] = Field(..., description="List of deals")

class Buyer(BaseModel):
    buyer_id: int = Field(..., description="Buyer ID")
    buyer_profile: str = Field(default="", description="Buyer profile")
    deal_history: Optional[DealHistory] = Field(default=None, description="Deal history")

class DealDTO(BaseModel):
    contract_terms: str = Field(..., description="Contract terms")

class Playbook(BaseModel):
    """Playbook for negotiation strategies"""
    playbook_id: UUID = Field(default_factory=uuid4)
    name: str
    strategy_type: str = Field(..., description="buyer or seller")
    
class Offer(BaseModel):
    offer_id: UUID = Field(default_factory=uuid4)
    playbook: Optional[Playbook] = Field(default=None)


class Result(BaseModel):
    product_type:str = Field(default="", description="Product type")
    text_input: str = Field(..., description="Transcribed text from audio input")
    list_excavator: List[Excavator] = Field(default=[], description="Matched excavator products")
    list_alu: List[AluminumSheet] = Field(default=[], description="Matched aluminum sheet products")
    buyer_playbook: str = Field(default="", description="Simple playbook reference")
    buyer_profile:str = Field(default="", description="Buyer profile")
