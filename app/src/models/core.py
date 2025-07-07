# Models will be built as needed from endpoint requirements 

from pydantic import BaseModel, Field, model_validator
from uuid import UUID, uuid4
from typing import Optional, List, Dict, Any

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

class FinalAgreedTerms(BaseModel):
    status: Optional[str] = Field(default=None, description="Status of the final agreed terms")
    price: Optional[float] = Field(default=None, description="Price of the final agreed terms")
    payment_terms: Optional[str] = Field(default=None, description="Payment terms of the final agreed terms")
    warranty: Optional[str] = Field(default=None, description="Warranty of the final agreed terms")
    maintenance_services: Optional[str] = Field(default=None, description="Maintenance services of the final agreed terms")
    additional_terms: Optional[str] = Field(default=None, description="Additional terms of the final agreed terms")

class Playbook(BaseModel):
    """Playbook for negotiation strategies"""
    playbook_id: UUID = Field(default_factory=uuid4)
    buyer_id: Optional[int] = None
    name: Optional[str] = None
    strategy_type: Optional[str] = Field(default=None, description="buyer or seller")
    
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


class ExcavatorOrAluminumSheet(BaseModel):
    is_excavator: Optional[bool] = Field(default=None, description="Is excavator")






# Models for ResultToAgent structure
class ProductPricing(BaseModel):
    walk_away_price_usd: Optional[float] = Field(default=None, alias="Walk-Away-Price (USD)")
    target_price_usd: Optional[float] = Field(default=None, alias="Target Price (USD)")
    starting_price: Optional[float] = Field(default=None, alias="Starting Price")

class RiskProfileDefinition(BaseModel):
    low_risk: Optional[str] = None
    medium_risk: Optional[str] = None
    high_risk: Optional[str] = None

class BuyerCriteria(BaseModel):
    risk_profile_definition: Optional[RiskProfileDefinition] = None

class ProductCriteria(BaseModel):
    product: Optional[ProductPricing] = None
    buyer: Optional[BuyerCriteria] = None

class PaymentTerms(BaseModel):
    goal: Optional[str] = None
    fallback_position: Optional[str] = None

class CollateralTerms(BaseModel):
    goal: Optional[str] = None
    fallback_position: Optional[str] = None

class RiskBuyerTerms(BaseModel):
    payment_terms: Optional[PaymentTerms] = Field(default=None, alias="Payment Terms")
    collateral_for_payment_default: Optional[CollateralTerms] = Field(default=None, alias="Collateral for Payment Default")

class IdealAcceptableTerms(BaseModel):
    high_risk_buyer: Optional[RiskBuyerTerms] = None
    medium_risk_buyer: Optional[RiskBuyerTerms] = None
    low_risk_buyer: Optional[Dict[str, Any]] = None

class SellerTradables(BaseModel):
    primary_goal: Optional[str] = Field(default=None, alias="Primary Goal")
    give_low_cost_to_us: Optional[List[str]] = Field(default=None, alias="Give (Low-cost to us)")
    get_high_value_to_us: Optional[List[str]] = Field(default=None, alias="Get (High value to us)")
    ideal_acceptable_terms: Optional[IdealAcceptableTerms] = Field(default=None, alias="Ideal & Acceptable Terms")

class SellerPlaybookDetails(BaseModel):
    criteria: Optional[ProductCriteria] = Field(default=None, alias="Criteria")
    negotiation_rules: Optional[List[str]] = Field(default=None, alias="Negotiation rules")
    tradables: Optional[SellerTradables] = Field(default=None, alias="Tradables")

class BuyerTermsDetail(BaseModel):
    target_purchase_price_usd: Optional[float] = Field(default=None, alias="Target Purchase Price (USD)")
    maximum_budget_usd: Optional[float] = Field(default=None, alias="Maximum Budget (USD)")
    ideal: Optional[str] = None
    fallback_position: Optional[str] = None

class BuyerIdealAcceptableTerms(BaseModel):
    price: Optional[BuyerTermsDetail] = Field(default=None, alias="Price")
    payment_terms: Optional[BuyerTermsDetail] = Field(default=None, alias="Payment Terms")
    warranty: Optional[BuyerTermsDetail] = Field(default=None, alias="Warranty")
    delivery: Optional[BuyerTermsDetail] = Field(default=None, alias="Delivery")

class BuyerTradables(BaseModel):
    primary_goal: Optional[str] = Field(default=None, alias="Primary Goal")
    get_high_value_to_us: Optional[List[str]] = Field(default=None, alias="Get (High value to us)")
    give_low_cost_to_us: Optional[List[str]] = Field(default=None, alias="Give (Low-cost to us)")
    ideal_acceptable_terms: Optional[BuyerIdealAcceptableTerms] = Field(default=None, alias="Ideal & Acceptable Terms")

class BuyerPlaybookDetails(BaseModel):
    negotiation_strategy: Optional[List[str]] = Field(default=None, alias="Negotiation Strategy")
    tradables: Optional[BuyerTradables] = Field(default=None, alias="Tradables")

class ProductDetails(BaseModel):
    seller_playbook: Optional[SellerPlaybookDetails] = None
    buyer_playbook: Optional[BuyerPlaybookDetails] = None

class BuyerProfile(BaseModel):
    credit_worthiness: Optional[int] = Field(default=None, alias="Credit Worthiness")
    recurring_customer: Optional[bool] = Field(default=None, alias="Recurring Customer")

class ResultData(BaseModel):
    product_type: Optional[str] = None
    product_details: Optional[ProductDetails] = None
    buyer_profile: Optional[BuyerProfile] = None

class ResultToAgent(BaseModel):
    result: Optional[ResultData] = None
    
