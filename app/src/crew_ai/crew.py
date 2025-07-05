#!/usr/bin/env python3
"""
Simple Multi-Agent Negotiation System with Ollama
Fast setup and execution for contract negotiations
"""

from crewai import Agent, LLM, Task, Crew, Process
import json
import os

def get_ollama_llm():
    """Get the best available Ollama model."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json()
            model_names = [model["name"] for model in models.get("models", [])]
            
            # Use best available model
            if "llama4:16x17b" in model_names:
                return LLM(model="ollama/llama4:16x17b", base_url="http://localhost:11434")
            elif any("llama3.1" in name and "70b" in name for name in model_names):
                model = next(name for name in model_names if "llama3.1" in name and "70b" in name)
                return LLM(model=f"ollama/{model}", base_url="http://localhost:11434")
            elif any("llama" in name for name in model_names):
                model = next(name for name in model_names if "llama" in name)
                return LLM(model=f"ollama/{model}", base_url="http://localhost:11434")
    except:
        pass
    
    # Fallback
    return LLM(model="ollama/llama3.1:70b", base_url="http://localhost:11434")

fast_llm = get_ollama_llm()

# 2. ====== PLAYBOOK LOADING ======
def load_buyer_playbook(playbook_path="src/knowledge_base/briqs_buyer_playbook.json"):
    """Load buyer negotiation playbook from JSON file."""
    try:
        with open(playbook_path, 'r') as f:
            playbook = json.load(f)
        return playbook
    except FileNotFoundError:
        print(f"⚠️  Playbook file not found: {playbook_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing playbook JSON: {e}")
        return None

def load_seller_playbook(playbook_path="src/knowledge_base/briqs_seller_playbook_2.json"):
    """Load seller negotiation playbook from JSON file."""
    try:
        with open(playbook_path, 'r') as f:
            playbook = json.load(f)
        return playbook
    except FileNotFoundError:
        print(f"⚠️  Seller playbook file not found: {playbook_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing seller playbook JSON: {e}")
        return None

def build_buyer_task_description(playbook, contract_type="heavy_equipment"):
    """Build dynamic buyer task description from playbook data."""
    if not playbook:
        return "Error: No playbook data available"
    
    # Extract key data from playbook
    strategies = playbook.get("Negotiation Strategy", [])
    tradables = playbook.get("Tradables", {})
    terms = playbook.get("Ideal & Acceptable Terms", {})
    
    # Build strategy section
    strategy_text = "\n    ".join([f"{i+1}. {strategy}" for i, strategy in enumerate(strategies)])
    
    # Build tradables sections
    primary_goal = tradables.get("Primary Goal", "")
    get_items = tradables.get("Get (High value to us)", [])
    give_items = tradables.get("Give (Low-cost to us)", [])
    
    get_text = "\n    - ".join(get_items)
    give_text = "\n    - ".join(give_items)
    
    # Build terms sections
    price_terms = terms.get("Price", {})
    payment_terms = terms.get("Payment Terms", {})
    warranty_terms = terms.get("Warranty", {})
    delivery_terms = terms.get("Delivery", {})
    
    target_price = price_terms.get("Target Purchase Price (USD)", 160000)
    max_budget = price_terms.get("Maximum Budget (USD)", 170000)
    
    description = f"""
    As the buyer representative, make your opening offer for the {contract_type} following your established negotiation playbook.
    
    YOUR BUYER PROFILE:
    - Credit Worthiness: 8/10 (Strong)
    - Recurring Customer: Yes (Established relationship)
    - Target Purchase Price: ${target_price:,}
    - Maximum Budget: ${max_budget:,} (NEVER REVEAL THIS)
    
    NEGOTIATION STRATEGY (From Playbook):
    {strategy_text}
    
    YOUR PRIMARY GOAL: {primary_goal}
    
    WHAT TO GET (High value to you):
    - {get_text}
    
    WHAT TO GIVE (Low-cost to you):
    - {give_text}
    
    IDEAL TERMS:
    - Price: {price_terms.get("Ideal", "At or below target price")}
    - Payment: {payment_terms.get("Ideal", "Net 60 terms with 0% down payment")}
    - Warranty: {warranty_terms.get("Ideal", "3-year comprehensive warranty")}
    - Delivery: {delivery_terms.get("Ideal", "Free delivery to site")}
    
    FALLBACK POSITION:
    - Price: {price_terms.get("Fallback Position", "Up to maximum budget with significant value")}
    - Payment: {payment_terms.get("Fallback Position", "Net 30 terms with minimal down payment")}
    - Warranty: {warranty_terms.get("Fallback Position", "Minimum 2-year powertrain warranty")}
    - Delivery: {delivery_terms.get("Fallback Position", "Reasonable delivery fee")}
    
    Make your opening offer based on this strategy. Start aggressive but realistic around your target price or lower.
    
    Output: JSON with {{opening_offer_price, offer_justification, payment_terms_requested, warranty_requirements, delivery_requirements, bundled_requests, commitment_level}}
    """
    
    return description

def build_seller_task_description(seller_playbook, buyer_playbook, buyer_risk_profile="low_risk"):
    """Build dynamic seller task description from playbook data."""
    if not seller_playbook:
        return "Error: No seller playbook data available"
    
    # Extract key data from seller playbook
    criteria = seller_playbook.get("Criteria", {})
    product = criteria.get("Product", {})
    buyer_criteria = criteria.get("Buyer", {})
    
    rules = seller_playbook.get("Negotiation rules", [])
    tradables = seller_playbook.get("Tradables", {})
    terms = seller_playbook.get("Ideal & Acceptable Terms", {})
    
    # Build pricing structure
    starting_price = product.get("Starting Price (USD)", 195000)
    target_price = product.get("Target Price (USD)", 180000)
    walkaway_price = product.get("Walk-Away-Price (USD)", 172500)
    
    # Build rules section
    rules_text = "\n    ".join([f"{i+1}. {rule}" for i, rule in enumerate(rules)])
    
    # Build tradables sections
    primary_goal = tradables.get("Primary Goal", "")
    give_items = tradables.get("Give (Low-cost to us)", [])
    get_items = tradables.get("Get (High value to us)", [])
    
    give_text = "\n    - ".join(give_items)
    get_text = "\n    - ".join(get_items)
    
    # Build risk profile definitions
    risk_definitions = buyer_criteria.get("risk_profile_definition", {})
    low_risk_def = risk_definitions.get("low_risk", "")
    medium_risk_def = risk_definitions.get("medium_risk", "")
    high_risk_def = risk_definitions.get("high_risk", "")
    
    # Get terms for current buyer risk profile
    current_buyer_terms = terms.get(f"{buyer_risk_profile.replace('_', ' ').title()} risk buyer", {})
    payment_terms = current_buyer_terms.get("Payment Terms", {})
    
    # Extract buyer budget info if available
    buyer_target = "Target price from buyer playbook"
    buyer_max = "Maximum budget from buyer playbook"
    if buyer_playbook:
        buyer_price_terms = buyer_playbook.get("Ideal & Acceptable Terms", {}).get("Price", {})
        buyer_target = f"${buyer_price_terms.get('Target Purchase Price (USD)', 160000):,}"
        buyer_max = f"${buyer_price_terms.get('Maximum Budget (USD)', 170000):,}"
    
    description = f"""
    As the seller representative, respond to the buyer's offer following your established negotiation playbook.
    
    YOUR SELLER PRICING STRUCTURE:
    - Starting Price: ${starting_price:,}
    - Target Price: ${target_price:,}
    - Walk-Away-Price: ${walkaway_price:,} (NEVER REVEAL THIS)
    
    BUYER RISK ASSESSMENT:
    - Credit Worthiness: 8/10 (Strong credit report)
    - Recurring Customer: Yes (Established relationship)
    - RISK PROFILE: {buyer_risk_profile.upper().replace('_', ' ')} RISK
    
    RISK PROFILE DEFINITIONS:
    - Low Risk: {low_risk_def}
    - Medium Risk: {medium_risk_def}
    - High Risk: {high_risk_def}
    
    NEGOTIATION RULES (From Playbook):
    {rules_text}
    
    YOUR PRIMARY GOAL: {primary_goal}
    
    WHAT TO GIVE (Low-cost to you, no future liability):
    - {give_text}
    
    WHAT TO GET (High value to you):
    - {get_text}
    
    {buyer_risk_profile.upper().replace('_', ' ')} RISK BUYER TERMS (Applicable to this buyer):
    - IDEAL Payment Terms: {payment_terms.get("Ideal", "Standard terms for risk profile")}
    - FALLBACK: {payment_terms.get("Fallback Position", "Risk-appropriate fallback terms")}
    
    BUYER CONTEXT (From their playbook):
    - Buyer Target Price: {buyer_target} (not revealed to you)
    - Buyer Maximum Budget: {buyer_max} (not revealed to you)
    
    RESPONSE STRATEGY:
    1. Evaluate buyer's offer against your Walk-Away-Price (${walkaway_price:,})
    2. If offer is below Walk-Away-Price: Firmly state it's significantly below valuation, invite revised offer
    3. If offer is above Walk-Away-Price: Consider strategic counter-offer moving toward Target Price
    4. Emphasize value proposition and justify pricing based on quality/service
    5. For any concessions requested, ask for risk-reducing concessions in return
    6. Apply terms appropriate for {buyer_risk_profile.replace('_', ' ')} risk buyer
    
    Buyer's offer details: (will be provided by orchestrator)
    
    Output: JSON with {{response_decision, counter_offer_price, risk_assessment_summary, value_justification, payment_terms_offered, concessions_requested, willingness_to_negotiate}}
    """
    
    return description

def build_orchestration_task_description(buyer_playbook, seller_playbook, contract_type="heavy_equipment", buyer_risk_profile="low_risk"):
    """Build dynamic orchestration task description from both playbooks."""
    
    # Extract buyer info
    buyer_target = 160000
    buyer_max = 170000
    if buyer_playbook:
        buyer_price_terms = buyer_playbook.get("Ideal & Acceptable Terms", {}).get("Price", {})
        buyer_target = buyer_price_terms.get("Target Purchase Price (USD)", 160000)
        buyer_max = buyer_price_terms.get("Maximum Budget (USD)", 170000)
    
    # Extract seller info
    seller_starting = 195000
    seller_target = 180000
    seller_walkaway = 172500
    if seller_playbook:
        product = seller_playbook.get("Criteria", {}).get("Product", {})
        seller_starting = product.get("Starting Price (USD)", 195000)
        seller_target = product.get("Target Price (USD)", 180000)
        seller_walkaway = product.get("Walk-Away-Price (USD)", 172500)
    
    # Calculate gap
    gap_percentage = abs(seller_starting - buyer_target) / buyer_target * 100
    
    description = f"""
    As the Negotiation Manager, orchestrate the sequential negotiation process between buyer and seller.
    
    CONTEXT (From Playbooks):
    - Contract type: {contract_type}
    - Buyer Profile: Credit Worthiness 8/10, Recurring Customer ({buyer_risk_profile.upper().replace('_', ' ')} RISK)
    - Buyer Target Price: ${buyer_target:,} (Max Budget: ${buyer_max:,} - not revealed to seller)
    - Seller Starting Price: ${seller_starting:,} (Target: ${seller_target:,}, Walk-Away: ${seller_walkaway:,})
    - Current Market Gap: Seller starts at ${seller_starting:,}, Buyer targets ${buyer_target:,} ({gap_percentage:.1f}% gap)
    
    ORCHESTRATION PROCESS:
    1. Receive buyer's opening offer from BuyerAgent (based on buyer playbook)
    2. Present buyer's offer to SellerAgent (without revealing buyer's maximum budget)
    3. Receive seller's response/counter-offer (based on seller playbook and risk assessment)
    4. Analyze if gap can be bridged through negotiation
    5. Facilitate up to 2 additional rounds of back-and-forth if needed
    6. Determine final outcome: Agreement or Mediation needed
    
    NEGOTIATION DYNAMICS TO CONSIDER:
    - Buyer wants: Target ${buyer_target:,}, warranty, payment terms, free delivery, maintenance included
    - Seller wants: Target ${seller_target:,}, secure payment terms, risk mitigation, minimal concessions
    - Overlap Zone: ${seller_walkaway:,} (seller walk-away) to ${buyer_max:,} (buyer max budget)
    - {buyer_risk_profile.replace('_', ' ').title()} risk buyer profile allows seller to offer appropriate terms
    
    AGREEMENT CRITERIA:
    - Price falls within overlap zone (${seller_walkaway:,} - ${buyer_max:,})
    - Both parties express acceptance of final terms
    - Payment terms are mutually acceptable
    - Key value-adds (warranty, delivery, service) are addressed
    
    MEDIATION CRITERIA:
    - Final price gap remains outside overlap zone after 2 rounds
    - Fundamental disagreement on critical terms (payment, warranty)
    - Either party approaches walkaway position
    - No creative solutions can bridge remaining gaps
    
    REQUIRED FINAL CONCLUSION:
    You MUST end with one of these clear outcomes:
    - "DEAL SUCCESSFULLY NEGOTIATED" with final agreed terms and price
    - "MEDIATION REQUIRED" and delegate to MediatorAgent with specific issues
    
    Output: JSON with {{buyer_opening_offer, seller_responses, negotiation_rounds, gap_analysis, price_progression, FINAL_CONCLUSION, agreed_terms_or_mediation_reason}}
    """
    
    return description

# 3. ====== NEGOTIATION AGENTS ======
BuyerAgent = Agent(
    role="Contract Buyer",
    goal="Secure favorable terms within budget constraints while maintaining relationships.",
    backstory="Experienced procurement professional with 10+ years in contract negotiation.",
    llm=fast_llm
)

SellerAgent = Agent(
    role="Contract Seller", 
    goal="Maximize revenue while ensuring deal closure and customer satisfaction.",
    backstory="Senior sales professional with expertise in B2B contract negotiations.",
    llm=fast_llm
)

OrchestratorAgent = Agent(
    role="Negotiation Manager",
    goal="Efficiently manage the negotiation crew, delegate tasks, and ensure successful deal completion.",
    backstory="You're an experienced negotiation manager with 15+ years in complex contract negotiations. You excel at coordinating teams, delegating tasks strategically, and knowing when to call in mediation support.",
    allow_delegation=True,  # Enable delegation for hierarchical management
    llm=fast_llm
)

MediatorAgent = Agent(
    role="Neutral Mediator",
    goal="Resolve conflicts and provide unbiased recommendations when needed.",
    backstory="Impartial negotiation expert specializing in conflict resolution.",
    llm=fast_llm
)

# 4. ====== NEGOTIATION SCENARIO ======
def run_negotiation(contract_type="heavy_equipment", buyer_target_price=160000, seller_starting_price=195000, buyer_risk_profile="low_risk"):
    """Run a complete negotiation scenario based on buyer and seller playbooks."""
    
    # Load buyer playbook
    print("📚 Loading buyer playbook...")
    buyer_playbook = load_buyer_playbook()
    
    if not buyer_playbook:
        print("❌ Failed to load buyer playbook! Using fallback values...")
        buyer_target_price = 160000
        buyer_max_budget = 170000
    else:
        print("✅ Buyer playbook loaded successfully!")
        # Extract prices from playbook
        price_terms = buyer_playbook.get("Ideal & Acceptable Terms", {}).get("Price", {})
        buyer_target_price = price_terms.get("Target Purchase Price (USD)", buyer_target_price)
        buyer_max_budget = price_terms.get("Maximum Budget (USD)", 170000)
    
    # Load seller playbook
    print("📚 Loading seller playbook...")
    seller_playbook = load_seller_playbook()
    
    if not seller_playbook:
        print("❌ Failed to load seller playbook! Using fallback values...")
        seller_starting_price = 195000
        seller_target_price = 180000
        seller_walkaway_price = 172500
    else:
        print("✅ Seller playbook loaded successfully!")
        # Extract prices from playbook
        product = seller_playbook.get("Criteria", {}).get("Product", {})
        seller_starting_price = product.get("Starting Price (USD)", seller_starting_price)
        seller_target_price = product.get("Target Price (USD)", 180000)
        seller_walkaway_price = product.get("Walk-Away-Price (USD)", 172500)
    
    # Create dynamic tasks using playbook data
    buyer_task_description = build_buyer_task_description(buyer_playbook, contract_type)
    seller_task_description = build_seller_task_description(seller_playbook, buyer_playbook, buyer_risk_profile)
    orchestration_task_description = build_orchestration_task_description(buyer_playbook, seller_playbook, contract_type, buyer_risk_profile)
    
    buyer_opening_task = Task(
        description=buyer_task_description,
        agent=BuyerAgent,
        expected_output=f"Strategic opening offer based on buyer playbook for {contract_type}"
    )
    
    seller_response_task = Task(
        description=seller_task_description,
        agent=SellerAgent,
        expected_output="Strategic response based on seller playbook with risk-appropriate terms"
    )
    
    orchestration_task = Task(
        description=orchestration_task_description,
        agent=OrchestratorAgent,
        expected_output="Sequential negotiation management with realistic playbook-based terms and clear conclusion"
    )
    
    # Create simplified mediation task (keeping original logic for now)
    mediation_task = Task(
        description=f"""
        MEDIATION ACTIVATION: You are called in when buyer and seller cannot reach agreement.
        
        CONTEXT (From Playbooks):
        - Contract type: {contract_type}
        - Buyer Profile: Credit Worthiness 8/10, Recurring Customer ({buyer_risk_profile.upper().replace('_', ' ')} RISK)
        - Buyer Target Price: ${buyer_target_price:,} (Max Budget: ${buyer_max_budget:,})
        - Seller Starting Price: ${seller_starting_price:,} (Target: ${seller_target_price:,}, Walk-Away: ${seller_walkaway_price:,})
        - Potential Deal Zone: ${seller_walkaway_price:,} - ${buyer_max_budget:,} (narrow overlap)
        
        FINAL DECISION REQUIRED:
        After analysis, you MUST provide a clear conclusion:
        - If compromise is viable within constraints: "DEAL AGREED" with final terms
        - If no viable compromise exists: "NEGOTIATION FAILED" with explanation
        
        Output: JSON with {{neutral_assessment, market_analysis, overlap_zone_analysis, recommended_terms, value_add_proposals, FINAL_DECISION, final_terms_or_reason}}
        """,
        agent=MediatorAgent,
        expected_output="Neutral mediation analysis with definitive DEAL AGREED or NEGOTIATION FAILED conclusion based on realistic playbook constraints"
    )
    
    # Create crew with dynamic tasks
    crew = Crew(
        agents=[BuyerAgent, SellerAgent, MediatorAgent],  # Subordinate agents under manager
        tasks=[buyer_opening_task, seller_response_task, orchestration_task, mediation_task],
        manager_agent=OrchestratorAgent,  # Custom manager agent with delegation authority
        process=Process.hierarchical,  # Enable hierarchical delegation and task management
        planning=False,  # Disable planning to avoid OpenAI dependency
        verbose=True,
        memory=False
    )
    
    # Calculate realistic gap based on playbook values
    gap_percentage = abs(seller_starting_price - buyer_target_price) / buyer_target_price * 100
    
    scenario = f"""
    NEGOTIATION SCENARIO - Heavy Equipment Purchase:
    - Contract Type: {contract_type}
    - Buyer Target Price: ${buyer_target_price:,} (Max Budget: ${buyer_max_budget:,} - not revealed)
    - Seller Starting Price: ${seller_starting_price:,} (Target: ${seller_target_price:,}, Walk-Away: ${seller_walkaway_price:,})
    - Initial Gap: {gap_percentage:.1f}%
    - Buyer Profile: Credit Worthiness 8/10, Recurring Customer ({buyer_risk_profile.upper().replace('_', ' ')} RISK)
    - Playbook Sources: 
      * Buyer: {'JSON loaded successfully' if buyer_playbook else 'Fallback values used'}
      * Seller: {'JSON loaded successfully' if seller_playbook else 'Fallback values used'}
    
    Requirements:
    - Equipment: Heavy construction machinery
    - Warranty: 3-year comprehensive preferred
    - Payment Terms: Net 60 preferred (Net 30 acceptable)
    - Delivery: Free delivery preferred
    - Service: First 2-3 maintenance services included preferred
    
    Begin negotiation process...
    """
    
    print("🎭 Multi-Agent Contract Negotiation - Heavy Equipment")
    print("=" * 60)
    print(scenario)
    print("=" * 60)
    
    # Set the negotiation context as inputs
    inputs = {
        "contract_type": contract_type,
        "buyer_risk_profile": buyer_risk_profile,
        "requirements": {
            "equipment_type": "heavy_construction_machinery",
            "warranty_preferred": "3-year comprehensive",
            "payment_terms_preferred": "Net 60",
            "delivery_preferred": "free_delivery",
            "service_preferred": "first_2-3_maintenance_included"
        }
    }
    
    # Execute negotiation with inputs
    result = crew.kickoff(inputs=inputs)
    
    print("\n🎯 NEGOTIATION COMPLETE!")
    print("=" * 60)
    
    # Try to extract and display the final decision clearly
    try:
        result_str = str(result)
        
        # Check for different outcome types
        if "DEAL SUCCESSFULLY NEGOTIATED" in result_str:
            print("✅ OUTCOME: DEAL SUCCESSFULLY NEGOTIATED")
            print("🎉 The negotiation was successful! Both parties reached agreement.")
        elif "MEDIATION REQUIRED" in result_str:
            print("🟡 OUTCOME: MEDIATION REQUIRED")
            print("⚖️  The negotiation gap was too large - escalated to mediation.")
        elif "DEAL AGREED" in result_str:
            print("✅ OUTCOME: DEAL AGREED")
            print("🎉 Successful agreement reached!")
        elif "NEGOTIATION FAILED" in result_str:
            print("❌ OUTCOME: NEGOTIATION FAILED")
            print("💔 No agreement could be reached.")
        else:
            print("⚠️  OUTCOME: UNCLEAR - Check detailed results")
            print("🔍 The outcome format may need review.")
            
        print("\nDETAILED RESULTS:")
        print(result)
    except Exception as e:
        print(f"⚠️  Error parsing results: {e}")
        print(f"Raw result: {result}")
    
    return result

if __name__ == "__main__":
    # Test negotiation scenario based on actual playbook values
    print("🚀 Testing Ollama Integration with Realistic Playbook Scenarios...")
    
    print("\n📋 Standard Playbook - Heavy Equipment Purchase (Low Risk Buyer)")
    print("Expected: Either negotiated deal or mediation (depends on negotiation skill)")
    print("Buyer wants $160K, Seller starts at $195K, overlap zone: $172.5K-$170K")
    run_negotiation("heavy_equipment", 160000, 195000, "low_risk")
    
    print("\n✅ Playbook-based scenario complete!")