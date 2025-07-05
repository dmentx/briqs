#!/usr/bin/env python3
"""
Simple Multi-Agent Negotiation System with Ollama
Fast setup and execution for contract negotiations
"""

from crewai import Agent, LLM, Task, Crew, Process, agent, crew
import json
import os

def get_ollama_llm():
    """Get llama4 model with context size."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json()
            model_names = [model["name"] for model in models.get("models", [])]
            
            if "llama4:16x17b" in model_names:
                return LLM(
                    model="ollama/llama4:16x17b", 
                    base_url="http://localhost:11434",
                    num_ctx=8192  # Context size parameter
                )
    except Exception as e:
        print(f"⚠️  Error connecting to Ollama: {e}")
    raise Exception("llama4:16x17b model not found. Please ensure it's installed in Ollama.")

llama_4 = get_ollama_llm()

# ====== PLAYBOOK LOADING ======
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

# ====== AGENTS ======
@agent
def buyer(self) -> Agent:
    return Agent(
        config=self.agents_config['buyer_agent'],
        verbose=True,
        llm=llama_4
    )

@agent
def seller(self) -> Agent:
    return Agent(
        config=self.agents_config['seller_agent'],
        verbose=True,
        llm=llama_4
    )

@agent
def orchestrator(self) -> Agent:
    return Agent(
        config=self.agents_config['orchestrator_agent'],
        verbose=True,
        llm=llama_4
    )

# ====== NEGOTIATION SCENARIO ======
def run_negotiation(contract_type="heavy_equipment", buyer_risk_profile="low_risk"):
    """Run a complete negotiation scenario based on buyer and seller playbooks."""
    
    # Load playbooks
    print("📚 Loading playbooks...")
    buyer_playbook = load_buyer_playbook()
    seller_playbook = load_seller_playbook()
    
    if not buyer_playbook:
        print("❌ Failed to load buyer playbook!")
        return None
    if not seller_playbook:
        print("❌ Failed to load seller playbook!")
        return None
    
    print("✅ Playbooks loaded successfully!")
    
    # Create tasks
    buyer_task = Task(
        description=build_buyer_task_description(buyer_playbook, contract_type),
        agent=BuyerAgent,
        expected_output="Opening offer with price, terms, and justification"
    )
    
    seller_task = Task(
        description=build_seller_task_description(seller_playbook, buyer_playbook, buyer_risk_profile),
        agent=SellerAgent,
        expected_output="Response to buyer's offer with counter-offer or acceptance"
    )
    
    orchestrator_task = Task(
        description=f"""
        You are the head negotiator coordinating this {contract_type} negotiation.
        
        Your role:
        1. Present buyer's opening offer to seller
        2. Present seller's response back to buyer
        3. Coordinate up to 4 rounds of back-and-forth
        4. Determine if a deal is reached or if negotiation fails
        
        Keep the conversation flowing and summarize key points.
        End with either "DEAL REACHED" or "NEGOTIATION FAILED".
        """,
        agent=OrchestratorAgent,
        expected_output="Coordinated negotiation with clear final outcome"
    )
    
    # Create crew
    @crew
    def crew(self) -> Crew:
        """Creates the research crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            planning=True,
            planning_llm=llama_4,
            verbose=True,
        )

    
    # Run negotiation
    print("🎭 Starting Negotiation...")
    print("=" * 50)
    
    result = crew.kickoff()
    
    print("\n🎯 NEGOTIATION COMPLETE!")
    print("=" * 50)
    print(result)
    
    return result

if __name__ == "__main__":
    print("🚀 Running Negotiation Scenario...")
    run_negotiation("heavy_equipment", "low_risk")