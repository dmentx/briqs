#!/usr/bin/env python3
"""
Simple Multi-Agent Negotiation System with Ollama
Fast setup and execution for contract negotiations
"""

from crewai import Agent, LLM, Task, Crew

# 1. Shared Ollama LLM - auto-detects available model
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

# 2. ====== NEGOTIATION AGENTS ======
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
    role="Negotiation Orchestrator",
    goal="Manage negotiation process, coordinate agents, and ensure fair outcomes.",
    backstory="Expert process manager with deep understanding of negotiation dynamics.",
    llm=fast_llm
)

MediatorAgent = Agent(
    role="Neutral Mediator",
    goal="Resolve conflicts and provide unbiased recommendations when needed.",
    backstory="Impartial negotiation expert specializing in conflict resolution.",
    llm=fast_llm
)

# 3. ====== NEGOTIATION TASKS ======
buyer_task = Task(
    description="""
    Analyze the {contract_type} contract requirements and develop a negotiation strategy.
    Your budget limit is ${buyer_budget}. The seller is asking ${seller_price}.
    Consider: budget constraints, risk factors, and relationship priorities.
    Requirements: {requirements}
    Output: JSON with {position, rationale, red_lines, flexibility_areas}
    """,
    agent=BuyerAgent,
    expected_output="JSON negotiation position for {contract_type}"
)

seller_task = Task(
    description="""
    Evaluate the {contract_type} opportunity and create a sales strategy.
    Your asking price is ${seller_price}. The buyer's budget is ${buyer_budget}.
    Consider: pricing optimization, value proposition, and deal structure.
    Requirements to deliver: {requirements}
    Output: JSON with {offer, value_points, concession_areas, walkaway_terms}
    """,
    agent=SellerAgent,
    expected_output="JSON sales position for {contract_type}"
)

orchestrator_task = Task(
    description="""
    Coordinate the {contract_type} negotiation between buyer and seller positions.
    Current gap: Buyer budget ${buyer_budget} vs Seller price ${seller_price}.
    Identify gaps, facilitate discussion, and guide toward resolution.
    Contract requirements: {requirements}
    Output: JSON with {gap_analysis, recommendations, next_steps}
    """,
    agent=OrchestratorAgent,
    expected_output="JSON orchestration plan for {contract_type}"
)

mediation_task = Task(
    description="""
    If needed, provide neutral analysis for the {contract_type} negotiation.
    Analyze the gap between ${buyer_budget} (buyer) and ${seller_price} (seller).
    Consider market fairness, mutual benefit, and deal viability for: {requirements}
    Output: JSON with {assessment, recommendations, fair_terms}
    """,
    agent=MediatorAgent,
    expected_output="JSON mediation recommendations for {contract_type}"
)

# 4. ====== CREW SETUP ======
crew = Crew(
    agents=[BuyerAgent, SellerAgent, OrchestratorAgent, MediatorAgent],
    tasks=[buyer_task, seller_task, orchestrator_task, mediation_task],
    verbose=True,
    memory=False
)

# 5. ====== NEGOTIATION SCENARIO ======
def run_negotiation(contract_type="software_license", buyer_budget=10000, seller_price=12000):
    """Run a complete negotiation scenario."""
    
    scenario = f"""
    NEGOTIATION SCENARIO:
    - Contract Type: {contract_type}
    - Buyer Budget: ${buyer_budget:,}
    - Seller Asking Price: ${seller_price:,}
    - Gap: {abs(seller_price - buyer_budget) / buyer_budget * 100:.1f}%
    
    Requirements:
    - License Duration: 2 years
    - Support Level: Premium
    - User Count: 100
    - Payment Terms: Quarterly
    
    Begin negotiation process...
    """
    
    print("🎭 Multi-Agent Contract Negotiation")
    print("=" * 50)
    print(scenario)
    print("=" * 50)
    
    # Set the negotiation context as inputs
    inputs = {
        "contract_type": contract_type,
        "buyer_budget": buyer_budget,
        "seller_price": seller_price,
        "requirements": {
            "license_duration": "2 years",
            "support_level": "premium", 
            "user_count": 100,
            "payment_terms": "quarterly"
        }
    }
    
    # Execute negotiation with inputs
    result = crew.kickoff(inputs=inputs)
    
    print("\n🎯 NEGOTIATION COMPLETE!")
    print("=" * 50)
    return result

if __name__ == "__main__":
    # Quick test scenarios
    print("🚀 Testing Ollama Integration...")
    
    # Test 1: Close gap scenario
    print("\n📋 Scenario 1: Close Gap Negotiation")
    run_negotiation("software_license", 10000, 11000)
    
    # Test 2: Challenging gap scenario  
    print("\n📋 Scenario 2: Challenging Gap Negotiation")
    run_negotiation("consulting_services", 15000, 20000)
    
    print("\n✅ All scenarios complete!") 