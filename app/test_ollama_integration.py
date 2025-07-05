#!/usr/bin/env python3
"""
Simple Multi-Agent Negotiation System with Ollama
Fast setup and execution for contract negotiations
"""

from crewai import Agent, LLM, Task, Crew, Process

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

# 3. ====== HIERARCHICAL NEGOTIATION TASKS ======
buyer_negotiation_task = Task(
    description="""
    As the buyer representative, develop a comprehensive negotiation position for the {contract_type}.
    
    CONTEXT:
    - Your maximum budget: ${buyer_budget}
    - Seller's asking price: ${seller_price}
    - Contract requirements: {requirements}
    
    DELIVERABLES:
    1. Initial negotiation position with rationale
    2. Identify your red lines (non-negotiable items)
    3. Areas where you have flexibility
    4. Risk assessment and mitigation strategies
    
    Output: Detailed JSON with {position, budget_analysis, red_lines, flexibility_areas, risk_factors}
    """,
    agent=BuyerAgent,
    expected_output="Comprehensive buyer negotiation strategy for {contract_type}"
)

seller_negotiation_task = Task(
    description="""
    As the seller representative, create a strategic sales approach for the {contract_type}.
    
    CONTEXT:
    - Your target price: ${seller_price}
    - Buyer's stated budget: ${buyer_budget}
    - Requirements to deliver: {requirements}
    
    DELIVERABLES:
    1. Value-based pricing strategy with justification
    2. Key value propositions and differentiators
    3. Potential concession areas and alternatives
    4. Minimum acceptable terms (walkaway point)
    
    Output: Detailed JSON with {pricing_strategy, value_proposition, concession_options, walkaway_terms}
    """,
    agent=SellerAgent,
    expected_output="Comprehensive seller negotiation strategy for {contract_type}"
)

final_negotiation_task = Task(
    description="""
    As the Negotiation Manager, coordinate the final deal between buyer and seller positions.
    
    CONTEXT:
    - Contract type: {contract_type}
    - Buyer budget: ${buyer_budget}
    - Seller price: ${seller_price}
    - Requirements: {requirements}
    
    PROCESS:
    1. Review both buyer and seller positions
    2. Identify the negotiation gap and key issues
    3. Facilitate resolution through strategic guidance
    4. If positions are irreconcilable, delegate to mediator
    5. Finalize deal terms or recommend next steps
    
    DECISION CRITERIA for Mediation:
    - Price gap > 20% with no movement after 2 rounds
    - Fundamental disagreement on key terms
    - Either party approaching walkaway position
    
    MEDIATION DELEGATION:
    If mediation is needed, delegate to the MediatorAgent to provide neutral analysis and recommendations.
    
    Output: JSON with {final_deal, gap_analysis, resolution_strategy, mediation_activated}
    """,
    agent=OrchestratorAgent,
    expected_output="Final negotiation outcome with deal terms or escalation plan"
)

# Optional mediation task - activated by orchestrator when needed
mediation_task = Task(
    description="""
    MEDIATION ACTIVATION: You are called in when buyer and seller cannot reach agreement.
    
    CONTEXT:
    - Contract type: {contract_type}
    - Buyer budget: ${buyer_budget}
    - Seller price: ${seller_price}
    - Requirements: {requirements}
    
    NEUTRAL ANALYSIS:
    1. Review both positions objectively
    2. Research market fairness and industry standards
    3. Identify mutually beneficial solutions
    4. Propose compromise terms that respect both parties' core needs
    5. Provide recommendations for deal structure
    
    MEDIATION PRINCIPLES:
    - Remain completely neutral and unbiased
    - Focus on mutual benefit and long-term relationship
    - Consider market rates and fair value exchange
    - Propose creative solutions (payment terms, scope adjustments, etc.)
    
    Output: JSON with {neutral_assessment, market_analysis, recommended_terms, compromise_options}
    """,
    agent=MediatorAgent,
    expected_output="Neutral mediation analysis with compromise recommendations"
)

# 4. ====== HIERARCHICAL CREW SETUP ======
crew = Crew(
    agents=[BuyerAgent, SellerAgent, MediatorAgent],  # Subordinate agents under manager
    tasks=[buyer_negotiation_task, seller_negotiation_task, final_negotiation_task, mediation_task],
    manager_agent=OrchestratorAgent,  # Custom manager agent with delegation authority
    process=Process.hierarchical,  # Enable hierarchical delegation and task management
    planning=True,  # Enable planning for better task coordination
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