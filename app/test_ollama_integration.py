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

# 3. ====== SEQUENTIAL NEGOTIATION TASKS ======
buyer_opening_task = Task(
    description="""
    As the buyer representative, make your opening offer for the {contract_type}.
    
    CONTEXT:
    - Your maximum budget: ${buyer_budget}
    - Seller's asking price: ${seller_price}
    - Contract requirements: {requirements}
    
    STRATEGY:
    1. Start with a reasonable but strategic opening offer (typically 70-80% of your max budget)
    2. Justify your offer with market research and budget constraints
    3. Highlight your commitment to the deal
    4. Include non-price terms that add value (payment schedule, contract length, etc.)
    
    OPENING OFFER GUIDELINES:
    - Be professional and respectful
    - Show genuine interest in the deal
    - Leave room for negotiation
    - Include rationale for your price point
    
    Output: JSON with {opening_offer_price, offer_justification, payment_terms, non_price_benefits, commitment_level}
    """,
    agent=BuyerAgent,
    expected_output="Professional opening offer with price and terms for {contract_type}"
)

seller_response_task = Task(
    description="""
    As the seller representative, respond to the buyer's offer presented by the orchestrator.
    
    CONTEXT:
    - Your target price: ${seller_price}
    - Contract requirements to deliver: {requirements}
    - Buyer's offer details: (will be provided by orchestrator)
    
    RESPONSE STRATEGY:
    1. Evaluate the buyer's offer against your minimum acceptable terms
    2. If offer is too low, make a counter-offer that moves toward agreement
    3. Emphasize value proposition and justify your pricing
    4. Look for creative ways to bridge the gap (terms, scope, timeline)
    5. Decide if offer is acceptable or requires negotiation
    
    DECISION FRAMEWORK:
    - If buyer's offer is within 15% of your target: Consider accepting or minor counter
    - If gap is 15-30%: Make strategic counter-offer 
    - If gap is >30%: Counter with significant justification or consider walking away
    
    Output: JSON with {response_decision, counter_offer_price, value_justification, alternative_terms, willingness_to_negotiate}
    """,
    agent=SellerAgent,
    expected_output="Professional response to buyer's offer with counter-proposal or acceptance"
)

orchestration_task = Task(
    description="""
    As the Negotiation Manager, orchestrate the sequential negotiation process.
    
    CONTEXT:
    - Contract type: {contract_type}
    - Buyer budget: ${buyer_budget}
    - Seller price: ${seller_price}
    - Requirements: {requirements}
    
    ORCHESTRATION PROCESS:
    1. Receive buyer's opening offer from BuyerAgent
    2. Present buyer's offer to SellerAgent (without revealing buyer's max budget)
    3. Receive seller's response/counter-offer
    4. Analyze if gap can be bridged through negotiation
    5. Facilitate 1-2 rounds of back-and-forth if needed
    6. Determine final outcome: Agreement or Mediation needed
    
    AGREEMENT CRITERIA:
    - Both parties express acceptance of terms
    - Price and terms are within acceptable ranges
    - No major outstanding issues
    
    MEDIATION CRITERIA:
    - Price gap remains >20% after 2 rounds
    - Fundamental disagreement on key terms
    - Either party approaching walkaway position
    
    REQUIRED FINAL CONCLUSION:
    You MUST end with one of these clear outcomes:
    - "DEAL SUCCESSFULLY NEGOTIATED" with final agreed terms
    - "MEDIATION REQUIRED" and delegate to MediatorAgent
    
    Output: JSON with {buyer_offer, seller_response, negotiation_rounds, gap_analysis, FINAL_CONCLUSION, agreed_terms_or_mediation_reason}
    """,
    agent=OrchestratorAgent,
    expected_output="Sequential negotiation management with clear deal conclusion or mediation trigger"
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
    
    FINAL DECISION REQUIRED:
    After analysis, you MUST provide a clear conclusion:
    - If compromise is viable within buyer's budget: "DEAL AGREED" with final terms
    - If no viable compromise exists: "NEGOTIATION FAILED" with explanation
    
    MEDIATION PRINCIPLES:
    - Remain completely neutral and unbiased
    - Focus on mutual benefit and long-term relationship
    - Consider market rates and fair value exchange
    - Propose creative solutions (payment terms, scope adjustments, etc.)
    
    Output: JSON with {neutral_assessment, market_analysis, recommended_terms, compromise_options, FINAL_DECISION, final_terms_or_reason}
    """,
    agent=MediatorAgent,
    expected_output="Neutral mediation analysis with definitive DEAL AGREED or NEGOTIATION FAILED conclusion"
)

# 4. ====== HIERARCHICAL CREW SETUP ======
crew = Crew(
    agents=[BuyerAgent, SellerAgent, MediatorAgent],  # Subordinate agents under manager
    tasks=[buyer_opening_task, seller_response_task, orchestration_task, mediation_task],
    manager_agent=OrchestratorAgent,  # Custom manager agent with delegation authority
    process=Process.hierarchical,  # Enable hierarchical delegation and task management
    planning=False,  # Disable planning to avoid OpenAI dependency
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
    # Test different negotiation scenarios
    print("🚀 Testing Ollama Integration with Multiple Scenarios...")
    
    # Test 1: Successful deal scenario (small gap)
    print("\n📋 Scenario 1: Likely Success - Software License")
    print("Expected: DEAL SUCCESSFULLY NEGOTIATED")
    run_negotiation("software_license", 10000, 10500)  # 5% gap - should succeed
    
    # Test 2: Moderate gap scenario (requires negotiation)
    print("\n📋 Scenario 2: Moderate Gap - Consulting Services")  
    print("Expected: Either deal or mediation depending on negotiation")
    run_negotiation("consulting_services", 15000, 17500)  # 16% gap - could go either way
    
    # Test 3: Challenging gap scenario (likely needs mediation)
    print("\n📋 Scenario 3: Challenging Gap - Equipment Purchase")
    print("Expected: MEDIATION REQUIRED")
    run_negotiation("equipment_purchase", 25000, 35000)  # 40% gap - likely needs mediation
    
    print("\n✅ All scenarios complete!") 