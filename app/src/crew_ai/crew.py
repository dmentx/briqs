import json
import os
import yaml  
import re

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain.tools import tool
from typing import Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    """Input schema for Calculator."""
    operation: str = Field(..., description="Mathematical expression to evaluate, e.g., '200*7' or '5000/2*10'")

class CalculatorTool(BaseTool):
    name: str = "calculate"
    description: str = "Useful to perform any mathematical calculations, like sum, minus, multiplication, division, etc. The input should be a mathematical expression, a couple examples are 200*7 or 5000/2*10"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, operation: str) -> str:
        """Execute the calculation."""
        try:
            result = eval(operation)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

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

try:
    ollama_llm = get_ollama_llm()
except Exception:
    ollama_llm = None


def load_playbook(playbook_path):
    """Generic function to load a playbook from a JSON file."""
    try:
        with open(playbook_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Playbook file not found: {playbook_path}")
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing playbook JSON in {playbook_path}: {e}")
    return None


# --- Task Description Builders ---

def build_buyer_task_description(playbook, previous_message=None, mediation_proposal=None):
    """Builds the buyer's task, including any mediator proposals."""

    # Extract product type from playbook
    product_type = playbook.get("result", {}).get("product_type", "PRODUCT").upper()
    
    # Extract buyer playbook from the nested structure
    buyer_playbook = playbook.get("result", {}).get("product_details", {}).get("buyer_playbook", {})
    
    # Extract key sections from the buyer playbook
    negotiation_strategy = buyer_playbook.get("Negotiation Strategy", [])
    tradables = buyer_playbook.get("Tradables", {})
    ideal_terms = buyer_playbook.get("Ideal & Acceptable Terms", {})
    
    # Format Negotiation Strategy
    strategy_text = ""
    if negotiation_strategy:
        strategy_text = "**NEGOTIATION STRATEGY:**\n"
        for i, rule in enumerate(negotiation_strategy, 1):
            strategy_text += f"{i}. {rule}\n"
    
    # Format Tradables
    tradables_text = ""
    if tradables:
        primary_goal = tradables.get("Primary Goal", "")
        if primary_goal:
            tradables_text += f"**PRIMARY GOAL:** {primary_goal}\n\n"
        
        # What you want to GET (high value to buyer)
        get_items = tradables.get("Get (High value to us)", [])
        if get_items:
            tradables_text += "**WHAT YOU WANT TO GET (High value to you):**\n"
            for item in get_items:
                tradables_text += f"• {item}\n"
            tradables_text += "\n"
        
        # What you're willing to GIVE (low cost to buyer)  
        give_items = tradables.get("Give (Low-cost to us)", [])
        if give_items:
            tradables_text += "**WHAT YOU'RE WILLING TO GIVE (Low-cost to you):**\n"
            for item in give_items:
                tradables_text += f"• {item}\n"
            tradables_text += "\n"
    
    # Format Ideal & Acceptable Terms
    terms_text = ""
    if ideal_terms:
        terms_text = "**IDEAL & ACCEPTABLE TERMS:**\n\n"
        
        # Price terms
        price_terms = ideal_terms.get("Price", {})
        if price_terms:
            terms_text += "**PRICE:**\n"
            target_price = price_terms.get("Target Purchase Price (USD)")
            max_budget = price_terms.get("Maximum Budget (USD)")
            ideal = price_terms.get("Ideal")
            fallback = price_terms.get("Fallback Position")
            
            if target_price:
                terms_text += f"• Target Purchase Price: ${target_price:,}\n"
            if max_budget:
                terms_text += f"• Maximum Budget: ${max_budget:,}\n"
            if ideal:
                terms_text += f"• Ideal: {ideal}\n"
            if fallback:
                terms_text += f"• Fallback Position: {fallback}\n"
            terms_text += "\n"
        
        # Payment terms
        payment_terms = ideal_terms.get("Payment Terms", {})
        if payment_terms:
            terms_text += "**PAYMENT TERMS:**\n"
            ideal = payment_terms.get("Ideal")
            fallback = payment_terms.get("Fallback Position")
            
            if ideal:
                terms_text += f"• Ideal: {ideal}\n"
            if fallback:
                terms_text += f"• Fallback Position: {fallback}\n"
            terms_text += "\n"
        
        # Warranty terms
        warranty_terms = ideal_terms.get("Warranty", {})
        if warranty_terms:
            terms_text += "**WARRANTY:**\n"
            ideal = warranty_terms.get("Ideal")
            fallback = warranty_terms.get("Fallback Position")
            
            if ideal:
                terms_text += f"• Ideal: {ideal}\n"
            if fallback:
                terms_text += f"• Fallback Position: {fallback}\n"
            terms_text += "\n"
        
        # Delivery terms
        delivery_terms = ideal_terms.get("Delivery", {})
        if delivery_terms:
            terms_text += "**DELIVERY:**\n"
            ideal = delivery_terms.get("Ideal")
            fallback = delivery_terms.get("Fallback Position")
            
            if ideal:
                terms_text += f"• Ideal: {ideal}\n"
            if fallback:
                terms_text += f"• Fallback Position: {fallback}\n"
            terms_text += "\n"
    
    base_description = f"""
    You are the buyer of {product_type}. Your goal is to secure the best deal based on your contract negotiation playbook.
    Refer to this playbook for 1. acceptable price range, 2. negotiation rules, 3. tradables that you want from the other party or are willing to give in exchange for concessions and 4. ideal/acceptable contract terms.

    Your playbook details: 
    
    {strategy_text}
    {tradables_text}
    {terms_text}

    **Stick strictly to these rules.**
    **Do not make concessions that are not part of your tradables and/or acceptable contract terms.**
    
    You can use the calculator tool to perform calculations.
    Output your response as a JSON object with your offer and justification.
    """

    mediation_injection = ""
    if mediation_proposal:
        mediation_injection = f"""
        ********************************
        ** MEDIATOR'S INTERVENTION **
        A neutral mediator has reviewed the prior round's deadlock and proposed the following compromise:
        '{mediation_proposal}'
        
        You MUST address this proposal in your next response. You can choose to accept it,
        reject it, or use it as a basis for a new counter-offer.
        ********************************
        """

    if previous_message:
        return f"""
        {base_description}
        {mediation_injection}

        ---
        Analyze the seller's last message below, keeping the mediator's proposal (if any) in mind.

        SELLER'S LAST MESSAGE:
        {previous_message}
        
        Now, provide your next response.
        """
    else: # Opening offer
        return f"""
        {base_description}
        """

def build_seller_task_description(playbook, buyer_risk_profile, buyer_message, mediation_proposal=None):
    """Builds the seller's task, including any mediator proposals."""

    # Extract product type from playbook
    product_type = playbook.get("result", {}).get("product_type", "PRODUCT").upper()
    
    # Extract buyer profile from playbook
    buyer_profile = playbook.get("result", {}).get("buyer_profile", {})
    credit_worthiness = buyer_profile.get("Credit Worthiness", "N/A")
    recurring_customer = buyer_profile.get("Recurring Customer", False)
    
    # Format buyer profile description
    buyer_profile_desc = f"Credit Worthiness: {credit_worthiness}, Recurring Customer: {'Yes' if recurring_customer else 'No'}"

    # Extract seller playbook from the nested structure
    seller_playbook = playbook.get("result", {}).get("product_details", {}).get("seller_playbook", {})
    
    # Extract key sections from the seller playbook
    criteria = seller_playbook.get("Criteria", {})
    negotiation_rules = seller_playbook.get("Negotiation rules", [])
    tradables = seller_playbook.get("Tradables", {})
    
    # Format Criteria section
    criteria_text = ""
    if criteria:
        criteria_text = "**PRICING CRITERIA:**\n"
        
        # Product pricing
        product_criteria = criteria.get("Product", {})
        if product_criteria:
            walk_away_price = product_criteria.get("Walk-Away-Price (USD)")
            target_price = product_criteria.get("Target Price (USD)")
            starting_price = product_criteria.get("Starting Price")
            
            if walk_away_price:
                criteria_text += f"• Walk-Away-Price: ${walk_away_price:,}\n"
            if target_price:
                criteria_text += f"• Target Price: ${target_price:,}\n"
            if starting_price:
                criteria_text += f"• Starting Price: ${starting_price:,}\n"
            criteria_text += "\n"
        
        # Buyer risk profile definitions
        buyer_criteria = criteria.get("Buyer", {})
        risk_definitions = buyer_criteria.get("risk_profile_definition", {})
        if risk_definitions:
            criteria_text += "**BUYER RISK PROFILE DEFINITIONS:**\n"
            for risk_level, definition in risk_definitions.items():
                criteria_text += f"• {risk_level.replace('_', ' ').title()}: {definition}\n"
            criteria_text += "\n"
    
    # Format Negotiation Rules
    rules_text = ""
    if negotiation_rules:
        rules_text = "**NEGOTIATION RULES:**\n"
        for i, rule in enumerate(negotiation_rules, 1):
            rules_text += f"{i}. {rule}\n"
        rules_text += "\n"
    
    # Format Tradables
    tradables_text = ""
    if tradables:
        primary_goal = tradables.get("Primary Goal", "")
        if primary_goal:
            tradables_text += f"**PRIMARY GOAL:** {primary_goal}\n\n"
        
        # What you're willing to GIVE (low cost to seller)
        give_items = tradables.get("Give (Low-cost to us)", [])
        if give_items:
            tradables_text += "**WHAT YOU'RE WILLING TO GIVE (Low-cost to you):**\n"
            for item in give_items:
                tradables_text += f"• {item}\n"
            tradables_text += "\n"
        
        # What you want to GET (high value to seller)
        get_items = tradables.get("Get (High value to us)", [])
        if get_items:
            tradables_text += "**WHAT YOU WANT TO GET (High value to you):**\n"
            for item in get_items:
                tradables_text += f"• {item}\n"
            tradables_text += "\n"
        
        # Ideal & Acceptable Terms by risk level
        ideal_terms = tradables.get("Ideal & Acceptable Terms", {})
        if ideal_terms:
            tradables_text += "**IDEAL & ACCEPTABLE TERMS BY BUYER RISK LEVEL:**\n\n"
            
            # High risk buyer terms
            high_risk_terms = ideal_terms.get("High risk buyer", {})
            if high_risk_terms:
                tradables_text += "**HIGH RISK BUYER:**\n"
                
                payment_terms = high_risk_terms.get("Payment Terms", {})
                if payment_terms:
                    tradables_text += "• Payment Terms:\n"
                    goal = payment_terms.get("Goal")
                    fallback = payment_terms.get("Fallback Position")
                    if goal:
                        tradables_text += f"  - Goal: {goal}\n"
                    if fallback:
                        tradables_text += f"  - Fallback: {fallback}\n"
                
                collateral_terms = high_risk_terms.get("Collateral for Payment Default", {})
                if collateral_terms:
                    tradables_text += "• Collateral for Payment Default:\n"
                    goal = collateral_terms.get("Goal")
                    fallback = collateral_terms.get("Fallback Position")
                    if goal:
                        tradables_text += f"  - Goal: {goal}\n"
                    if fallback:
                        tradables_text += f"  - Fallback: {fallback}\n"
                tradables_text += "\n"
            
            # Medium risk buyer terms
            medium_risk_terms = ideal_terms.get("Medium risk buyer", {})
            if medium_risk_terms:
                tradables_text += "**MEDIUM RISK BUYER:**\n"
                
                payment_terms = medium_risk_terms.get("Payment Terms", {})
                if payment_terms:
                    tradables_text += "• Payment Terms:\n"
                    goal = payment_terms.get("Goal")
                    fallback = payment_terms.get("Fallback Position")
                    if goal:
                        tradables_text += f"  - Goal: {goal}\n"
                    if fallback:
                        tradables_text += f"  - Fallback: {fallback}\n"
                tradables_text += "\n"
            
            # Low risk buyer terms
            low_risk_terms = ideal_terms.get("Low risk buyer", {})
            if low_risk_terms:
                tradables_text += "**LOW RISK BUYER:**\n"
                goal = low_risk_terms.get("Goal")
                fallback = low_risk_terms.get("Fallback Position")
                if goal:
                    tradables_text += f"• Goal: {goal}\n"
                if fallback:
                    tradables_text += f"• Fallback: {fallback}\n"
                tradables_text += "\n"

    base_description = f"""
    You are the seller of {product_type}. Your goal is to secure the best deal based on your contract negotiation playbook.
    Refer to this playbook for 1. acceptable price range, 2. negotiation rules, 3. tradables that you want from the other party or are willing to give in exchange for concessions and 4. ideal/acceptable contract terms.
    The buyer has a '{buyer_risk_profile}' profile with the following characteristics: {buyer_profile_desc}. Choose the correct contract terms based on the buyers risk score.

    Your playbook details: 
    
    {criteria_text}
    {rules_text}
    {tradables_text}

    **Stick strictly to these rules.**
    **Do not make concessions that are not part of your tradables and/or acceptable contract terms.**

    You can use the calculator tool to perform calculations.

    Output your response as a JSON object with your counter-offer and justification.
    If a deal is reached, start your response with "DEAL REACHED".
    """
    
    mediation_injection = ""
    if mediation_proposal:
        mediation_injection = f"""
        ********************************
        ** MEDIATOR'S INTERVENTION **
        A neutral mediator has reviewed the prior round's deadlock and proposed the following compromise:
        '{mediation_proposal}'

        You MUST address this proposal in your next response. You can choose to accept it,
        reject it, or use it as a basis for a new counter-offer.
        ********************************
        """

    return f"""
    {base_description}
    {mediation_injection}

    ---
    Analyze the buyer's offer below, keeping the mediator's proposal (if any) in mind.

    BUYER'S MESSAGE:
    {buyer_message}

    Now, provide your response.
    """

def run_final_mediation(negotiation_history: list[str], combined_playbook: dict, backup_terms: dict) -> dict:
    """Invokes a Mediator agent to analyze a failed negotiation and propose a final compromise."""
    print("\n" + "="*20 + " MEDIATION STAGE " + "="*20)
    print("⚖️  The main negotiation failed. A Chief Mediator is being called in...")

    history_str = "\n".join(negotiation_history)
    with open('src/config_crewai/agents.yaml', 'r') as f:
        agents_config = yaml.safe_load(f)
    mediator_agent = Agent(**agents_config['mediator_agent'], llm=ollama_llm, verbose=True)

    # The Mediator's task is very specific and detailed here.
    mediation_task = Task(
        description=f"""
        You are a mediator for a failed negotiation. Your task is to propose a final, acceptable compromise.
        
        **Combined Playbook (Buyer & Seller Goals):**
        {json.dumps(combined_playbook, indent=2)}

        **Backup Terms (for reference):**
        {json.dumps(backup_terms, indent=2)}

        **Full Negotiation History:**
        ---
        {history_str}
        ---

        Analyze the conflicts and playbooks. Propose a compromise or declare a stalemate.
        Your output MUST be a JSON object with two keys: "decision" and "proposal_text".
        - "decision": "PROPOSE_COMPROMISE" or "DECLARE_STALEMATE".
        - "proposal_text": Contains the proposal details or the reason for the stalemate.
        """,
        agent=mediator_agent,
        expected_output='A JSON object with "decision" and "proposal_text" keys.'
    )

    crew = Crew(agents=[mediator_agent], tasks=[mediation_task], process=Process.sequential)
    result = crew.kickoff()
    
    print(f"⚖️  Mediator's Verdict: {result}")
    
    try:
        json_match = re.search(r'\{.*\}', str(result), re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        return {"decision": "DECLARE_STALEMATE", "proposal_text": "Mediator did not return valid JSON."}
    except (json.JSONDecodeError, TypeError):
        return {"decision": "DECLARE_STALEMATE", "proposal_text": "Mediation failed to produce a valid outcome."}


# ====== NEGOTIATION SCENARIO ======
def run_negotiation(buyer_risk_profile="low_risk", max_rounds=4):
    """
    Runs a negotiation that can proceed to a final mediation stage if no deal is reached.
    """
    if not ollama_llm: return

    # 1. Load configurations
    print("📚 Loading configurations...")
    combined_playbook = load_playbook("src/knowledge_base/excavator_seller1.json")
    backup_terms = load_playbook("src/knowledge_base/briqs_backup_terms_excavator.json")
    with open('src/config_crewai/agents.yaml', 'r') as f: agents_config = yaml.safe_load(f)
    if not all([combined_playbook, backup_terms, agents_config]): return
    print("✅ Configurations loaded.")

    # Create calculator tool instance
    calculator_tool = CalculatorTool()
    
    buyer_agent = Agent(**agents_config['buyer_agent'], llm=ollama_llm, verbose=True, tools=[calculator_tool])
    seller_agent = Agent(**agents_config['seller_agent'], llm=ollama_llm, verbose=True, tools=[calculator_tool])

    # 2. Main Negotiation Phase
    print("\n🎭 Starting Main Negotiation...")
    negotiation_history = []
    deal_reached = False
    
    # Opening Move
    print("\n--- ROUND 1 ---")
    task = Task(description=build_buyer_task_description(combined_playbook), agent=buyer_agent, expected_output="A JSON object with your offer and justification.")
    last_message = Crew(agents=[buyer_agent], tasks=[task]).kickoff()
    negotiation_history.append(f"BUYER: {last_message}")
    print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20); print(last_message)
    
    # Main Loop
    for round_number in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_number + 1} ---")
        
        task = Task(description=build_seller_task_description(combined_playbook, buyer_risk_profile, last_message), agent=seller_agent, expected_output="A JSON object with your counter-offer and justification.")
        last_message = Crew(agents=[seller_agent], tasks=[task]).kickoff()
        negotiation_history.append(f"SELLER: {last_message}")
        print("\n" + "="*20 + " SELLER'S RESPONSE " + "="*20); print(last_message)
        if "DEAL REACHED" in str(last_message).upper(): deal_reached = True; break

        task = Task(description=build_buyer_task_description(combined_playbook, last_message), agent=buyer_agent, expected_output="A JSON object with your offer and justification.")
        last_message = Crew(agents=[buyer_agent], tasks=[task]).kickoff()
        negotiation_history.append(f"BUYER: {last_message}")
        print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20); print(last_message)
        if "DEAL REACHED" in str(last_message).upper(): deal_reached = True; break
    
    # 3. Mediation Phase (if necessary)
    if not deal_reached:
        print("\n🏁 Main negotiation concluded without a deal.")
        mediation_result = run_final_mediation(negotiation_history, combined_playbook, backup_terms)
        
        if mediation_result and mediation_result.get("decision") == "PROPOSE_COMPROMISE":
            proposal_data = mediation_result.get("proposal_text") or mediation_result.get("proposal")
            formatted_proposal = json.dumps(proposal_data, indent=2) if isinstance(proposal_data, dict) else str(proposal_data)
            negotiation_history.append(f"MEDIATOR: {formatted_proposal}")

            print("\n--- FINAL MEDIATION ROUND ---")
            
            final_buyer_task = Task(description=build_buyer_task_description(combined_playbook, "The Mediator has made a final proposal.", formatted_proposal), agent=buyer_agent, expected_output="A JSON object with your final offer and justification.")
            buyer_final_word = Crew(agents=[buyer_agent], tasks=[final_buyer_task]).kickoff()
            negotiation_history.append(f"BUYER (Final Word): {buyer_final_word}")
            print("\n" + "="*20 + " BUYER'S FINAL WORD " + "="*20); print(buyer_final_word)

            final_seller_task = Task(description=build_seller_task_description(combined_playbook, buyer_risk_profile, buyer_final_word, formatted_proposal), agent=seller_agent, expected_output="A JSON object with your final counter-offer and justification.")
            last_message = Crew(agents=[seller_agent], tasks=[final_seller_task]).kickoff()
            negotiation_history.append(f"SELLER (Final Word): {last_message}")
        else:
            failure_reason = mediation_result.get('proposal_text') or 'Mediator declared a stalemate.'
            last_message = f"NEGOTIATION FAILED: {failure_reason}"
            negotiation_history.append(f"MEDIATOR: {last_message}")

    # 4. Final Outcome
    print("\n🎯 NEGOTIATION COMPLETE!")
    print("=" * 50)
    print("\nFull Negotiation History:")
    for entry in negotiation_history:
        print(f"- {entry}\n")
    print("\nFinal Outcome:")
    print(last_message)


if __name__ == "__main__":
    run_negotiation()