import json
import os
import yaml  
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain.tools import tool
from typing import Type
from pydantic import BaseModel, Field


# def get_ollama_llm():
#     """Get llama4 model with context size."""
#     import requests
#     try:
#         response = requests.get("http://localhost:11434/api/tags", timeout=3)
#         if response.status_code == 200:
#             models = response.json()
#             model_names = [model["name"] for model in models.get("models", [])]
            
#             if "llama4:16x17b" in model_names:
#                 return LLM(
#                     model="ollama/llama4:16x17b", 
#                     base_url="http://localhost:11434",
#                     num_ctx=8192  # Context size parameter
#                 )
#     except Exception as e:
#         print(f"⚠️  Error connecting to Ollama: {e}")
#     raise Exception("llama4:16x17b model not found. Please ensure it's installed in Ollama.")

# try:
#     llm_llama4 = get_ollama_llm()
# except Exception:
#     llm_llama4 = None

# Configure Groq LLM with API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")

llm_llama4 = LLM(
    model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4
)


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
    
    Output your response as a JSON object with your offer and justification.
    """

    mediation_injection = ""
    if mediation_proposal:
        mediation_injection = f"""
        ********************************
        ** MEDIATOR'S INTERVENTION **
        A neutral mediator has reviewed the prior round's deadlock and proposed the following compromise:
        '{mediation_proposal}'
        
        You MUST address this proposal in your next response. You can choose to accept it, reject it, or use it as a basis for a new counter-offer.
        You should consider accepting the mediator's proposal because we are trying to reach a deal.

        You should really consider accepting the mediator's proposal because we are trying to reach a deal.
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

def build_seller_task_description(playbook, buyer_message, mediation_proposal=None):
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
    The buyer has a buyer risk profile with the following characteristics: {buyer_profile_desc}. Choose the correct contract terms based on the buyers risk score. Calculate the buyer risk profile based on 'credit worthiness' and 'recurring customer'. Buyer risk can be low_risk, medium_risk, high_risk.

    Your playbook details: 
    
    {criteria_text}
    {rules_text}
    {tradables_text}

    **Stick strictly to these rules.**
    **Do not make concessions that are not part of your tradables and/or acceptable contract terms.**

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

        You MUST address this proposal in your next response. You can choose to accept it, reject it, or use it as a basis for a new counter-offer.
        You should consider accepting the mediator's proposal because we are trying to reach a deal.

        You should really consider accepting the mediator's proposal because we are trying to reach a deal.
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

def adjudicate_round(last_buyer_offer: str, last_seller_response: str) -> str:
    """
    Uses a dedicated Adjudicator agent to determine if a deal has been reached.
    Always returns JSON string - either with agreed terms or continuation status.
    """
    print("\n⚖️  Adjudicator is checking for a deal...")
    
    # The adjudicator agent is simple and focused
    adjudicator_agent = Agent(
        role="Deal Adjudicator",
        goal="Analyze the last two messages in a negotiation to determine if an explicit deal has been reached and return a structured JSON response.",
        backstory="You are an impartial judge. Your only job is to compare an offer and a response to see if they are in perfect agreement. You are strict; a counter-offer is NOT a deal. You always return JSON format responses.",
        llm=llm_llama4,
        verbose=True
    )

    adjudicator_task = Task(
        description=f"""
        Analyze the following buyer offer and seller response.

        **Buyer's Last Offer:**
        ---
        {last_buyer_offer}
        ---

        **Seller's Response to that Offer:**
        ---
        {last_seller_response}
        ---

        **Your Task:**
        Has the seller explicitly and unconditionally accepted the buyer's exact offer?
        - If the seller's message is a clear, unconditional "I accept your offer", "Deal", or similar, AND it does not introduce new terms or change existing ones, then a deal has been reached.
        - If the seller's message, even if it contains the words "DEAL REACHED", is actually a COUNTER-OFFER with different terms (e.g., a different price, different payment terms), it is NOT a deal.

        **CRITICAL: YOU MUST RESPOND ONLY WITH VALID JSON FORMAT. NO OTHER TEXT.**
        
        If a deal IS reached, respond with ONLY this JSON structure:
        {{
          "status": "DEAL_REACHED",
          "price": "$X,XXX",
          "payment_terms": "...",
          "warranty": "...",
          "delivery": "...",
          "maintenance_services": "...",
          "additional_terms": "..."
        }}
        
        If NO deal is reached, respond with ONLY this JSON structure:
        {{
          "status": "NO_DEAL_REACHED",
          "reason": "Explain why no deal was reached - e.g., 'Seller made counter-offer with different price', 'Terms not fully agreed', etc.. State details of the negotiation and the reason for the stalemate. Be specific and detailed."
        }}
        
        DO NOT include any text before or after the JSON. DO NOT include explanations. ONLY return the JSON object.
        Extract the agreed terms from the buyer's offer and seller's acceptance when a deal is reached. Include all relevant terms discussed.
        """,
        agent=adjudicator_agent,
        expected_output="ONLY a valid JSON object with either deal terms or continuation status. No other text."
    )

    crew = Crew(agents=[adjudicator_agent], tasks=[adjudicator_task], process=Process.sequential)
    verdict = crew.kickoff()
    
    clean_verdict = verdict.raw.strip()
    print(f"⚖️  Adjudicator's Raw Response: {clean_verdict}")
    
    # Try to extract JSON from the response
    try:
        json_match = re.search(r'\{.*\}', clean_verdict, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Validate it's proper JSON
            parsed_json = json.loads(json_str)
            print("⚖️  Adjudicator's JSON Response: {json_str}")
            return json_str
        else:
            # If no JSON found, create a default continue response
            print("⚖️  No JSON found in response, using default continue")
            return '{"status": "CONTINUE", "reason": "Adjudicator did not return JSON format"}'
    except (json.JSONDecodeError, TypeError) as e:
        # If JSON parsing fails, return a default continue response
        print(f"⚖️  JSON parsing failed: {e}")
        return '{"status": "CONTINUE", "reason": "Adjudicator response parsing failed"}'

def run_final_mediation(negotiation_history: list[str], combined_playbook: dict, backup_terms: dict) -> dict:
    """Invokes a Mediator agent to analyze a failed negotiation and propose a final compromise."""
    print("\n" + "="*20 + " MEDIATION STAGE " + "="*20)
    print("⚖️  The main negotiation failed. A Chief Mediator is being called in...")

    history_str = "\n".join(negotiation_history)
    with open('src/config_crewai/agents.yaml', 'r') as f:
        agents_config = yaml.safe_load(f)
    mediator_agent = Agent(**agents_config['mediator_agent'], llm=llm_llama4, verbose=True)

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
def run_negotiation(max_rounds=4):
    """
    Runs a negotiation with a neutral Adjudicator checking each round,
    and a final mediation stage if no deal is reached.
    """
    if not llm_llama4: return

    # 1. Load configurations
    print("📚 Loading configurations...")
    combined_playbook = load_playbook("src/knowledge_base/excavator_seller4.json")
    backup_terms = load_playbook("src/knowledge_base/briqs_backup_terms_excavator.json")
    with open('src/config_crewai/agents.yaml', 'r') as f:
        agents_config = yaml.safe_load(f)
    if not all([combined_playbook, backup_terms, agents_config]): return
    print("✅ Configurations loaded.")

    
    # Give the agents the calculator tool
    buyer_agent = Agent(**agents_config['buyer_agent'], llm=llm_llama4, verbose=True)
    seller_agent = Agent(**agents_config['seller_agent'], llm=llm_llama4, verbose=True)

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
    
    # Main Loop with Adjudicator
    for round_number in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_number + 1} ---")
        
        last_buyer_message = last_message # Store the buyer's message for adjudication

        # Seller's Turn
        task = Task(description=build_seller_task_description(combined_playbook, last_buyer_message), agent=seller_agent, expected_output="A JSON object with your counter-offer and justification.")
        seller_response = Crew(agents=[seller_agent], tasks=[task]).kickoff()
        negotiation_history.append(f"SELLER: {seller_response}")
        print("\n" + "="*20 + " SELLER'S RESPONSE " + "="*20); print(seller_response)
        
        # Adjudicator's Turn to Check for a Deal
        verdict = adjudicate_round(last_buyer_message, seller_response)
        try:
            verdict_json = json.loads(verdict)
            if verdict_json.get("status") == "DEAL_REACHED":
                deal_reached = True
                print("\n🎉 DEAL REACHED!")
                print("=" * 50)
                print("\nFINAL AGREED TERMS:")
                print(json.dumps(verdict_json, indent=2))
                last_message = f"DEAL REACHED: {json.dumps(verdict_json, indent=2)}"
                negotiation_history.append(f"ADJUDICATOR: {last_message}")
                break
            else:
                # Continue negotiation - log the reason if provided
                reason = verdict_json.get("reason", "No specific reason provided")
                print(f"⚖️  Continuing negotiation: {reason}")
                print(f"⚖️  Adjudicator's assessment: {json.dumps(verdict_json, indent=2)}")
                negotiation_history.append(f"ADJUDICATOR: CONTINUE - {json.dumps(verdict_json, indent=2)}")
        except (json.JSONDecodeError, TypeError) as e:
            # If JSON parsing fails, assume continue
            print(f"⚖️  JSON parsing failed: {e}")
            print(f"⚖️  Raw response: {verdict}")
            default_continue = '{"status": "CONTINUE", "reason": "Adjudicator response parsing failed"}'
            negotiation_history.append(f"ADJUDICATOR: CONTINUE - {default_continue}")
            pass

        # If it's the last round, don't let the buyer respond again, just exit the loop
        if round_number == max_rounds:
            break

        # Buyer's Turn to respond to the counter-offer
        task = Task(description=build_buyer_task_description(combined_playbook, seller_response), agent=buyer_agent, expected_output="A JSON object with your offer and justification.")
        last_message = Crew(agents=[buyer_agent], tasks=[task]).kickoff()
        negotiation_history.append(f"BUYER: {last_message}")
        print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20); print(last_message)

    # 3. Mediation Phase (if no deal was reached in the main loop)
    if not deal_reached:
        print("\n🏁 Main negotiation concluded without a deal.")
        mediation_result = run_final_mediation(negotiation_history, combined_playbook, backup_terms)
        
        if mediation_result and mediation_result.get("decision") == "PROPOSE_COMPROMISE":
            # This part of your code for the final mediation round is already correct.
            proposal_data = mediation_result.get("proposal_text") or mediation_result.get("proposal")
            formatted_proposal = json.dumps(proposal_data, indent=2) if isinstance(proposal_data, dict) else str(proposal_data)
            negotiation_history.append(f"MEDIATOR: {formatted_proposal}")

            print("\n--- FINAL MEDIATION ROUND ---")
            
            final_buyer_task = Task(description=build_buyer_task_description(combined_playbook, "\n\nThe Mediator has made a final proposal. You should really consider accepting it very seriously.\n\n Mediator's offer: \n\n", formatted_proposal), agent=buyer_agent, expected_output="A JSON object with your final offer.")
            buyer_final_word = Crew(agents=[buyer_agent], tasks=[final_buyer_task]).kickoff()
            negotiation_history.append(f"BUYER (Final Word): {buyer_final_word}")
            print("\n" + "="*20 + " BUYER'S FINAL WORD " + "="*20); print(buyer_final_word)

            final_seller_task = Task(description=build_seller_task_description(combined_playbook, "\n\nYou will receive the buyer's final word. You should really consider accepting it very seriously. The buyer has made a final offer based on a mediation proposal, which is considered to be fair for both parties.\n\n" + str(buyer_final_word) + "\n\nMeditator's offer:\n\n", formatted_proposal), agent=seller_agent, expected_output="A JSON object with your final decision.")
            seller_final_word = Crew(agents=[seller_agent], tasks=[final_seller_task]).kickoff()
            negotiation_history.append(f"SELLER (Final Word): {seller_final_word}")
            print("\n" + "="*20 + " SELLER'S FINAL WORD " + "="*20); print(seller_final_word)
            
            # Final adjudication after mediation round
            print("\n--- FINAL ADJUDICATION ---")
            final_verdict = adjudicate_round(buyer_final_word, seller_final_word)
            
            try:
                final_verdict_json = json.loads(final_verdict)
                if final_verdict_json.get("status") == "DEAL_REACHED":
                    deal_reached = True
                    # Extract and display the final agreed terms
                    print("\n🎉 DEAL REACHED!")
                    print("=" * 50)
                    print("\nFINAL AGREED TERMS:")
                    
                    # The final_verdict now contains the JSON terms directly from the adjudicator
                    print(json.dumps(final_verdict_json, indent=2))
                    last_message = f"DEAL REACHED: {json.dumps(final_verdict_json, indent=2)}"
                    negotiation_history.append(f"ADJUDICATOR: {last_message}")
                else:
                    # No deal reached - get the reason from the JSON and display the full JSON
                    reason = final_verdict_json.get("reason", "No agreement reached after mediation")
                    print("\n❌ NEGOTIATION FAILED")
                    print("=" * 50)
                    print("\nADJUDICATOR'S FINAL ASSESSMENT:")
                    print(json.dumps(final_verdict_json, indent=2))
                    last_message = f"NEGOTIATION FAILED: {json.dumps(final_verdict_json, indent=2)}"
                    negotiation_history.append(f"ADJUDICATOR: {last_message}")
            except (json.JSONDecodeError, TypeError) as e:
                # If JSON parsing fails, assume no deal
                print("\n❌ NEGOTIATION FAILED")
                print("=" * 50)
                print(f"\nADJUDICATOR RESPONSE PARSING ERROR: {e}")
                print(f"RAW RESPONSE: {final_verdict}")
                last_message = f"NEGOTIATION FAILED: {{'status': 'CONTINUE', 'reason': 'Adjudicator response format unclear', 'raw_response': '{final_verdict}'}}"
                negotiation_history.append(f"ADJUDICATOR: {last_message}")
        else:
            failure_reason = mediation_result.get('proposal_text') or 'Mediator declared a stalemate.'
            failure_json = json.dumps({"status": "CONTINUE", "reason": failure_reason}, indent=2)
            last_message = f"NEGOTIATION FAILED: {failure_json}"
            negotiation_history.append(f"MEDIATOR: {last_message}")

    # 4. Final Outcome
    print("\n🎯 NEGOTIATION COMPLETE!")
    print("=" * 50)
    print("\nFull Negotiation History:")
    for entry in negotiation_history:
        print(f"- {entry}\n")
    
    print("\nFinal Outcome:")
    print("=" * 50)
    if last_message.startswith("DEAL REACHED:"):
        print("🎉 DEAL SUCCESSFULLY REACHED!")
        # Extract and display the JSON part nicely
        json_part = last_message.replace("DEAL REACHED: ", "")
        try:
            final_terms = json.loads(json_part)
            print("\nFinal Agreed Terms:")
            print(json.dumps(final_terms, indent=2))
        except (json.JSONDecodeError, TypeError):
            print(f"\n{last_message}")
    else:
        print("❌ NEGOTIATION FAILED")
        # Extract and display the JSON part nicely
        json_part = last_message.replace("NEGOTIATION FAILED: ", "")
        try:
            failure_details = json.loads(json_part)
            print("\nFailure Details:")
            print(json.dumps(failure_details, indent=2))
        except (json.JSONDecodeError, TypeError):
            print(f"\n{last_message}")
    print("=" * 50)


if __name__ == "__main__":
    run_negotiation()