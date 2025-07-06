import json
import os
import yaml  # Make sure to install with: pip install pyyaml

from crewai import Agent, Task, Crew, Process, LLM

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

def build_buyer_task_description(playbook, contract_type, previous_message=None):
    """Builds the buyer's task, optionally including the seller's last message."""

    base_description = f"""
    You are the buyer. Your goal is to secure the best deal for a {contract_type}.
    Refer to your playbook for strategy, goals, and terms.
    Your playbook details: {json.dumps(playbook, indent=2)}
    
    Output your response as a JSON object with your offer and justification.
    """

    if previous_message:
        return f"""
        {base_description}
        ---
        You have received a new message from the seller. Analyze their response and formulate your counter-offer based on your playbook.

        SELLER'S LAST MESSAGE:
        {previous_message}
        
        Now, provide your next response.
        """
    else: # This is the opening offer
        return f"""
        {base_description}
        ---
        This is the start of the negotiation. Make your aggressive but realistic opening offer based on your playbook.
        """

def build_seller_task_description(playbook, buyer_risk_profile, buyer_message, contract_type):
    """Builds the seller's task based on the buyer's message."""

    base_description = f"""
    You are the seller. Your goal is to maximize revenue for a {contract_type}.
    The buyer has a '{buyer_risk_profile}' profile.
    Refer to your playbook for rules, tradables, and terms.
    Your playbook details: {json.dumps(playbook, indent=2)}

    Output your response as a JSON object with your counter-offer and justification.
    If a deal is reached, start your response with "DEAL REACHED".
    """
    
    return f"""
    {base_description}
    ---
    You have received an offer from the buyer. Analyze it against your playbook, especially your walk-away price.
    Formulate your response or counter-offer.

    BUYER'S MESSAGE:
    {buyer_message}

    Now, provide your response.
    """

def check_deal_status(negotiation_history: list[str], buyer_playbook: dict, seller_playbook: dict) -> str:
    """
    Uses an LLM to adjudicate the state of the negotiation.
    Returns "DEAL", "NO_DEAL", or "STALEMATE".
    """
    print("\n⚖️  Adjudicator is checking the deal status...")
    
    # We only need the last few turns for context
    history_str = "\n".join(negotiation_history[-4:]) # Get last 2 rounds (buyer, seller, buyer, seller)

    # Simplified goals for the adjudicator prompt
    buyer_goal = buyer_playbook.get("Tradables", {}).get("Primary Goal", "No goal specified.")
    seller_goal = seller_playbook.get("Tradables", {}).get("Primary Goal", "No goal specified.")

    adjudicator_task = Task(
        description=f"""
        You are a neutral contract adjudicator. Your sole job is to determine if a deal has been reached.
        Analyze the following negotiation history.

        **Buyer's Primary Goal:** {buyer_goal}
        **Seller's Primary Goal:** {seller_goal}

        **Recent Negotiation History:**
        ---
        {history_str}
        ---

        Based ONLY on the explicit statements in the history, has a final agreement been reached on the core terms (price, payment, warranty, delivery)?
        - If both parties have explicitly agreed to the same final terms, respond with the single word: DEAL
        - If the negotiation has stalled, with one party stating they are walking away, respond with the single word: STALEMATE
        - Otherwise, if the negotiation is still ongoing with offers and counter-offers, respond with the single word: NO_DEAL

        Your output MUST be one of these three words and nothing else.
        """,
        agent=Agent(role="Adjudicator", goal="Determine negotiation outcome", backstory="A neutral third party.", llm=ollama_llm),
        expected_output="A single word: DEAL, NO_DEAL, or STALEMATE."
    )

    # Use a temporary crew to execute this single check
    crew = Crew(agents=[adjudicator_task.agent], tasks=[adjudicator_task], process=Process.sequential)
    result = crew.kickoff()
    
    print(f"⚖️  Adjudicator's verdict: {result}")
    # Extract the actual text from the CrewOutput object
    result_text = str(result.raw) if hasattr(result, 'raw') else str(result)
    return result_text.strip().upper()

# ====== NEGOTIATION SCENARIO ======
def run_negotiation(contract_type="heavy_equipment", buyer_risk_profile="low_risk", max_rounds=4):
    """Run a complete negotiation scenario with an adjudicator checking the status each round."""
    
    if not ollama_llm:
        print("❌ Cannot run negotiation without a working LLM connection.")
        return

    # 1. Load Playbooks and Agent Configs
    print("📚 Loading playbooks and agent configurations...")
    buyer_playbook = load_playbook("src/knowledge_base/briqs_buyer_playbook.json")
    seller_playbook = load_playbook("src/knowledge_base/briqs_seller_playbook_1.json")
    
    with open('src/config_crewai/agents.yaml', 'r') as f:
        agents_config = yaml.safe_load(f)

    if not all([buyer_playbook, seller_playbook, agents_config]):
        print("❌ Failed to load necessary configuration files. Exiting.")
        return

    print("✅ Playbooks and configs loaded successfully!")

    # 2. Create Agents
    buyer_agent = Agent(
        **agents_config['buyer_agent'],
        llm=ollama_llm,
        verbose=True
    )
    seller_agent = Agent(
        **agents_config['seller_agent'],
        llm=ollama_llm,
        verbose=True
    )

    # 3. Initialize Negotiation
    print("\n🎭 Starting Negotiation...")
    print("=" * 50)
    
    negotiation_history = []
    
    # --- Opening Move: Buyer's First Offer ---
    print("\n--- ROUND 1: Buyer makes an opening offer ---")
    buyer_task = Task(
      description=build_buyer_task_description(buyer_playbook, contract_type),
      expected_output="A JSON object containing the opening offer, price, and justifications.",
      agent=buyer_agent
    )
    
    crew = Crew(agents=[buyer_agent], tasks=[buyer_task], process=Process.sequential, verbose=1)
    last_message = crew.kickoff()
    negotiation_history.append(f"BUYER (Opening Offer): {last_message}")
    print("\n" + "="*20 + " BUYER'S OPENING OFFER " + "="*20)
    print(last_message)

    # --- Main Negotiation Loop ---
    for round_number in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_number + 1} ---")
        
        # Seller's Turn
        print(f"SELLER is responding...")
        seller_task = Task(
            description=build_seller_task_description(seller_playbook, buyer_risk_profile, last_message, contract_type),
            expected_output="A JSON object with a counter-offer or acceptance, and justification.",
            agent=seller_agent
        )
        crew = Crew(agents=[seller_agent], tasks=[seller_task], process=Process.sequential, verbose=1)
        last_message = crew.kickoff()
        negotiation_history.append(f"SELLER: {last_message}")
        print("\n" + "="*20 + " SELLER'S RESPONSE " + "="*20)
        print(last_message)

        # Buyer's Turn
        print(f"BUYER is responding...")
        buyer_task = Task(
            description=build_buyer_task_description(buyer_playbook, contract_type, last_message),
            expected_output="A JSON object with a counter-offer or acceptance, and justification.",
            agent=buyer_agent
        )
        crew = Crew(agents=[buyer_agent], tasks=[buyer_task], process=Process.sequential, verbose=1)
        last_message = crew.kickoff()
        negotiation_history.append(f"BUYER: {last_message}")
        print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20)
        print(last_message)
        
        # Adjudicator's Turn to Check for Deal
        status = check_deal_status(negotiation_history, buyer_playbook, seller_playbook)
        if status == "DEAL":
            print("\n🏁 Adjudicator confirms: DEAL REACHED!")
            last_message = "DEAL REACHED. Final terms agreed upon by both parties."
            break
        elif status == "STALEMATE":
            print("\n🏁 Adjudicator confirms: NEGOTIATION FAILED (STALEMATE).")
            last_message = "NEGOTIATION FAILED. Parties are at a stalemate."
            break
        else: # NO_DEAL
            print(" Negotiation continues...")
        
        if round_number == max_rounds:
            print(f"\n🏁 Reached maximum of {max_rounds} rounds. Negotiation concluded without a deal.")
            last_message = "NEGOTIATION FAILED. Maximum rounds reached."


    print("\n🎯 NEGOTIATION COMPLETE!")
    print("=" * 50)
    print("\nFull Negotiation History:")
    for entry in negotiation_history:
        print(f"- {entry}\n")
    
    print("\nFinal Outcome:")
    print(last_message)


if __name__ == "__main__":
    run_negotiation()
