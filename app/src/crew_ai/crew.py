import json
import os
import yaml  # Make sure to install with: pip install pyyaml

from crewai import Agent, Task, Crew, Process

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
    # (Your original `build_buyer_task_description` logic is excellent, so we'll just wrap it)
    base_description = "..." # Imagine your full description builder logic here. For brevity.
    
    # For this example, let's use a simplified version. You should keep your detailed one.
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

def build_seller_task_description(playbook, buyer_risk_profile, buyer_message):
    """Builds the seller's task based on the buyer's message."""
    # (Your original `build_seller_task_description` logic is excellent)
    # For this example, a simplified version. You should keep your detailed one.
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

# ====== NEGOTIATION SCENARIO ======
def run_negotiation(contract_type="heavy_equipment", buyer_risk_profile="low_risk", max_rounds=4):
    """Run a complete negotiation scenario based on buyer and seller playbooks."""
    
    if not ollama_llm:
        print("❌ Cannot run negotiation without a working LLM connection.")
        return

    # 1. Load Playbooks and Agent Configs
    print("📚 Loading playbooks and agent configurations...")
    buyer_playbook = load_playbook("buyer.json")
    seller_playbook = load_playbook("seller.json")
    
    with open('agents.yaml', 'r') as f:
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

    # 3. Negotiation Loop (Python as Orchestrator)
    print("\n🎭 Starting Negotiation...")
    print("=" * 50)
    
    negotiation_history = []
    last_message = None

    # --- Round 1: Buyer's Opening Offer ---
    print("\n--- ROUND 1: Buyer makes an opening offer ---")
    buyer_task = Task(
      description=build_buyer_task_description(buyer_playbook, contract_type),
      expected_output="A JSON object containing the opening offer, price, and justifications.",
      agent=buyer_agent
    )
    
    # Execute the first task
    opening_crew = Crew(agents=[buyer_agent], tasks=[buyer_task], process=Process.sequential, verbose=1)
    last_message = opening_crew.kickoff()
    negotiation_history.append(f"BUYER: {last_message}")
    print("\n" + "="*20 + " BUYER'S OPENING OFFER " + "="*20)
    print(last_message)


    # --- Rounds 2 to max_rounds: Back and Forth ---
    for i in range(2, max_rounds * 2): # Loop for multiple turns (seller, then buyer, etc.)
        
        # Check for deal completion
        if "DEAL REACHED" in last_message.upper() or "NEGOTIATION FAILED" in last_message.upper():
            print("\n🏁 Negotiation has concluded.")
            break

        turn = (i // 2) + 1
        
        # Determine whose turn it is
        if i % 2 == 0: # Seller's turn
            print(f"\n--- ROUND {turn}: Seller responds ---")
            
            seller_task = Task(
                description=build_seller_task_description(seller_playbook, buyer_risk_profile, last_message),
                expected_output="A JSON object with a counter-offer or acceptance, and justification.",
                agent=seller_agent
            )
            crew = Crew(agents=[seller_agent], tasks=[seller_task], process=Process.sequential, verbose=1)
            last_message = crew.kickoff()
            negotiation_history.append(f"SELLER: {last_message}")
            print("\n" + "="*20 + " SELLER'S RESPONSE " + "="*20)
            print(last_message)
        
        else: # Buyer's turn
            print(f"\n--- ROUND {turn}: Buyer responds ---")

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


    print("\n🎯 NEGOTIATION COMPLETE!")
    print("=" * 50)
    print("\nFull Negotiation History:")
    for message in negotiation_history:
        print(f"- {message}\n")
    
    print("\nFinal Outcome:")
    print(last_message)


if __name__ == "__main__":
    run_negotiation()
