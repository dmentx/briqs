import json
import os
import yaml  
import re

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

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

def build_buyer_task_description(playbook, contract_type, previous_message=None, mediation_proposal=None):
    """Builds the buyer's task, including any mediator proposals."""
    
    base_description = f"""
    You are the buyer. Your goal is to secure the best deal for a {contract_type}.
    Refer to your playbook for strategy, goals, and terms.
    Your playbook details: {json.dumps(playbook, indent=2)}
    
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
        ---
        This is the start of the negotiation. Make your aggressive but realistic opening offer based on your playbook.
        """

def build_seller_task_description(playbook, buyer_risk_profile, buyer_message, contract_type, mediation_proposal=None):
    """Builds the seller's task, including any mediator proposals."""

    base_description = f"""
    You are the seller. Your goal is to maximize revenue for a {contract_type}.
    The buyer has a '{buyer_risk_profile}' profile. Refer to your playbook.
    Your playbook details: {json.dumps(playbook, indent=2)}

    Output your response as a JSON object with your counter-offer and justification.
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

class MediationTool(BaseTool):
    name: str = "Mediation Tool"
    description: str = (
        "Use this tool when a negotiation has stalled on a specific, minor point (like a small price gap). "
        "It will provide a neutral, third-party compromise suggestion."
    )

    def _run(self, deadlock_topic: str, negotiation_history: str) -> str:
        """Invokes the Mediator agent to propose a compromise."""
        print("🛠️  Mediation Tool activated.")
        
        # Load mediator config
        with open('src/config_crewai/agents.yaml', 'r') as f:
            agents_config = yaml.safe_load(f)
        
        mediator = Agent(
            **agents_config['mediator_agent'],
            llm=ollama_llm,
            verbose=True
        )

        mediation_task = Task(
            description=f"""
            A negotiation is deadlocked. Your expertise is needed.
            The specific point of contention is: {deadlock_topic}.

            Here is the recent history of the negotiation:
            ---
            {negotiation_history}
            ---

            Your task: Based on the history, propose a single, concrete, and fair compromise to resolve the deadlock on the topic of '{deadlock_topic}'.
            Frame your response as a neutral third party. For example: 'A potential compromise could be...'.
            Your proposal should be clear, actionable, and address ONLY the deadlock topic.
            """,
            agent=mediator,
            expected_output="A string containing a clear and actionable compromise suggestion."
        )

        # Create a temporary crew to run the mediation
        crew = Crew(agents=[mediator], tasks=[mediation_task], process=Process.sequential)
        compromise_proposal = crew.kickoff()
        
        return compromise_proposal

mediation_tool = MediationTool()

def run_final_mediation(negotiation_history: list[str]) -> dict:
    """
    Invokes a Mediator agent to analyze a failed negotiation and propose a final compromise.
    """
    print("\n" + "="*20 + " MEDIATION STAGE " + "="*20)
    print("⚖️  The main negotiation failed. A Chief Mediator is being called in for a final attempt...")

    history_str = "\n".join(negotiation_history)
    with open('src/config_crewai/agents.yaml', 'r') as f:
        agents_config = yaml.safe_load(f)
    mediator_agent = Agent(**agents_config['mediator_agent'], llm=ollama_llm, verbose=True)

    mediation_task = Task(
        description=f"""
        You are a Chief Mediator. A contract negotiation has failed after multiple rounds.
        Your task is to review the entire negotiation history and determine if a final, reasonable compromise is possible.

        **Full Negotiation History:**
        ---
        {history_str}
        ---

        **Instructions:**
        1.  Analyze the final positions of the buyer and seller.
        2.  If you believe a compromise is achievable, formulate a "Final Compromise Proposal".
        3.  If you believe the positions are too far apart, state that clearly.

        **Output Format:**
        Your output MUST be a JSON object with EXACTLY these two keys:
        - "decision": Must be either "PROPOSE_COMPROMISE" or "DECLARE_STALEMATE".
        - "proposal_text": If "PROPOSE_COMPROMISE", this should contain the proposal (can be a string or a JSON object). If "DECLARE_STALEMATE", this contains the reason.
        
        IMPORTANT: Use "proposal_text" as the key name, NOT "proposal".
        """,
        agent=mediator_agent,
        expected_output='A JSON object with "decision" and "proposal_text" keys.'
    )

    crew = Crew(agents=[mediator_agent], tasks=[mediation_task], process=Process.sequential)
    result = crew.kickoff()
    
    print(f"⚖️  Mediator's Verdict: {result}")
    
    # --- THIS IS THE CORRECTED PARSING LOGIC ---
    try:
        # Use regex to find the JSON block, ignoring markdown fences
        json_match = re.search(r'\{.*\}', str(result), re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # If no JSON is found, fail gracefully
            return {"decision": "DECLARE_STALEMATE", "proposal_text": "Mediator did not return a valid JSON object."}
    except (json.JSONDecodeError, TypeError):
        return {"decision": "DECLARE_STALEMATE", "proposal_text": "Mediation failed to produce a valid outcome."}

def adjudicate_round(negotiation_history: list[str], buyer_playbook: dict, seller_playbook: dict) -> dict:
    """
    Uses a Lead Adjudicator agent to evaluate the negotiation.
    The Adjudicator can use a Mediator tool if a deadlock is found.
    """
    print("\n⚖️  Adjudicator is evaluating the round...")
    
    history_str = "\n".join(negotiation_history[-4:]) # Get last 2 rounds

    # Define the Adjudicator Agent with the new tool
    adjudicator_agent = Agent(
        role="Lead Adjudicator and Negotiation Analyst",
        goal="Analyze the negotiation status, declare a final outcome, or use a mediation tool to resolve a deadlock.",
        backstory="You are a master negotiator who oversees discussions. You can spot a deal, a failure, or a deadlock that needs mediation.",
        tools=[mediation_tool], # <-- The Mediator is now a tool!
        llm=ollama_llm,
        verbose=True
    )

    # A more advanced task for the Adjudicator
    adjudicator_task = Task(
        description=f"""
        You are the Lead Adjudicator. Analyze the latest turn in the negotiation based on the history provided.

        **Recent Negotiation History:**
        ---
        {history_str}
        ---

        Your job is to decide the state of the negotiation by choosing ONE of the following statuses:
        1.  **DEAL**: If the parties have explicitly agreed on all core terms (price, delivery, warranty).
        2.  **STALEMATE**: If a party has declared they are walking away or the negotiation is hopelessly stuck.
        3.  **MEDIATE**: If the parties are stuck in a loop on a *specific, minor* point (e.g., a price difference of less than 5%, or a single warranty term).
        4.  **CONTINUE**: If the negotiation is still progressing, even if slowly.

        **Instructions:**
        - If you decide **MEDIATE**, you MUST use the `Mediation Tool`. Your final answer should contain the proposal from the tool.
        - If you decide any other status, do NOT use the tool.

        Return your final decision as a JSON object with two keys:
        - "status": (must be "DEAL", "STALEMATE", "MEDIATE", or "CONTINUE")
        - "reasoning": (A brief explanation of your decision. If you chose MEDIATE, this MUST contain the compromise proposal from the tool).

        Example for MEDIATE:
        {{
          "status": "MEDIATE",
          "reasoning": "The Mediator suggests the following compromise on price: Split the difference at $168,500, with the buyer agreeing to Net 30 payment terms."
        }}
        """,
        agent=adjudicator_agent,
        expected_output="A JSON object with 'status' and 'reasoning' keys."
    )

    crew = Crew(agents=[adjudicator_agent], tasks=[adjudicator_task], process=Process.sequential)
    result = crew.kickoff()
    
    print(f"⚖️  Adjudicator's verdict: {result}")

    try:
        # Extract the actual text from the CrewOutput object
        result_text = str(result.raw) if hasattr(result, 'raw') else str(result)
        return json.loads(result_text)
    except json.JSONDecodeError:
        # Fallback if the LLM fails to return valid JSON
        return {"status": "CONTINUE", "reasoning": "Adjudicator failed to produce valid JSON, continuing negotiation."}


# ====== NEGOTIATION SCENARIO ======
def run_negotiation(contract_type="heavy_equipment", buyer_risk_profile="low_risk", max_rounds=4):
    """
    Runs a negotiation that can proceed to a final mediation stage if no deal is reached.
    """
    if not ollama_llm: return

    # 1. Load configurations
    print("📚 Loading configurations...")
    buyer_playbook = load_playbook("src/knowledge_base/briqs_buyer_playbook.json")
    seller_playbook = load_playbook("src/knowledge_base/briqs_seller_playbook_1.json")
    with open('src/config_crewai/agents.yaml', 'r') as f: agents_config = yaml.safe_load(f)
    if not all([buyer_playbook, seller_playbook, agents_config]): return
    print("✅ Configurations loaded.")

    buyer_agent = Agent(**agents_config['buyer_agent'], llm=ollama_llm, verbose=True)
    seller_agent = Agent(**agents_config['seller_agent'], llm=ollama_llm, verbose=True)

    # 2. Main Negotiation Phase
    print("\n🎭 Starting Main Negotiation...")
    negotiation_history = []
    deal_reached = False
    
    # Opening Move
    print("\n--- ROUND 1 ---")
    last_message = Crew(agents=[buyer_agent], tasks=[Task(description=build_buyer_task_description(buyer_playbook, contract_type), agent=buyer_agent, expected_output="A JSON object with your opening offer and justification.")]).kickoff()
    negotiation_history.append(f"BUYER: {last_message}")
    print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20); print(last_message)
    
    # Main Loop
    for round_number in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_number + 1} ---")
        
        task = Task(description=build_seller_task_description(seller_playbook, buyer_risk_profile, last_message, contract_type), agent=seller_agent, expected_output="A JSON object with your counter-offer and justification.")
        last_message = Crew(agents=[seller_agent], tasks=[task]).kickoff()
        negotiation_history.append(f"SELLER: {last_message}")
        print("\n" + "="*20 + " SELLER'S RESPONSE " + "="*20); print(last_message)
        if "DEAL REACHED" in str(last_message).upper(): deal_reached = True; break

        task = Task(description=build_buyer_task_description(buyer_playbook, contract_type, last_message), agent=buyer_agent, expected_output="A JSON object with your response and justification.")
        last_message = Crew(agents=[buyer_agent], tasks=[task]).kickoff()
        negotiation_history.append(f"BUYER: {last_message}")
        print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20); print(last_message)
        if "DEAL REACHED" in str(last_message).upper(): deal_reached = True; break
    
    # 3. Mediation Phase (if necessary)
    if not deal_reached:
        print("\n🏁 Main negotiation concluded without a deal.")
        mediation_result = run_final_mediation(negotiation_history)
        
        if mediation_result and mediation_result.get("decision") == "PROPOSE_COMPROMISE":
            # Handle both "proposal_text" and "proposal" keys for robustness
            proposal_data = mediation_result.get("proposal_text") or mediation_result.get("proposal")
            
            # --- THIS IS THE CORRECTED PROPOSAL HANDLING ---
            # Format the proposal nicely, whether it's a string or a dict/object
            if isinstance(proposal_data, dict):
                formatted_proposal = json.dumps(proposal_data, indent=2)
            else:
                formatted_proposal = str(proposal_data)

            negotiation_history.append(f"MEDIATOR: {formatted_proposal}")
            print("\n--- FINAL MEDIATION ROUND ---")
            
            # Final Buyer Turn with Mediator's Proposal
            print("BUYER is considering the final proposal...")
            final_buyer_task = Task(description=build_buyer_task_description(buyer_playbook, contract_type, "The Mediator has made a final proposal.", formatted_proposal), agent=buyer_agent, expected_output="A JSON object with your final decision on the mediator's proposal.")
            buyer_final_word = Crew(agents=[buyer_agent], tasks=[final_buyer_task]).kickoff()
            negotiation_history.append(f"BUYER (Final Word): {buyer_final_word}")
            print("\n" + "="*20 + " BUYER'S FINAL WORD " + "="*20); print(buyer_final_word)

            # Final Seller Turn to seal the deal
            print("SELLER is considering the final proposal and buyer's response...")
            final_seller_task = Task(description=build_seller_task_description(seller_playbook, buyer_risk_profile, buyer_final_word, contract_type, formatted_proposal), agent=seller_agent, expected_output="A JSON object with your final decision on the mediator's proposal.")
            last_message = Crew(agents=[seller_agent], tasks=[final_seller_task]).kickoff()
            negotiation_history.append(f"SELLER (Final Word): {last_message}")
        else:
            # Handle both possible key names for the failure reason
            failure_reason = mediation_result.get('proposal_text') or mediation_result.get('proposal') or 'Mediator declared a stalemate.'
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