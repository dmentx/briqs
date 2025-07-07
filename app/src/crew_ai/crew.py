import json
import os
import yaml  
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ..models.core import ResultToAgent
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain.tools import tool
from typing import Type
from pydantic import BaseModel, Field


class NegotiationEngine:
    """
    A complete negotiation engine that handles buyer-seller negotiations with mediation.
    """
    
    def __init__(self, result_to_agent: ResultToAgent):
        """Initialize the negotiation engine with LLM and configurations."""
        self.result_to_agent = result_to_agent
        self.llm_llama4 = self._setup_llm()
        
    def _setup_llm(self):
        """Setup and return the LLM instance."""
        # Configure Groq LLM with API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")

        return LLM(
            model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.4
        )





    def build_buyer_task_description(self, previous_message=None, mediation_proposal=None):
        """Builds the buyer's task, including any mediator proposals."""

        # Extract product type from playbook with null checks
        product_type = self.result_to_agent.result.product_type if self.result_to_agent.result else "product"
        
        # Extract buyer playbook from the nested structure with null checks
        buyer_playbook = self.result_to_agent.result.product_details.buyer_playbook if (
            self.result_to_agent.result and 
            self.result_to_agent.result.product_details
        ) else None
        
        # Extract key sections from the buyer playbook with null checks
        negotiation_strategy = buyer_playbook.negotiation_strategy if buyer_playbook else None
        tradables = buyer_playbook.tradables if buyer_playbook else None
        ideal_terms = buyer_playbook.ideal_acceptable_terms if buyer_playbook else None
        
        # Format Negotiation Strategy
        strategy_text = ""
        if negotiation_strategy:
            strategy_text = "**NEGOTIATION STRATEGY:**\n"
            for i, rule in enumerate(negotiation_strategy, 1):
                strategy_text += f"{i}. {rule}\n"
        
        # Format Tradables
        tradables_text = ""
        if tradables:
            # Use the BuyerTradables model attribute for the primary goal
            if tradables.primary_goal:
                tradables_text += f"**PRIMARY GOAL:** {tradables.primary_goal}\n\n"
            
            # What you want to GET (high value to buyer)
            get_items = tradables.get_high_value_to_us
            if get_items:
                tradables_text += "**WHAT YOU WANT TO GET (High value to you):**\n"
                for item in get_items:
                    tradables_text += f"• {item}\n"
                tradables_text += "\n"
            
            # What you're willing to GIVE (low cost to buyer)  
            give_items = tradables.give_low_cost_to_us
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
            price_terms = ideal_terms.price
            if price_terms:
                terms_text += "**PRICE:**\n"
                target_price = price_terms.target_purchase_price_usd
                max_budget = price_terms.maximum_budget_usd
                ideal = price_terms.ideal
                fallback = price_terms.fallback_position
                
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
            payment_terms = ideal_terms.payment_terms
            if payment_terms:
                terms_text += "**PAYMENT TERMS:**\n"
                ideal = payment_terms.ideal
                fallback = payment_terms.fallback_position
                
                if ideal:
                    terms_text += f"• Ideal: {ideal}\n"
                if fallback:
                    terms_text += f"• Fallback Position: {fallback}\n"
                terms_text += "\n"
            
            # Warranty terms
            warranty_terms = ideal_terms.warranty
            if warranty_terms:
                terms_text += "**WARRANTY:**\n"
                ideal = warranty_terms.ideal
                fallback = warranty_terms.fallback_position
                
                if ideal:
                    terms_text += f"• Ideal: {ideal}\n"
                if fallback:
                    terms_text += f"• Fallback Position: {fallback}\n"
                terms_text += "\n"
            
            # Delivery terms
            delivery_terms = ideal_terms.delivery
            if delivery_terms:
                terms_text += "**DELIVERY:**\n"
                ideal = delivery_terms.ideal
                fallback = delivery_terms.fallback_position
                
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

    def build_seller_task_description(self, buyer_message, mediation_proposal=None):
        """Builds the seller's task, including any mediator proposals."""
        
        # Get the seller playbook from the main data model with null checks
        playbook = self.result_to_agent.result.product_details.seller_playbook if (
            self.result_to_agent.result and 
            self.result_to_agent.result.product_details
        ) else None

        # Extract product type from playbook with null checks
        product_type = self.result_to_agent.result.product_type if self.result_to_agent.result else "product"
        
        # Extract buyer profile from playbook with null checks
        buyer_profile = self.result_to_agent.result.buyer_profile if self.result_to_agent.result else None
        credit_worthiness = buyer_profile.credit_worthiness if buyer_profile else "Unknown"
        recurring_customer = buyer_profile.recurring_customer if buyer_profile else False
        
        # Format buyer profile description
        buyer_profile_desc = f"Credit Worthiness: {credit_worthiness}, Recurring Customer: {'Yes' if recurring_customer else 'No'}"

        # Extract seller playbook from the nested structure with null checks
        seller_playbook = playbook
        
        # Extract key sections from the seller playbook with null checks
        criteria = seller_playbook.criteria if seller_playbook else None
        negotiation_rules = seller_playbook.negotiation_rules if seller_playbook else None
        tradables = seller_playbook.tradables if seller_playbook else None
        
        # Format Criteria section
        criteria_text = ""
        if criteria:
            criteria_text = "**PRICING CRITERIA:**\n"
            
            # Product pricing
            product_criteria = criteria.product
            if product_criteria:
                walk_away_price = product_criteria.walk_away_price_usd
                target_price = product_criteria.target_price_usd
                starting_price = product_criteria.starting_price
                
                if walk_away_price:
                    criteria_text += f"• Walk-Away-Price: ${walk_away_price:,}\n"
                if target_price:
                    criteria_text += f"• Target Price: ${target_price:,}\n"
                if starting_price:
                    criteria_text += f"• Starting Price: ${starting_price:,}\n"
                criteria_text += "\n"
            
            # Buyer risk profile definitions
            buyer_criteria = criteria.buyer
            if buyer_criteria:
                risk_definitions = buyer_criteria.risk_profile_definition
                if risk_definitions:
                    criteria_text += "**BUYER RISK PROFILE DEFINITIONS:**\n"
                    if risk_definitions.high_risk:
                        criteria_text += f"• High Risk: {risk_definitions.high_risk}\n"
                    if risk_definitions.medium_risk:
                        criteria_text += f"• Medium Risk: {risk_definitions.medium_risk}\n"
                    if risk_definitions.low_risk:
                        criteria_text += f"• Low Risk: {risk_definitions.low_risk}\n"
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
            # Use the SellerTradables model attribute for the primary goal
            if tradables.primary_goal:
                tradables_text += f"**PRIMARY GOAL:** {tradables.primary_goal}\n\n"
            
            # What you're willing to GIVE (low cost to seller)
            give_items = tradables.give_low_cost_to_us
            if give_items:
                tradables_text += "**WHAT YOU'RE WILLING TO GIVE (Low-cost to you):**\n"
                for item in give_items:
                    tradables_text += f"• {item}\n"
                tradables_text += "\n"
            
            # What you want to GET (high value to seller)
            get_items = tradables.get_high_value_to_us
            if get_items:
                tradables_text += "**WHAT YOU WANT TO GET (High value to you):**\n"
                for item in get_items:
                    tradables_text += f"• {item}\n"
                tradables_text += "\n"
            
            # Ideal & Acceptable Terms by risk level
            ideal_terms = seller_playbook.ideal_acceptable_terms if seller_playbook else None
            if ideal_terms:
                tradables_text += "**IDEAL & ACCEPTABLE TERMS BY BUYER RISK LEVEL:**\n\n"
                
                # High risk buyer terms
                high_risk_terms = ideal_terms.high_risk_buyer
                if high_risk_terms:
                    tradables_text += "**HIGH RISK BUYER:**\n"
                    
                    payment_terms = high_risk_terms.payment_terms
                    if payment_terms:
                        tradables_text += "• Payment Terms:\n"
                        if payment_terms.ideal:
                            tradables_text += f"  - Ideal: {payment_terms.ideal}\n"
                        if payment_terms.fallback_position:
                            tradables_text += f"  - Fallback: {payment_terms.fallback_position}\n"
                    
                    collateral_terms = high_risk_terms.collateral_for_payment_default
                    if collateral_terms:
                        tradables_text += "• Collateral for Payment Default:\n"
                        if collateral_terms.ideal:
                            tradables_text += f"  - Ideal: {collateral_terms.ideal}\n"
                        if collateral_terms.fallback_position:
                            tradables_text += f"  - Fallback: {collateral_terms.fallback_position}\n"
                    tradables_text += "\n"
                
                # Medium risk buyer terms
                medium_risk_terms = ideal_terms.medium_risk_buyer
                if medium_risk_terms:
                    tradables_text += "**MEDIUM RISK BUYER:**\n"
                    
                    payment_terms = medium_risk_terms.payment_terms
                    if payment_terms:
                        tradables_text += "• Payment Terms:\n"
                        if payment_terms.ideal:
                            tradables_text += f"  - Ideal: {payment_terms.ideal}\n"
                        if payment_terms.fallback_position:
                            tradables_text += f"  - Fallback: {payment_terms.fallback_position}\n"
                    tradables_text += "\n"
                
                # Low risk buyer terms
                low_risk_terms = ideal_terms.low_risk_buyer
                if low_risk_terms:
                    tradables_text += "**LOW RISK BUYER:**\n"
                    # This model can have a simple or complex structure
                    # Simple structure
                    if low_risk_terms.ideal:
                        tradables_text += f"• Ideal: {low_risk_terms.ideal}\n"
                    if low_risk_terms.fallback_position:
                        tradables_text += f"• Fallback Position: {low_risk_terms.fallback_position}\n"
                    
                    # Complex structure
                    if low_risk_terms.payment_terms:
                        tradables_text += "• Payment Terms:\n"
                        if low_risk_terms.payment_terms.ideal:
                            tradables_text += f"  - Ideal: {low_risk_terms.payment_terms.ideal}\n"
                        if low_risk_terms.payment_terms.fallback_position:
                            tradables_text += f"  - Fallback Position: {low_risk_terms.payment_terms.fallback_position}\n"
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

    def adjudicate_round(self, last_buyer_offer: str, last_seller_response: str) -> str:
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
            llm=self.llm_llama4,
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

    def run_final_mediation(self, negotiation_history: list[str]) -> dict:
        """Invokes a Mediator agent to analyze a failed negotiation and propose a final compromise."""
        print("\n" + "="*20 + " MEDIATION STAGE " + "="*20)
        print("⚖️  The main negotiation failed. A Chief Mediator is being called in...")

        # Extract necessary data from result_to_agent
        buyer_playbook = self.result_to_agent.result.product_details.buyer_playbook if self.result_to_agent.result and self.result_to_agent.result.product_details else None
        seller_playbook = self.result_to_agent.result.product_details.seller_playbook if self.result_to_agent.result and self.result_to_agent.result.product_details else None
        product_type = self.result_to_agent.result.product_type if self.result_to_agent.result and self.result_to_agent.result.product_type else "product"

        history_str = "\n".join(negotiation_history)
        with open('src/config_crewai/agents.yaml', 'r') as f:
            agents_config = yaml.safe_load(f)
        mediator_agent = Agent(**agents_config['mediator_agent'], llm=self.llm_llama4, verbose=True)

        # The Mediator's task is very specific and detailed here.
        mediation_task = Task(
            description=f"""
            You are a mediator for a failed negotiation for {product_type}. Your task is to propose a final, acceptable compromise.
            
            **Buyer's Key Goals:**
            - Negotiation Strategy: {buyer_playbook.negotiation_strategy if buyer_playbook.negotiation_strategy else 'Not specified'}
            - Primary Goal: {buyer_playbook.tradables.primary_goal if buyer_playbook.tradables and buyer_playbook.tradables.primary_goal else 'Not specified'}
            - Target Price: {buyer_playbook.ideal_acceptable_terms.price.target_purchase_price_usd if buyer_playbook.ideal_acceptable_terms and buyer_playbook.ideal_acceptable_terms.price and buyer_playbook.ideal_acceptable_terms.price.target_purchase_price_usd else 'Not specified'}
            - Max Budget: {buyer_playbook.ideal_acceptable_terms.price.maximum_budget_usd if buyer_playbook.ideal_acceptable_terms and buyer_playbook.ideal_acceptable_terms.price and buyer_playbook.ideal_acceptable_terms.price.maximum_budget_usd else 'Not specified'}

            **Seller's Key Goals:**
            - Primary Goal: {seller_playbook.tradables.primary_goal if seller_playbook.tradables and seller_playbook.tradables.primary_goal else 'Not specified'}
            - Walk-Away Price: {seller_playbook.criteria.product.walk_away_price_usd if seller_playbook.criteria and seller_playbook.criteria.product and seller_playbook.criteria.product.walk_away_price_usd else 'Not specified'}
            - Target Price: {seller_playbook.criteria.product.target_price_usd if seller_playbook.criteria and seller_playbook.criteria.product and seller_playbook.criteria.product.target_price_usd else 'Not specified'}

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


    def run_negotiation(self, max_rounds=4):
        """
        Runs a negotiation with a neutral Adjudicator checking each round,
        and a final mediation stage if no deal is reached.
        """
        if not self.llm_llama4: 
            return

        # 1. Load configurations
        print("📚 Loading configurations...")
        with open('src/config_crewai/agents.yaml', 'r') as f:
            agents_config = yaml.safe_load(f)
        if not agents_config: 
            return
        print("✅ Configurations loaded.")

        
        # Give the agents the calculator tool
        buyer_agent = Agent(**agents_config['buyer_agent'], llm=self.llm_llama4, verbose=True)
        seller_agent = Agent(**agents_config['seller_agent'], llm=self.llm_llama4, verbose=True)

        # 2. Main Negotiation Phase
        print("\n🎭 Starting Main Negotiation...")
        negotiation_history = []
        deal_reached = False
        
        # Opening Move
        print("\n--- ROUND 1 ---")
        task = Task(description=self.build_buyer_task_description(), agent=buyer_agent, expected_output="A JSON object with your offer and justification.")
        last_message = Crew(agents=[buyer_agent], tasks=[task]).kickoff()
        negotiation_history.append(f"BUYER: {last_message}")
        print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20); print(last_message)
    
        # Main Loop with Adjudicator
        for round_number in range(1, max_rounds + 1):
            print(f"\n--- ROUND {round_number + 1} ---")
            
            last_buyer_message = last_message # Store the buyer's message for adjudication

            # Seller's Turn
            task = Task(description=self.build_seller_task_description(last_buyer_message), agent=seller_agent, expected_output="A JSON object with your counter-offer and justification.")
            seller_response = Crew(agents=[seller_agent], tasks=[task]).kickoff()
            negotiation_history.append(f"SELLER: {seller_response}")
            print("\n" + "="*20 + " SELLER'S RESPONSE " + "="*20); print(seller_response)
            
            # Adjudicator's Turn to Check for a Deal
            verdict = self.adjudicate_round(last_buyer_message, seller_response)
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
            task = Task(description=self.build_buyer_task_description(seller_response), agent=buyer_agent, expected_output="A JSON object with your offer and justification.")
            last_message = Crew(agents=[buyer_agent], tasks=[task]).kickoff()
            negotiation_history.append(f"BUYER: {last_message}")
            print("\n" + "="*20 + " BUYER'S RESPONSE " + "="*20); print(last_message)

        # 3. Mediation Phase (if no deal was reached in the main loop)
        if not deal_reached:
            print("\n🏁 Main negotiation concluded without a deal.")
            mediation_result = self.run_final_mediation(negotiation_history)
            
            if mediation_result and mediation_result.get("decision") == "PROPOSE_COMPROMISE":
                # This part of your code for the final mediation round is already correct.
                proposal_data = mediation_result.get("proposal_text") or mediation_result.get("proposal")
                formatted_proposal = json.dumps(proposal_data, indent=2) if isinstance(proposal_data, dict) else str(proposal_data)
                negotiation_history.append(f"MEDIATOR: {formatted_proposal}")

                print("\n--- FINAL MEDIATION ROUND ---")
                
                final_buyer_task = Task(description=self.build_buyer_task_description("\n\nThe Mediator has made a final proposal. You should really consider accepting it very seriously.\n\n Mediator's offer: \n\n", formatted_proposal), agent=buyer_agent, expected_output="A JSON object with your final offer.")
                buyer_final_word = Crew(agents=[buyer_agent], tasks=[final_buyer_task]).kickoff()
                negotiation_history.append(f"BUYER (Final Word): {buyer_final_word}")
                print("\n" + "="*20 + " BUYER'S FINAL WORD " + "="*20); print(buyer_final_word)

                final_seller_task = Task(description=self.build_seller_task_description("\n\nYou will receive the buyer's final word. You should really consider accepting it very seriously. The buyer has made a final offer based on a mediation proposal, which is considered to be fair for both parties.\n\n" + str(buyer_final_word) + "\n\nMeditator's offer:\n\n", formatted_proposal), agent=seller_agent, expected_output="A JSON object with your final decision.")
                seller_final_word = Crew(agents=[seller_agent], tasks=[final_seller_task]).kickoff()
                negotiation_history.append(f"SELLER (Final Word): {seller_final_word}")
                print("\n" + "="*20 + " SELLER'S FINAL WORD " + "="*20); print(seller_final_word)
                
                # Final adjudication after mediation round
                print("\n--- FINAL ADJUDICATION ---")
                final_verdict = self.adjudicate_round(buyer_final_word, seller_final_word)
                
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
                return final_terms
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
                return failure_details
            except (json.JSONDecodeError, TypeError):
                print(f"\n{last_message}")
        print("=" * 50)

    def start(self, max_rounds=4):
        """Start the negotiation process."""
        return self.run_negotiation(max_rounds)