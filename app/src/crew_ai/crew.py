"""
Multi-Agent Negotiation System - CrewAI Integration
Four-agent orchestrated negotiation system with buyer, seller, orchestrator, and mediator agents.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class NegotiationCrew:
    """
    Multi-Agent Negotiation System using CrewAI framework.
    
    This crew manages a four-agent negotiation system:
    - Buyer Agent: Represents purchasing interests
    - Seller Agent: Represents selling interests  
    - Orchestrator Agent: Manages negotiation flow and coordination
    - Mediator Agent: Provides neutral conflict resolution
    """
    
    agents_config = '../config_crewai/agents.yaml'
    tasks_config = '../config_crewai/tasks.yaml'
    
    def __init__(self, **kwargs):
        """Initialize the negotiation crew with configuration."""
        super().__init__(**kwargs)
        self.negotiation_context = {}
        self.negotiation_history = []
        
        # Load configuration files
        self._load_configs()
    
    def _load_configs(self):
        """Load YAML configuration files."""
        import yaml
        from pathlib import Path
        
        # Get the directory of this file
        current_dir = Path(__file__).parent
        
        # Load agents config
        agents_config_path = current_dir / self.agents_config
        with open(agents_config_path, 'r') as f:
            self.agents_config_data = yaml.safe_load(f)
        
        # Load tasks config
        tasks_config_path = current_dir / self.tasks_config
        with open(tasks_config_path, 'r') as f:
            self.tasks_config_data = yaml.safe_load(f)
        
    @agent
    def buyer_agent(self) -> Agent:
        """Create the buyer negotiation agent."""
        config = self.agents_config_data['buyer_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=config.get('verbose', True),
            memory=config.get('memory', True),
            allow_delegation=config.get('allow_delegation', False),
            max_iter=config.get('max_iter', 3)
        )
    
    @agent
    def seller_agent(self) -> Agent:
        """Create the seller negotiation agent."""
        config = self.agents_config_data['seller_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=config.get('verbose', True),
            memory=config.get('memory', True),
            allow_delegation=config.get('allow_delegation', False),
            max_iter=config.get('max_iter', 3)
        )
    
    @agent
    def orchestrator_agent(self) -> Agent:
        """Create the orchestrator process management agent."""
        config = self.agents_config_data['orchestrator_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=config.get('verbose', True),
            memory=config.get('memory', True),
            allow_delegation=config.get('allow_delegation', True),
            max_iter=config.get('max_iter', 5)
        )
    
    @agent
    def mediator_agent(self) -> Agent:
        """Create the mediator conflict resolution agent."""
        config = self.agents_config_data['mediator_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=config.get('verbose', True),
            memory=config.get('memory', True),
            allow_delegation=config.get('allow_delegation', False),
            max_iter=config.get('max_iter', 2)
        )
    
    @task
    def orchestration_task(self) -> Task:
        """Create the orchestration task for managing negotiation flow."""
        config = self.tasks_config_data['orchestration_task']
        return Task(
            description=config['description'],
            expected_output=config['expected_output'],
            agent=self.orchestrator_agent(),
            output_file=config.get('output_file')
        )
    
    @task
    def buyer_negotiation_task(self) -> Task:
        """Create the buyer negotiation task."""
        config = self.tasks_config_data['buyer_negotiation_task']
        return Task(
            description=config['description'],
            expected_output=config['expected_output'],
            agent=self.buyer_agent(),
            output_file=config.get('output_file')
        )
    
    @task
    def seller_negotiation_task(self) -> Task:
        """Create the seller negotiation task."""
        config = self.tasks_config_data['seller_negotiation_task']
        return Task(
            description=config['description'],
            expected_output=config['expected_output'],
            agent=self.seller_agent(),
            output_file=config.get('output_file')
        )
    
    @task
    def mediation_task(self) -> Task:
        """Create the mediation task for conflict resolution."""
        config = self.tasks_config_data['mediation_task']
        return Task(
            description=config['description'],
            expected_output=config['expected_output'],
            agent=self.mediator_agent(),
            output_file=config.get('output_file')
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the negotiation crew with all agents and tasks."""
        return Crew(
            agents=[
                self.orchestrator_agent(),
                self.buyer_agent(),
                self.seller_agent(),
                self.mediator_agent()
            ],
            tasks=[
                self.orchestration_task(),
                self.buyer_negotiation_task(),
                self.seller_negotiation_task(),
                self.mediation_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            planning=True,
            embedder={
                "provider": "groq",
                "config": {
                    "model": "llama3-8b-8192",
                    "temperature": 0.7
                }
            }
        )
    
    def set_negotiation_context(self, context: Dict[str, Any]) -> None:
        """
        Set the negotiation context parameters.
        
        Args:
            context: Dictionary containing negotiation parameters like:
                - contract_type: Type of contract being negotiated
                - budget_range: Budget constraints for buyer
                - price_range: Price range for seller
                - key_terms: Important contract terms
                - timeline: Negotiation timeline
                - requirements: Specific requirements
        """
        self.negotiation_context = context
        
    def get_negotiation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the negotiation results.
        
        Returns:
            Dictionary containing negotiation outcomes and analysis
        """
        return {
            "context": self.negotiation_context,
            "history": self.negotiation_history,
            "agents_involved": [
                "orchestrator_agent",
                "buyer_agent", 
                "seller_agent",
                "mediator_agent"
            ],
            "process_type": "sequential_with_mediation"
        }
    
    def run_negotiation(self, 
                       contract_type: str,
                       buyer_budget: float,
                       seller_price: float,
                       requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a complete negotiation process.
        
        Args:
            contract_type: Type of contract (e.g., 'software_license', 'consulting', 'product_sale')
            buyer_budget: Maximum budget for buyer
            seller_price: Initial asking price from seller
            requirements: Additional negotiation requirements
            
        Returns:
            Dictionary containing negotiation results
        """
        # Set negotiation context
        context = {
            "contract_type": contract_type,
            "buyer_budget": buyer_budget,
            "seller_price": seller_price,
            "requirements": requirements or {},
            "negotiation_id": f"negotiation_{len(self.negotiation_history) + 1}"
        }
        
        self.set_negotiation_context(context)
        
        # Execute the crew
        try:
            result = self.crew().kickoff(inputs=context)
            
            # Store negotiation history
            self.negotiation_history.append({
                "context": context,
                "result": result,
                "timestamp": None  # Will be set by CrewAI
            })
            
            return {
                "success": True,
                "result": result,
                "context": context,
                "summary": self.get_negotiation_summary()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "context": context
            }


def create_negotiation_crew(**kwargs) -> NegotiationCrew:
    """
    Factory function to create a negotiation crew instance.
    
    Args:
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured NegotiationCrew instance
    """
    return NegotiationCrew(**kwargs)


def run_sample_negotiation() -> Dict[str, Any]:
    """
    Run a sample negotiation for testing purposes.
    
    Returns:
        Dictionary containing sample negotiation results
    """
    crew = create_negotiation_crew()
    
    return crew.run_negotiation(
        contract_type="software_license",
        buyer_budget=10000.0,
        seller_price=12000.0,
        requirements={
            "license_duration": "2 years",
            "support_level": "premium",
            "user_count": 100,
            "payment_terms": "quarterly"
        }
    )


if __name__ == "__main__":
    # Run sample negotiation for testing
    print("Running sample negotiation...")
    result = run_sample_negotiation()
    print(f"Negotiation result: {result}")
