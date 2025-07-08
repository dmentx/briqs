"""
Enhanced B2B Negotiation Agent Crew with Knowledge Graph Integration

This module creates a multi-agent crew that handles B2B negotiations with support for
knowledge graph-powered recommendations and insights.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from crewai import Agent, Crew, Task, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our direct knowledge graph tools
from crew_ai.knowledge_graph_tools import (
    GetOfferRecommendationsTool,
    AnalyzeBuyerPreferencesTool,
    GetSimilarOffersTool,
    GetBuyerInsightsTool
)

class BriqsNegotiationCrew:
    """
    Enhanced negotiation crew with knowledge graph recommendations
    """
    
    def __init__(self):
        """Initialize the crew with all agents and tools"""
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        # Initialize knowledge graph tools
        self.kg_tools = [
            GetOfferRecommendationsTool(),
            AnalyzeBuyerPreferencesTool(),
            GetSimilarOffersTool(),
            GetBuyerInsightsTool()
        ]
        
        # Create agents
        self.mediator_agent = self._create_mediator_agent()
        self.market_analyst = self._create_market_analyst()
        self.contract_specialist = self._create_contract_specialist()
        
        # Create crew
        self.crew = Crew(
            agents=[self.mediator_agent, self.market_analyst, self.contract_specialist],
            process=Process.sequential,
            verbose=True
        )
        
    def _create_mediator_agent(self) -> Agent:
        """Create the main mediator agent with knowledge graph tools"""
        return Agent(
            role="B2B Negotiation Mediator",
            goal="""
            Facilitate successful B2B negotiations between buyers and suppliers using AI-powered 
            insights from our knowledge graph. Provide personalized recommendations, analyze market 
            dynamics, and guide both parties toward mutually beneficial agreements.
            """,
            backstory="""
            You are an experienced B2B negotiation specialist with access to advanced knowledge graph 
            analytics. You understand buyer preferences, market trends, and can identify similar 
            successful deals. Your mission is to create win-win scenarios by leveraging data-driven 
            insights and relationship history.
            """,
            tools=self.kg_tools,  # Direct knowledge graph tools
            llm=self.llm,
            verbose=True,
            max_iter=3,
            memory=True
        )
    
    def _create_market_analyst(self) -> Agent:
        """Create market analysis agent"""
        return Agent(
            role="Market Intelligence Analyst", 
            goal="""
            Provide comprehensive market analysis including price trends, supplier reliability,
            and competitive intelligence to support negotiation strategies.
            """,
            backstory="""
            You are a data-driven market analyst specializing in B2B industrial markets. You excel
            at identifying market opportunities, analyzing supplier performance, and providing
            strategic pricing recommendations.
            """,
            tools=[],  # Market-specific tools can be added later
            llm=self.llm,
            verbose=True
        )
        
    def _create_contract_specialist(self) -> Agent:
        """Create contract and legal specialist agent"""
        return Agent(
            role="Contract Negotiation Specialist",
            goal="""
            Draft, review, and optimize contract terms to ensure legal compliance and favorable
            conditions for all parties involved in the negotiation.
            """,
            backstory="""
            You are an expert in B2B contract law with extensive experience in industrial supply
            agreements. You focus on creating clear, enforceable contracts that protect all parties
            while enabling smooth business operations.
            """,
            tools=[],  # Contract-specific tools can be added later
            llm=self.llm,
            verbose=True
        )

    def analyze_buyer_and_recommend(self, buyer_id: int, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive buyer analysis and recommendation generation
        
        Args:
            buyer_id: ID of the buyer to analyze
            requirements: Dict containing buyer requirements like budget, product_type, etc.
            
        Returns:
            Dict containing analysis results and recommendations
        """
        
        # Create task for comprehensive buyer analysis
        analysis_task = Task(
            description=f"""
            Perform comprehensive analysis for buyer {buyer_id} and generate personalized recommendations.
            
            Requirements: {requirements}
            
            Your analysis should include:
            1. Buyer profile and purchase history analysis using analyze_buyer_preferences
            2. Personalized offer recommendations using get_offer_recommendations  
            3. Market insights and risk assessment using get_buyer_insights
            4. Strategic negotiation recommendations based on the data
            
            Use the knowledge graph tools to gather data, then provide strategic insights and
            actionable recommendations for the negotiation.
            """,
            agent=self.mediator_agent,
            expected_output="""
            A comprehensive analysis report containing:
            - Buyer profile summary with key metrics
            - Top 5 personalized offer recommendations with reasoning
            - Market position and risk assessment
            - Strategic negotiation approach recommendations
            - Key talking points and value propositions
            """
        )
        
        # Execute the task
        result = self.crew.kickoff(tasks=[analysis_task])
        
        return {
            "analysis_result": result,
            "buyer_id": buyer_id,
            "requirements": requirements
        }
    
    def find_similar_deals(self, offer_id: str, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Find and analyze similar deals for negotiation insights
        
        Args:
            offer_id: Reference offer ID
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Dict containing similar deals analysis
        """
        
        similar_deals_task = Task(
            description=f"""
            Find and analyze deals similar to offer {offer_id} to provide negotiation insights.
            
            Use get_similar_offers with threshold {similarity_threshold} to find comparable deals.
            
            Analyze the results to provide:
            1. Market benchmarking data
            2. Pricing strategies used in similar deals
            3. Success factors from comparable negotiations
            4. Recommended negotiation tactics based on historical data
            """,
            agent=self.market_analyst,
            expected_output="""
            Similar deals analysis report containing:
            - List of similar offers with similarity scores
            - Price comparison and market positioning
            - Success patterns identified from historical deals
            - Recommended negotiation strategies
            """
        )
        
        result = self.crew.kickoff(tasks=[similar_deals_task])
        
        return {
            "similar_deals_analysis": result,
            "reference_offer": offer_id,
            "threshold": similarity_threshold
        }

    def negotiate_deal(self, buyer_id: int, offer_id: str, negotiation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Facilitate a complete negotiation process using knowledge graph insights
        
        Args:
            buyer_id: ID of the buyer
            offer_id: ID of the offer being negotiated
            negotiation_context: Context and parameters for the negotiation
            
        Returns:
            Dict containing negotiation results and recommendations
        """
        
        # Task 1: Buyer and offer analysis
        analysis_task = Task(
            description=f"""
            Analyze buyer {buyer_id} and offer {offer_id} for negotiation preparation.
            
            Use analyze_buyer_preferences and get_buyer_insights to understand the buyer.
            Find similar offers using get_similar_offers to benchmark the current offer.
            
            Context: {negotiation_context}
            """,
            agent=self.mediator_agent,
            expected_output="Comprehensive buyer and offer analysis with negotiation context"
        )
        
        # Task 2: Market and pricing analysis
        market_task = Task(
            description=f"""
            Based on the analysis from Task 1, provide market intelligence and pricing strategy.
            
            Analyze market position, competitive landscape, and pricing recommendations
            for the negotiation between buyer {buyer_id} and offer {offer_id}.
            """,
            agent=self.market_analyst,
            expected_output="Market analysis and pricing strategy recommendations"
        )
        
        # Task 3: Contract and terms optimization
        contract_task = Task(
            description=f"""
            Based on insights from Tasks 1 and 2, recommend optimal contract terms and conditions.
            
            Focus on creating a win-win scenario that addresses both buyer needs and supplier constraints
            for the negotiation of offer {offer_id}.
            """,
            agent=self.contract_specialist,
            expected_output="Optimized contract terms and negotiation strategy"
        )
        
        # Execute all tasks
        result = self.crew.kickoff(tasks=[analysis_task, market_task, contract_task])
        
        return {
            "negotiation_result": result,
            "buyer_id": buyer_id,
            "offer_id": offer_id,
            "context": negotiation_context
        }

def create_enhanced_crew() -> BriqsNegotiationCrew:
    """Factory function to create the enhanced negotiation crew"""
    return BriqsNegotiationCrew()

# Test function for the enhanced crew
def test_enhanced_crew():
    """Test the enhanced crew with knowledge graph integration"""
    
    print("🚀 Testing Enhanced CrewAI + Knowledge Graph Integration")
    print("=" * 60)
    
    try:
        # Create the enhanced crew
        crew = create_enhanced_crew()
        print("✅ Enhanced crew created successfully!")
        
        # Test 1: Buyer analysis and recommendations
        print("\n📊 Test 1: Buyer Analysis and Recommendations")
        print("-" * 50)
        
        requirements = {
            "budget_max": 500000,
            "product_type": "excavator",
            "urgency": "medium"
        }
        
        analysis_result = crew.analyze_buyer_and_recommend(
            buyer_id=1,
            requirements=requirements
        )
        
        print("✅ Buyer analysis completed!")
        print(f"Buyer ID: {analysis_result['buyer_id']}")
        print(f"Requirements: {analysis_result['requirements']}")
        
        # Test 2: Similar deals analysis
        print("\n🔍 Test 2: Similar Deals Analysis")
        print("-" * 50)
        
        similar_deals = crew.find_similar_deals(
            offer_id="EXC-001",
            similarity_threshold=0.6
        )
        
        print("✅ Similar deals analysis completed!")
        print(f"Reference offer: {similar_deals['reference_offer']}")
        
        print("\n🎯 All tests completed successfully!")
        print("Enhanced CrewAI with Knowledge Graph integration is working!")
        
        return crew
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_enhanced_crew()