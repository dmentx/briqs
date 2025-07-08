"""
Direct Knowledge Graph Tools for CrewAI Integration

This module provides CrewAI tools that directly access our knowledge graph
without requiring MCP server setup. This is a simpler approach for MVP.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Add the src directory to path so we can import our knowledge graph
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_graph.graph import create_knowledge_graph, KnowledgeGraph

class GetOfferRecommendationsInput(BaseModel):
    """Input for get_offer_recommendations tool."""
    buyer_id: int = Field(description="The ID of the buyer to get recommendations for")
    budget_max: float = Field(default=999999, description="Maximum budget for recommendations")
    product_type: str = Field(default="", description="Filter by product type (excavator, aluminum_sheet)")
    limit: int = Field(default=5, description="Maximum number of recommendations to return")

class AnalyzeBuyerPreferencesInput(BaseModel):
    """Input for analyze_buyer_preferences tool."""
    buyer_id: int = Field(description="The ID of the buyer to analyze")

class GetSimilarOffersInput(BaseModel):
    """Input for get_similar_offers tool.""" 
    offer_id: str = Field(description="The ID of the reference offer")
    similarity_threshold: float = Field(default=0.5, description="Minimum similarity score (0-1)")
    limit: int = Field(default=5, description="Maximum number of similar offers to return")

class GetBuyerInsightsInput(BaseModel):
    """Input for get_buyer_insights tool."""
    buyer_id: int = Field(description="The ID of the buyer to get insights for")

class KnowledgeGraphToolsMixin:
    """Mixin class that provides knowledge graph functionality to tools."""
    
    _graph_instance: KnowledgeGraph = None
    
    @classmethod
    def get_graph(cls) -> KnowledgeGraph:
        """Get or create the knowledge graph instance."""
        if cls._graph_instance is None:
            cls._graph_instance = create_knowledge_graph()
        return cls._graph_instance

class GetOfferRecommendationsTool(BaseTool, KnowledgeGraphToolsMixin):
    """CrewAI tool for getting personalized offer recommendations."""
    
    name: str = "get_offer_recommendations"
    description: str = "Get personalized offer recommendations for a buyer using knowledge graph analysis"
    args_schema: type[BaseModel] = GetOfferRecommendationsInput
    
    def _run(self, buyer_id: int, budget_max: float = 999999, product_type: str = "", limit: int = 5) -> str:
        """Execute the tool."""
        try:
            graph = self.get_graph()
            
            if buyer_id not in graph.buyers:
                return json.dumps({"error": f"Buyer {buyer_id} not found"})
            
            buyer = graph.buyers[buyer_id]
            recommendations = []
            
            # Get buyer's purchase history for preference analysis
            purchased_offers = graph.get_neighbors(f"buyer_{buyer_id}", "PURCHASED")
            purchased_ids = {offer_id for offer_id, _ in purchased_offers}
            
            # Score all available offers
            for offer_id, offer in graph.offers.items():
                if offer_id in purchased_ids:
                    continue  # Skip already purchased
                
                if product_type and offer.product_type != product_type:
                    continue  # Filter by product type
                    
                if offer.price > budget_max:
                    continue  # Filter by budget
                
                # Calculate recommendation score
                score = self._calculate_recommendation_score(buyer, offer, purchased_offers, graph)
                
                recommendations.append({
                    "offer_id": offer_id,
                    "product_type": offer.product_type,
                    "price": offer.price,
                    "supplier": offer.supplier_name,
                    "score": round(score, 3),
                    "specifications": offer.specifications,
                    "reasoning": f"Score based on buyer preferences ({score:.1%} match)"
                })
            
            # Sort by score and limit results
            recommendations.sort(key=lambda x: x["score"], reverse=True)
            recommendations = recommendations[:limit]
            
            result = {
                "buyer_id": buyer_id,
                "recommendations": recommendations,
                "total_found": len(recommendations),
                "filters_applied": {
                    "budget_max": budget_max if budget_max != 999999 else None,
                    "product_type": product_type if product_type else None
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
    
    def _calculate_recommendation_score(self, buyer, offer, purchased_offers, graph):
        """Calculate recommendation score for an offer."""
        score = 0.0
        
        # Base score from offer success rate
        score += offer.success_rate * 0.3
        
        # Category preference scoring
        if offer.category in buyer.preferred_categories:
            score += 0.4
        
        # Price range compatibility (prefer offers in buyer's typical range)
        if purchased_offers:
            avg_purchase_price = sum(graph.offers[offer_id].price 
                                   for offer_id, _ in purchased_offers 
                                   if offer_id in graph.offers) / len(purchased_offers)
            
            price_similarity = 1.0 - abs(offer.price - avg_purchase_price) / max(offer.price, avg_purchase_price)
            score += price_similarity * 0.2
        
        # Similarity to previous purchases
        similarity_bonus = 0.0
        for purchased_id, weight in purchased_offers:
            if purchased_id in graph.offers:
                purchased_offer = graph.offers[purchased_id]
                if purchased_offer.product_type == offer.product_type:
                    similarity = graph._calculate_offer_similarity(purchased_offer, offer)
                    similarity_bonus = max(similarity_bonus, similarity * weight)
        
        score += similarity_bonus * 0.1
        
        return min(1.0, score)

class AnalyzeBuyerPreferencesTool(BaseTool, KnowledgeGraphToolsMixin):
    """CrewAI tool for analyzing buyer preferences and purchase patterns."""
    
    name: str = "analyze_buyer_preferences"
    description: str = "Analyze buyer preferences and purchase patterns using knowledge graph data"
    args_schema: type[BaseModel] = AnalyzeBuyerPreferencesInput
    
    def _run(self, buyer_id: int) -> str:
        """Execute the tool."""
        try:
            graph = self.get_graph()
            
            if buyer_id not in graph.buyers:
                return json.dumps({"error": f"Buyer {buyer_id} not found"})
            
            buyer = graph.buyers[buyer_id]
            
            # Get purchase history
            purchased_offers = graph.get_neighbors(f"buyer_{buyer_id}", "PURCHASED")
            
            # Analyze preferences
            preferences = graph.get_neighbors(f"buyer_{buyer_id}", "PREFERS")
            
            # Calculate spending patterns
            purchase_data = []
            total_spent = 0
            product_types = {}
            
            for offer_id, weight in purchased_offers:
                if offer_id in graph.offers:
                    offer = graph.offers[offer_id]
                    purchase_data.append({
                        "offer_id": offer_id,
                        "product_type": offer.product_type,
                        "price": offer.price,
                        "purchase_weight": weight
                    })
                    total_spent += offer.price
                    product_types[offer.product_type] = product_types.get(offer.product_type, 0) + 1
            
            result = {
                "buyer_id": buyer_id,
                "buyer_profile": {
                    "credit_score": buyer.credit_score,
                    "recurring_customer": buyer.recurring_customer,
                    "purchase_count": buyer.purchase_count,
                    "total_spent": total_spent,
                    "preferred_categories": list(buyer.preferred_categories)
                },
                "purchase_patterns": {
                    "total_purchases": len(purchase_data),
                    "avg_purchase_price": total_spent / len(purchase_data) if purchase_data else 0,
                    "product_type_distribution": product_types,
                    "recent_purchases": purchase_data[:5]  # Last 5 purchases
                },
                "preferences": [{"category": pref_id.replace("category_", ""), "strength": weight} 
                              for pref_id, weight in preferences]
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

class GetSimilarOffersTool(BaseTool, KnowledgeGraphToolsMixin):
    """CrewAI tool for finding offers similar to a given offer."""
    
    name: str = "get_similar_offers"
    description: str = "Find offers similar to a given offer using multi-dimensional similarity analysis"
    args_schema: type[BaseModel] = GetSimilarOffersInput
    
    def _run(self, offer_id: str, similarity_threshold: float = 0.5, limit: int = 5) -> str:
        """Execute the tool."""
        try:
            graph = self.get_graph()
            
            if offer_id not in graph.offers:
                return json.dumps({"error": f"Offer {offer_id} not found"})
            
            # Get similar offers from graph
            similar_offers = graph.get_neighbors(offer_id, "SIMILAR_TO")
            
            # Filter by threshold and format results
            results = []
            for similar_id, similarity in similar_offers:
                if similarity >= similarity_threshold and similar_id in graph.offers:
                    offer = graph.offers[similar_id]
                    results.append({
                        "offer_id": similar_id,
                        "similarity_score": round(similarity, 3),
                        "product_type": offer.product_type,
                        "price": offer.price,
                        "supplier": offer.supplier_name,
                        "specifications": offer.specifications
                    })
            
            # Sort by similarity and limit
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results = results[:limit]
            
            base_offer = graph.offers[offer_id]
            
            result = {
                "base_offer": {
                    "offer_id": offer_id,
                    "product_type": base_offer.product_type,
                    "price": base_offer.price,
                    "supplier": base_offer.supplier_name
                },
                "similar_offers": results,
                "similarity_threshold": similarity_threshold,
                "total_found": len(results)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

class GetBuyerInsightsTool(BaseTool, KnowledgeGraphToolsMixin):
    """CrewAI tool for getting comprehensive buyer insights."""
    
    name: str = "get_buyer_insights"
    description: str = "Get comprehensive insights about a buyer including market position and risk profile"
    args_schema: type[BaseModel] = GetBuyerInsightsInput
    
    def _run(self, buyer_id: int) -> str:
        """Execute the tool."""
        try:
            graph = self.get_graph()
            
            if buyer_id not in graph.buyers:
                return json.dumps({"error": f"Buyer {buyer_id} not found"})
            
            # Get detailed analysis using the preferences tool
            preferences_tool = AnalyzeBuyerPreferencesTool()
            preferences_result = json.loads(preferences_tool._run(buyer_id))
            
            # Add market insights
            buyer = graph.buyers[buyer_id]
            
            # Compare with other buyers
            all_buyers = list(graph.buyers.values())
            avg_credit_score = sum(b.credit_score for b in all_buyers) / len(all_buyers)
            avg_purchase_count = sum(b.purchase_count for b in all_buyers) / len(all_buyers)
            
            market_position = {
                "credit_score_percentile": len([b for b in all_buyers if b.credit_score <= buyer.credit_score]) / len(all_buyers),
                "purchase_activity_percentile": len([b for b in all_buyers if b.purchase_count <= buyer.purchase_count]) / len(all_buyers),
                "above_avg_credit": buyer.credit_score > avg_credit_score,
                "above_avg_activity": buyer.purchase_count > avg_purchase_count
            }
            
            result = {
                **preferences_result,
                "market_insights": market_position,
                "risk_profile": {
                    "credit_tier": "high" if buyer.credit_score >= 8.5 else "medium" if buyer.credit_score >= 7.0 else "low",
                    "customer_stability": "high" if buyer.recurring_customer else "medium",
                    "purchasing_power": "high" if buyer.total_spent > 500000 else "medium" if buyer.total_spent > 200000 else "low"
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

# Export all tools for easy import
__all__ = [
    "GetOfferRecommendationsTool",
    "AnalyzeBuyerPreferencesTool", 
    "GetSimilarOffersTool",
    "GetBuyerInsightsTool"
] 