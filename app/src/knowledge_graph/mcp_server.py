"""
Knowledge Graph Offer Recommendations MCP Server

This MCP server exposes our knowledge graph-based recommendation system 
through standard MCP tools, enabling external AI systems to query for 
personalized B2B offer recommendations.

Features:
- Offer recommendation tools with similarity scoring
- Buyer preference analysis and modeling  
- Graph relationship exploration
- Real-time similarity calculations
- Access to offer and buyer data
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Standard MCP imports
import mcp.server.stdio
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, EmbeddedResource

# Our knowledge graph
from .graph import create_knowledge_graph, KnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphMCPServer:
    """MCP Server for Knowledge Graph Offer Recommendations"""
    
    def __init__(self):
        self.server = Server("knowledge-graph-recommendations")
        self.graph: Optional[KnowledgeGraph] = None
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="get_offer_recommendations",
                    description="Get personalized offer recommendations for a buyer",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "buyer_id": {"type": "integer", "description": "Buyer ID"},
                            "budget_max": {"type": "number", "description": "Maximum budget"},
                            "product_type": {"type": "string", "description": "Product type filter"},
                            "limit": {"type": "integer", "description": "Number of recommendations", "default": 5}
                        },
                        "required": ["buyer_id"]
                    }
                ),
                Tool(
                    name="analyze_buyer_preferences",
                    description="Analyze buyer preferences and purchase patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "buyer_id": {"type": "integer", "description": "Buyer ID"}
                        },
                        "required": ["buyer_id"]
                    }
                ),
                Tool(
                    name="get_similar_offers",
                    description="Find offers similar to a given offer",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "offer_id": {"type": "string", "description": "Base offer ID"},
                            "similarity_threshold": {"type": "number", "description": "Minimum similarity (0-1)", "default": 0.5},
                            "limit": {"type": "integer", "description": "Number of results", "default": 5}
                        },
                        "required": ["offer_id"]
                    }
                ),
                Tool(
                    name="get_buyer_insights",
                    description="Get comprehensive insights about a buyer",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "buyer_id": {"type": "integer", "description": "Buyer ID"}
                        },
                        "required": ["buyer_id"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
            """Handle tool calls"""
            try:
                # Initialize graph if needed
                if self.graph is None:
                    self.graph = create_knowledge_graph()
                
                # Route to appropriate handler
                if name == "get_offer_recommendations":
                    result = await self._get_offer_recommendations(arguments or {})
                elif name == "analyze_buyer_preferences":
                    result = await self._analyze_buyer_preferences(arguments or {})
                elif name == "get_similar_offers":
                    result = await self._get_similar_offers(arguments or {})
                elif name == "get_buyer_insights":
                    result = await self._get_buyer_insights(arguments or {})
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                logger.error(f"Tool call error: {e}")
                raise RuntimeError(f"Tool execution failed: {str(e)}")
    
    async def _get_offer_recommendations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized offer recommendations for a buyer"""
        buyer_id = args.get("buyer_id")
        budget_max = args.get("budget_max", float('inf'))
        product_type = args.get("product_type")
        limit = args.get("limit", 5)
        
        if buyer_id not in self.graph.buyers:
            return {"error": f"Buyer {buyer_id} not found"}
        
        buyer = self.graph.buyers[buyer_id]
        recommendations = []
        
        # Get buyer's purchase history for preference analysis
        purchased_offers = self.graph.get_neighbors(f"buyer_{buyer_id}", "PURCHASED")
        purchased_ids = {offer_id for offer_id, _ in purchased_offers}
        
        # Score all available offers
        for offer_id, offer in self.graph.offers.items():
            if offer_id in purchased_ids:
                continue  # Skip already purchased
            
            if product_type and offer.product_type != product_type:
                continue  # Filter by product type
                
            if offer.price > budget_max:
                continue  # Filter by budget
            
            # Calculate recommendation score
            score = await self._calculate_recommendation_score(buyer, offer, purchased_offers)
            
            recommendations.append({
                "offer_id": offer_id,
                "product_type": offer.product_type,
                "price": offer.price,
                "supplier": offer.supplier_name,
                "score": score,
                "specifications": offer.specifications,
                "contract_terms": offer.contract_terms[:100] + "..." if len(offer.contract_terms) > 100 else offer.contract_terms
            })
        
        # Sort by score and limit results
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        recommendations = recommendations[:limit]
        
        return {
            "buyer_id": buyer_id,
            "recommendations": recommendations,
            "total_found": len(recommendations),
            "filters_applied": {
                "budget_max": budget_max if budget_max != float('inf') else None,
                "product_type": product_type
            }
        }
    
    async def _calculate_recommendation_score(self, buyer, offer, purchased_offers):
        """Calculate recommendation score for an offer"""
        score = 0.0
        
        # Base score from offer success rate
        score += offer.success_rate * 0.3
        
        # Category preference scoring
        if offer.category in buyer.preferred_categories:
            score += 0.4
        
        # Price range compatibility (prefer offers in buyer's typical range)
        if purchased_offers:
            avg_purchase_price = sum(self.graph.offers[offer_id].price 
                                   for offer_id, _ in purchased_offers 
                                   if offer_id in self.graph.offers) / len(purchased_offers)
            
            price_similarity = 1.0 - abs(offer.price - avg_purchase_price) / max(offer.price, avg_purchase_price)
            score += price_similarity * 0.2
        
        # Similarity to previous purchases
        similarity_bonus = 0.0
        for purchased_id, weight in purchased_offers:
            if purchased_id in self.graph.offers:
                purchased_offer = self.graph.offers[purchased_id]
                if purchased_offer.product_type == offer.product_type:
                    similarity = self.graph._calculate_offer_similarity(purchased_offer, offer)
                    similarity_bonus = max(similarity_bonus, similarity * weight)
        
        score += similarity_bonus * 0.1
        
        return min(1.0, score)
    
    async def _analyze_buyer_preferences(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze buyer preferences and purchase patterns"""
        buyer_id = args.get("buyer_id")
        
        if buyer_id not in self.graph.buyers:
            return {"error": f"Buyer {buyer_id} not found"}
        
        buyer = self.graph.buyers[buyer_id]
        
        # Get purchase history
        purchased_offers = self.graph.get_neighbors(f"buyer_{buyer_id}", "PURCHASED")
        
        # Analyze preferences
        preferences = self.graph.get_neighbors(f"buyer_{buyer_id}", "PREFERS")
        
        # Calculate spending patterns
        purchase_data = []
        total_spent = 0
        product_types = {}
        
        for offer_id, weight in purchased_offers:
            if offer_id in self.graph.offers:
                offer = self.graph.offers[offer_id]
                purchase_data.append({
                    "offer_id": offer_id,
                    "product_type": offer.product_type,
                    "price": offer.price,
                    "purchase_weight": weight
                })
                total_spent += offer.price
                product_types[offer.product_type] = product_types.get(offer.product_type, 0) + 1
        
        return {
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
    
    async def _get_similar_offers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find offers similar to a given offer"""
        offer_id = args.get("offer_id")
        similarity_threshold = args.get("similarity_threshold", 0.5)
        limit = args.get("limit", 5)
        
        if offer_id not in self.graph.offers:
            return {"error": f"Offer {offer_id} not found"}
        
        # Get similar offers from graph
        similar_offers = self.graph.get_neighbors(offer_id, "SIMILAR_TO")
        
        # Filter by threshold and format results
        results = []
        for similar_id, similarity in similar_offers:
            if similarity >= similarity_threshold and similar_id in self.graph.offers:
                offer = self.graph.offers[similar_id]
                results.append({
                    "offer_id": similar_id,
                    "similarity_score": similarity,
                    "product_type": offer.product_type,
                    "price": offer.price,
                    "supplier": offer.supplier_name,
                    "specifications": offer.specifications
                })
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:limit]
        
        base_offer = self.graph.offers[offer_id]
        
        return {
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
    
    async def _get_buyer_insights(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive insights about a buyer"""
        buyer_id = args.get("buyer_id")
        
        if buyer_id not in self.graph.buyers:
            return {"error": f"Buyer {buyer_id} not found"}
        
        # Get detailed analysis
        preferences_result = await self._analyze_buyer_preferences(args)
        
        # Add market insights
        buyer = self.graph.buyers[buyer_id]
        
        # Compare with other buyers
        all_buyers = list(self.graph.buyers.values())
        avg_credit_score = sum(b.credit_score for b in all_buyers) / len(all_buyers)
        avg_purchase_count = sum(b.purchase_count for b in all_buyers) / len(all_buyers)
        
        market_position = {
            "credit_score_percentile": len([b for b in all_buyers if b.credit_score <= buyer.credit_score]) / len(all_buyers),
            "purchase_activity_percentile": len([b for b in all_buyers if b.purchase_count <= buyer.purchase_count]) / len(all_buyers),
            "above_avg_credit": buyer.credit_score > avg_credit_score,
            "above_avg_activity": buyer.purchase_count > avg_purchase_count
        }
        
        return {
            **preferences_result,
            "market_insights": market_position,
            "risk_profile": {
                "credit_tier": "high" if buyer.credit_score >= 8.5 else "medium" if buyer.credit_score >= 7.0 else "low",
                "customer_stability": "high" if buyer.recurring_customer else "medium",
                "purchasing_power": "high" if buyer.total_spent > 500000 else "medium" if buyer.total_spent > 200000 else "low"
            }
        }

    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="knowledge-graph-recommendations",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

# Main entry point
async def main():
    """Main entry point for the MCP server"""
    server = KnowledgeGraphMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 