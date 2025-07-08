"""
Knowledge Graph for B2B Offer Recommendations

This module implements a graph-based recommendation system that models:
- Buyers and their purchasing patterns
- Offers and their specifications
- Product categories and relationships
- Supplier information and ratings

Graph Structure: Buyer → Purchase History → Product Preferences
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BuyerNode:
    """Represents a buyer in the knowledge graph"""
    buyer_id: int
    credit_score: float = 8.0
    recurring_customer: bool = True
    activity_score: float = 1.0
    purchase_count: int = 0
    total_spent: float = 0.0
    preferred_categories: Set[str] = None
    
    def __post_init__(self):
        if self.preferred_categories is None:
            self.preferred_categories = set()


@dataclass
class OfferNode:
    """Represents an offer/deal in the knowledge graph"""
    offer_id: str
    deal_id: int
    product_type: str  # "excavator" | "aluminum_sheet"
    price: float
    specifications: Dict[str, Any]
    contract_terms: str
    supplier_name: str
    success_rate: float = 1.0
    category: str = ""
    
    def get_spec_vector(self) -> Dict[str, float]:
        """Convert specifications to numerical vector for similarity calculation"""
        vector = {}
        
        if self.product_type == "excavator":
            vector.update({
                'lifting_capacity': self.specifications.get('lifting_capacity_tons', 0),
                'operating_weight': self.specifications.get('operating_weight_tons', 0),
                'digging_depth': self.specifications.get('max_digging_depth_m', 0),
                'bucket_capacity': self.specifications.get('bucket_capacity_m3', 0),
                'price_range': self.price / 50000,  # Normalize to 0-10 range
                'year': self.specifications.get('year', 2020)
            })
        elif self.product_type == "aluminum_sheet":
            vector.update({
                'thickness': self.specifications.get('thickness_mm', 0),
                'weight': self.specifications.get('total_weight_kg', 0),
                'price_per_unit': self.price,
                'availability': self.specifications.get('availability', 0)
            })
        
        return vector


@dataclass 
class ProductNode:
    """Represents a product category in the knowledge graph"""
    product_id: str
    product_type: str
    category: str
    avg_price: float = 0.0
    common_specs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.common_specs is None:
            self.common_specs = {}


@dataclass
class SupplierNode:
    """Represents a supplier in the knowledge graph"""
    supplier_id: str
    name: str
    rating: float = 4.5
    reliability_score: float = 0.8
    product_categories: Set[str] = None
    
    def __post_init__(self):
        if self.product_categories is None:
            self.product_categories = set()


@dataclass
class GraphEdge:
    """Represents a relationship between nodes in the graph"""
    from_node: str
    to_node: str
    relationship_type: str
    weight: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RelationshipTypes:
    """Constants for relationship types in the knowledge graph"""
    PURCHASED = "PURCHASED"          # Buyer -> Offer
    PREFERS = "PREFERS"              # Buyer -> Product 
    SIMILAR_TO = "SIMILAR_TO"        # Offer -> Offer
    OFFERED_BY = "OFFERED_BY"        # Offer -> Supplier
    COMPETES_WITH = "COMPETES_WITH"  # Product -> Product


class KnowledgeGraph:
    """
    Main knowledge graph class for B2B offer recommendations
    
    Manages nodes, edges, and provides traversal/query capabilities
    """
    
    def __init__(self):
        self.buyers: Dict[int, BuyerNode] = {}
        self.offers: Dict[str, OfferNode] = {}
        self.products: Dict[str, ProductNode] = {}
        self.suppliers: Dict[str, SupplierNode] = {}
        self.edges: List[GraphEdge] = []
        self.edge_index: Dict[str, List[GraphEdge]] = {}
        
    def add_buyer(self, buyer: BuyerNode) -> None:
        """Add a buyer node to the graph"""
        self.buyers[buyer.buyer_id] = buyer
        logger.debug(f"Added buyer {buyer.buyer_id}")
        
    def add_offer(self, offer: OfferNode) -> None:
        """Add an offer node to the graph"""
        self.offers[offer.offer_id] = offer
        logger.debug(f"Added offer {offer.offer_id}")
        
    def add_product(self, product: ProductNode) -> None:
        """Add a product node to the graph"""
        self.products[product.product_id] = product
        logger.debug(f"Added product {product.product_id}")
        
    def add_supplier(self, supplier: SupplierNode) -> None:
        """Add a supplier node to the graph"""
        self.suppliers[supplier.supplier_id] = supplier
        logger.debug(f"Added supplier {supplier.supplier_id}")
        
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph and update the index"""
        self.edges.append(edge)
        
        # Update edge index for fast lookups
        if edge.from_node not in self.edge_index:
            self.edge_index[edge.from_node] = []
        self.edge_index[edge.from_node].append(edge)
        
        logger.debug(f"Added edge {edge.from_node} -[{edge.relationship_type}]-> {edge.to_node}")
        
    def get_edges_from_node(self, node_id: str, relationship_type: str = None) -> List[GraphEdge]:
        """Get all edges originating from a node, optionally filtered by relationship type"""
        edges = self.edge_index.get(node_id, [])
        
        if relationship_type:
            edges = [e for e in edges if e.relationship_type == relationship_type]
            
        return edges
        
    def get_neighbors(self, node_id: str, relationship_type: str = None) -> List[Tuple[str, float]]:
        """Get neighboring nodes with their edge weights"""
        edges = self.get_edges_from_node(node_id, relationship_type)
        return [(edge.to_node, edge.weight) for edge in edges]
        
    def load_from_deals_data(self, deals_file_path: str) -> None:
        """Load graph data from deals.json file"""
        logger.info(f"Loading deals data from {deals_file_path}")
        
        try:
            with open(deals_file_path, 'r') as f:
                deals_data = json.load(f)
                
            self._process_deals_data(deals_data)
            self._calculate_derived_relationships()
            
            logger.info(f"Successfully loaded {len(self.buyers)} buyers, {len(self.offers)} offers, "
                       f"{len(self.suppliers)} suppliers, {len(self.edges)} relationships")
                       
        except Exception as e:
            logger.error(f"Error loading deals data: {e}")
            raise
            
    def _process_deals_data(self, deals_data: List[Dict]) -> None:
        """Process raw deals data into graph nodes and edges"""
        
        for deal in deals_data:
            buyer_id = deal['buyer_id']
            deal_id = deal['deal_id']
            
            # Create or update buyer node
            if buyer_id not in self.buyers:
                self.add_buyer(BuyerNode(
                    buyer_id=buyer_id,
                    credit_score=8.0,  # Default, could be loaded from separate data
                    recurring_customer=True
                ))
            
            buyer = self.buyers[buyer_id]
            
            # Process each product in the deal
            for product_item in deal['product']:
                self._process_product_item(product_item, deal_id, buyer_id, deal['contract_terms'])
                
    def _process_product_item(self, product_item: Dict, deal_id: int, buyer_id: int, contract_terms: str) -> None:
        """Process individual product within a deal"""
        
        # Handle excavator
        if product_item.get('excavator'):
            excavator = product_item['excavator']
            offer_id = excavator['id']
            
            # Create offer node
            offer = OfferNode(
                offer_id=offer_id,
                deal_id=deal_id,
                product_type="excavator",
                price=excavator['price'],
                specifications={
                    'lifting_capacity_tons': excavator['lifting_capacity_tons'],
                    'operating_weight_tons': excavator['operating_weight_tons'],
                    'max_digging_depth_m': excavator['max_digging_depth_m'],
                    'bucket_capacity_m3': excavator['bucket_capacity_m3'],
                    'brand': excavator['brand'],
                    'model': excavator['model'],
                    'year': excavator['year'],
                    'condition': excavator['condition']
                },
                contract_terms=contract_terms,
                supplier_name=excavator['brand'],
                category="heavy_equipment"
            )
            
            self.add_offer(offer)
            self._create_supplier_if_needed(excavator['brand'], "excavator")
            self._create_purchase_relationship(buyer_id, offer_id, excavator['price'])
            
        # Handle aluminum sheet
        if product_item.get('aluminum_sheet'):
            aluminum = product_item['aluminum_sheet']
            offer_id = aluminum['id']
            
            # Create offer node
            offer = OfferNode(
                offer_id=offer_id,
                deal_id=deal_id,
                product_type="aluminum_sheet",
                price=aluminum['price'],
                specifications={
                    'thickness_mm': aluminum['thickness_mm'],
                    'total_weight_kg': aluminum['total_weight_kg'],
                    'seller_name': aluminum['seller_name'],
                    'availability': aluminum['availability'],
                    'name': aluminum['name']
                },
                contract_terms=contract_terms,
                supplier_name=aluminum['seller_name'],
                category="industrial_materials"
            )
            
            self.add_offer(offer)
            self._create_supplier_if_needed(aluminum['seller_name'], "aluminum_sheet")
            self._create_purchase_relationship(buyer_id, offer_id, aluminum['price'])
            
    def _create_supplier_if_needed(self, supplier_name: str, product_type: str) -> None:
        """Create supplier node if it doesn't exist"""
        if supplier_name not in self.suppliers:
            self.add_supplier(SupplierNode(
                supplier_id=supplier_name,
                name=supplier_name,
                rating=4.5,  # Default rating
                reliability_score=0.8,
                product_categories={product_type}
            ))
        else:
            # Add product category to existing supplier
            self.suppliers[supplier_name].product_categories.add(product_type)
            
    def _create_purchase_relationship(self, buyer_id: int, offer_id: str, price: float) -> None:
        """Create PURCHASED relationship between buyer and offer"""
        
        # Calculate relationship weight based on recency and price
        weight = min(1.0, price / 100000)  # Normalize price influence
        
        edge = GraphEdge(
            from_node=f"buyer_{buyer_id}",
            to_node=offer_id,
            relationship_type=RelationshipTypes.PURCHASED,
            weight=weight,
            metadata={'price': price, 'timestamp': datetime.now().isoformat()}
        )
        
        self.add_edge(edge)
        
        # Update buyer statistics
        buyer = self.buyers[buyer_id]
        buyer.purchase_count += 1
        buyer.total_spent += price
        
        # Track preferred category
        offer = self.offers[offer_id]
        buyer.preferred_categories.add(offer.category)
        
    def _calculate_derived_relationships(self) -> None:
        """Calculate derived relationships like SIMILAR_TO and PREFERS"""
        self._calculate_offer_similarities()
        self._calculate_buyer_preferences()
        self._create_supplier_relationships()
        
    def _calculate_offer_similarities(self) -> None:
        """Calculate SIMILAR_TO relationships between offers"""
        offer_list = list(self.offers.values())
        
        for i, offer1 in enumerate(offer_list):
            for offer2 in offer_list[i+1:]:
                if offer1.product_type == offer2.product_type:
                    similarity = self._calculate_offer_similarity(offer1, offer2)
                    
                    if similarity > 0.5:  # Only create edges for meaningful similarities
                        # Bidirectional similarity
                        self.add_edge(GraphEdge(
                            from_node=offer1.offer_id,
                            to_node=offer2.offer_id,
                            relationship_type=RelationshipTypes.SIMILAR_TO,
                            weight=similarity
                        ))
                        
                        self.add_edge(GraphEdge(
                            from_node=offer2.offer_id,
                            to_node=offer1.offer_id,
                            relationship_type=RelationshipTypes.SIMILAR_TO,
                            weight=similarity
                        ))
                        
    def _calculate_offer_similarity(self, offer1: OfferNode, offer2: OfferNode) -> float:
        """
        Calculate similarity between two offers using multi-dimensional scoring
        
        Factors:
        - Product specifications similarity (40%)
        - Price range similarity (25%)
        - Contract terms similarity (20%) 
        - Supplier reputation similarity (15%)
        """
        if offer1.product_type != offer2.product_type:
            return 0.0
            
        # Specification similarity
        spec_score = self._calculate_spec_similarity(offer1, offer2)
        
        # Price similarity (closer prices = higher similarity)
        price_diff = abs(offer1.price - offer2.price) / max(offer1.price, offer2.price)
        price_score = max(0, 1 - price_diff)
        
        # Contract terms similarity (simple keyword matching for MVP)
        terms_score = self._calculate_terms_similarity(offer1.contract_terms, offer2.contract_terms)
        
        # Supplier similarity (same supplier = 1.0, different = 0.5)
        supplier_score = 1.0 if offer1.supplier_name == offer2.supplier_name else 0.5
        
        # Weighted combination
        similarity = (spec_score * 0.4 + price_score * 0.25 + 
                     terms_score * 0.2 + supplier_score * 0.15)
        
        return min(1.0, similarity)
        
    def _calculate_spec_similarity(self, offer1: OfferNode, offer2: OfferNode) -> float:
        """Calculate similarity based on product specifications"""
        vec1 = offer1.get_spec_vector()
        vec2 = offer2.get_spec_vector()
        
        if not vec1 or not vec2:
            return 0.0
            
        # Calculate cosine similarity for numerical specs
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0
            
        dot_product = sum(vec1[key] * vec2[key] for key in common_keys)
        norm1 = math.sqrt(sum(vec1[key]**2 for key in common_keys))
        norm2 = math.sqrt(sum(vec2[key]**2 for key in common_keys))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def _calculate_terms_similarity(self, terms1: str, terms2: str) -> float:
        """Calculate similarity based on contract terms (simple keyword matching)"""
        if not terms1 or not terms2:
            return 0.0
            
        # Simple keyword-based similarity for MVP
        words1 = set(terms1.lower().split())
        words2 = set(terms2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
        
    def _calculate_buyer_preferences(self) -> None:
        """Calculate PREFERS relationships between buyers and product categories"""
        for buyer_id, buyer in self.buyers.items():
            category_scores = {}
            
            # Analyze purchase history
            purchase_edges = self.get_edges_from_node(f"buyer_{buyer_id}", RelationshipTypes.PURCHASED)
            
            for edge in purchase_edges:
                offer = self.offers.get(edge.to_node)
                if offer:
                    category = offer.category
                    if category not in category_scores:
                        category_scores[category] = 0.0
                    category_scores[category] += edge.weight
                    
            # Create PREFERS relationships for strong preferences
            for category, score in category_scores.items():
                if score > 0.3:  # Threshold for meaningful preference
                    self.add_edge(GraphEdge(
                        from_node=f"buyer_{buyer_id}",
                        to_node=f"category_{category}",
                        relationship_type=RelationshipTypes.PREFERS,
                        weight=min(1.0, score),
                        metadata={'purchase_count': buyer.purchase_count}
                    ))
                    
    def _create_supplier_relationships(self) -> None:
        """Create OFFERED_BY relationships between offers and suppliers"""
        for offer_id, offer in self.offers.items():
            supplier = self.suppliers.get(offer.supplier_name)
            if supplier:
                self.add_edge(GraphEdge(
                    from_node=offer_id,
                    to_node=offer.supplier_name,
                    relationship_type=RelationshipTypes.OFFERED_BY,
                    weight=supplier.reliability_score,
                    metadata={'supplier_rating': supplier.rating}
                ))
                
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        return {
            'buyers': len(self.buyers),
            'offers': len(self.offers),
            'products': len(self.products),
            'suppliers': len(self.suppliers),
            'relationships': len(self.edges),
            'relationship_types': {
                rel_type: len([e for e in self.edges if e.relationship_type == rel_type])
                for rel_type in [RelationshipTypes.PURCHASED, RelationshipTypes.PREFERS, 
                               RelationshipTypes.SIMILAR_TO, RelationshipTypes.OFFERED_BY]
            }
        }


def create_knowledge_graph(deals_file_path: str = "mock_data/deals.json") -> KnowledgeGraph:
    """Factory function to create and populate a knowledge graph"""
    graph = KnowledgeGraph()
    
    # Get absolute path relative to the src directory
    current_dir = Path(__file__).parent.parent
    full_path = current_dir / deals_file_path
    
    graph.load_from_deals_data(str(full_path))
    
    logger.info("Knowledge graph created successfully")
    logger.info(f"Graph stats: {graph.get_graph_stats()}")
    
    return graph


# Example usage and testing
if __name__ == "__main__":
    # Create and test the knowledge graph
    graph = create_knowledge_graph()
    
    # Print some basic statistics
    stats = graph.get_graph_stats()
    print("Knowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # Test buyer recommendations
    buyer_1_purchases = graph.get_neighbors("buyer_1", RelationshipTypes.PURCHASED)
    print(f"\nBuyer 1 purchases: {buyer_1_purchases}")
    
    # Test offer similarities
    if graph.offers:
        first_offer_id = list(graph.offers.keys())[0]
        similar_offers = graph.get_neighbors(first_offer_id, RelationshipTypes.SIMILAR_TO)
        print(f"\nOffers similar to {first_offer_id}: {similar_offers}")