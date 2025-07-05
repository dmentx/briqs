#!/usr/bin/env python3
"""
Contract Negotiation Playbooks Demonstration
Shows how the knowledge bases guide agent behavior in different scenarios
"""

import sys
import os
import yaml
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_playbook(playbook_path):
    """Load a playbook YAML file."""
    with open(playbook_path, 'r') as f:
        return yaml.safe_load(f)

def demonstrate_buyer_playbook():
    """Demonstrate the buyer negotiation playbook."""
    print("🛒 BUYER AGENT PLAYBOOK DEMONSTRATION")
    print("="*60)
    
    playbook = load_playbook("src/knowledge_base/buyer_playbook.yaml")
    
    print("📋 PRIMARY OBJECTIVES:")
    for objective in playbook['negotiation_strategy']['primary_objectives']:
        print(f"   • {objective}")
    
    print("\n💰 BUDGET MANAGEMENT TACTICS:")
    for tactic in playbook['budget_management']['cost_optimization_tactics']:
        print(f"   • {tactic}")
    
    print("\n⚠️  KEY RISK FACTORS:")
    for risk in playbook['risk_assessment']['key_risk_factors']:
        print(f"   • {risk}")
    
    print("\n🎯 DECISION FRAMEWORK (Must-Haves):")
    for criteria in playbook['decision_frameworks']['go_no_go_criteria']['must_haves']:
        print(f"   • {criteria}")
    
    print("\n💬 SAMPLE BUYER SCENARIO:")
    print("   Scenario: Software license negotiation, $10k budget, seller asking $12k")
    print("   Buyer Strategy:")
    print("   1. Research market rates and competitive pricing")
    print("   2. Emphasize long-term partnership potential")
    print("   3. Consider total cost of ownership")
    print("   4. Explore volume discounts or extended payment terms")
    print("   5. Maintain budget discipline while seeking value")

def demonstrate_seller_playbook():
    """Demonstrate the seller negotiation playbook."""
    print("\n💼 SELLER AGENT PLAYBOOK DEMONSTRATION")
    print("="*60)
    
    playbook = load_playbook("src/knowledge_base/seller_playbook.yaml")
    
    print("📈 PRIMARY OBJECTIVES:")
    for objective in playbook['sales_strategy']['primary_objectives']:
        print(f"   • {objective}")
    
    print("\n💎 VALUE-BASED PRICING STRATEGIES:")
    for strategy in playbook['pricing_strategy']['value_based_pricing']:
        print(f"   • {strategy}")
    
    print("\n🤝 RELATIONSHIP BUILDING:")
    for tactic in playbook['customer_relationship_management']['relationship_building']:
        print(f"   • {tactic}")
    
    print("\n🛡️  OBJECTION HANDLING - Price Too High:")
    for response in playbook['objection_handling']['common_objections']['price_too_high']:
        print(f"   • {response}")
    
    print("\n💬 SAMPLE SELLER SCENARIO:")
    print("   Scenario: Software license negotiation, standard price $12k, buyer budget $10k")
    print("   Seller Strategy:")
    print("   1. Reinforce value proposition and ROI")
    print("   2. Provide competitive analysis and benchmarking")
    print("   3. Offer payment terms or financing options")
    print("   4. Consider scope adjustments to meet budget")
    print("   5. Explore multi-year deal for volume discount")

def demonstrate_orchestrator_guidelines():
    """Demonstrate the orchestrator process management guidelines."""
    print("\n🎭 ORCHESTRATOR AGENT GUIDELINES DEMONSTRATION")
    print("="*60)
    
    guidelines = load_playbook("src/knowledge_base/orchestration_guidelines.yaml")
    
    print("🎯 PRIMARY RESPONSIBILITIES:")
    for responsibility in guidelines['process_management']['primary_responsibilities']:
        print(f"   • {responsibility}")
    
    print("\n🔄 NEGOTIATION FLOW PHASES:")
    print("   1. INITIALIZATION:")
    for step in guidelines['negotiation_flow_control']['initialization_phase']:
        print(f"      • {step}")
    
    print("\n⚠️  DEADLOCK WARNING SIGNS:")
    for sign in guidelines['conflict_detection_and_escalation']['deadlock_identification']['warning_signs']:
        print(f"   • {sign}")
    
    print("\n🚨 ESCALATION TRIGGERS:")
    for trigger in guidelines['conflict_detection_and_escalation']['deadlock_identification']['escalation_triggers']:
        print(f"   • {trigger}")
    
    print("\n💬 SAMPLE ORCHESTRATION SCENARIO:")
    print("   Scenario: Managing software license negotiation")
    print("   Orchestrator Actions:")
    print("   1. Set ground rules: budget $10k, asking $12k, 2-week timeline")
    print("   2. Facilitate buyer's opening offer presentation")
    print("   3. Coordinate seller's response and counter-offers")
    print("   4. Monitor gap convergence (20% apart = escalation trigger)")
    print("   5. Document all positions and maintain neutrality")
    print("   6. Activate mediator if no progress after 3 rounds")

def demonstrate_mediator_guidelines():
    """Demonstrate the mediator conflict resolution guidelines."""
    print("\n⚖️  MEDIATOR AGENT GUIDELINES DEMONSTRATION")
    print("="*60)
    
    guidelines = load_playbook("src/knowledge_base/mediation_guidelines.yaml")
    
    print("🎯 CORE PRINCIPLES:")
    for principle in guidelines['mediation_philosophy']['core_principles']:
        print(f"   • {principle}")
    
    print("\n📊 DEAL VIABILITY CRITERIA:")
    financial = guidelines['deal_viability_assessment']['viability_criteria']['financial_viability']
    print("   Financial Viability:")
    for criteria in financial:
        print(f"      • {criteria}")
    
    print("\n🔧 BRIDGING STRATEGIES - Price Gap Resolution:")
    for strategy in guidelines['solution_development']['bridging_strategies']['price_gap_resolution']:
        print(f"   • {strategy}")
    
    print("\n📋 RECOMMENDATION TYPES:")
    for rec_type, details in guidelines['recommendation_framework']['recommendation_types'].items():
        print(f"   {rec_type.replace('_', ' ').title()}:")
        for detail in details:
            print(f"      • {detail}")
    
    print("\n💬 SAMPLE MEDIATION SCENARIO:")
    print("   Scenario: Deadlock at buyer $10k vs seller $12k")
    print("   Mediator Analysis:")
    print("   1. Review negotiation history objectively")
    print("   2. Research market rates (industry standard: $10.5k-$11.5k)")
    print("   3. Assess both positions against market fairness")
    print("   4. Recommend compromise: $11k with extended payment terms")
    print("   5. Propose 18-month deal vs 12-month for better rate")
    print("   6. Suggest performance-based pricing adjustments")

def demonstrate_integration_scenario():
    """Show how all playbooks work together in a complete scenario."""
    print("\n🎪 COMPLETE INTEGRATION SCENARIO")
    print("="*60)
    
    print("📝 SCENARIO: Enterprise Software License Negotiation")
    print("   • Contract Type: 2-year enterprise software license")
    print("   • Buyer Budget: $50,000 maximum")
    print("   • Seller Asking Price: $60,000")
    print("   • Timeline: 3 weeks to decision")
    print("   • Requirements: 500 users, premium support, quarterly payments")
    
    print("\n🎬 NEGOTIATION FLOW:")
    
    print("\n🎭 ORCHESTRATOR PHASE 1 - Initialization:")
    print("   • Sets ground rules and timeline")
    print("   • Confirms budget range ($40k-$50k) and asking price ($60k)")
    print("   • Establishes success criteria and decision framework")
    
    print("\n🛒 BUYER PHASE 1 - Opening Strategy:")
    print("   • Researches market rates ($45k-$55k for similar solutions)")
    print("   • Emphasizes 500-user volume and 2-year commitment value")
    print("   • Opens with $45k offer citing market research")
    print("   • Highlights long-term partnership potential")
    
    print("\n💼 SELLER PHASE 1 - Response Strategy:")
    print("   • Acknowledges buyer's research and volume commitment")
    print("   • Reinforces premium support value proposition")
    print("   • Counter-offers $55k with quarterly payment terms")
    print("   • Offers additional training and implementation support")
    
    print("\n🎭 ORCHESTRATOR PHASE 2 - Progress Monitoring:")
    print("   • Documents positions: Buyer $45k vs Seller $55k (18% gap)")
    print("   • Facilitates continued negotiation rounds")
    print("   • Monitors for convergence and deadlock signals")
    
    print("\n🛒 BUYER PHASE 2 - Counter-Strategy:")
    print("   • Appreciates additional services offered")
    print("   • Proposes $48k with extended 2.5-year commitment")
    print("   • Requests performance guarantees and SLA commitments")
    
    print("\n💼 SELLER PHASE 2 - Final Push:")
    print("   • Evaluates extended commitment value")
    print("   • Final offer: $52k for 2.5-year deal with premium SLA")
    print("   • Includes free annual version upgrades")
    
    print("\n🎭 ORCHESTRATOR PHASE 3 - Gap Analysis:")
    print("   • Current positions: Buyer $48k vs Seller $52k (8% gap)")
    print("   • Gap within acceptable range, no mediation needed")
    print("   • Facilitates final negotiations")
    
    print("\n✅ SUCCESSFUL RESOLUTION:")
    print("   • Final Agreement: $50k for 2.5-year license")
    print("   • Terms: 500 users, premium support, quarterly payments")
    print("   • Extras: Annual upgrades, extended implementation support")
    print("   • Both parties achieve core objectives within constraints")
    
    print("\n📊 OUTCOME ANALYSIS:")
    print("   • Buyer: Achieved budget target with extended value")
    print("   • Seller: Secured profitable deal with long-term relationship")
    print("   • Process: Efficient 4-round negotiation, no mediation required")
    print("   • Relationship: Foundation established for future partnerships")

def main():
    """Run the playbook demonstration."""
    print("🎭 MULTI-AGENT NEGOTIATION SYSTEM")
    print("📚 Contract Negotiation Playbooks Demonstration")
    print("="*70)
    
    try:
        demonstrate_buyer_playbook()
        demonstrate_seller_playbook()
        demonstrate_orchestrator_guidelines()
        demonstrate_mediator_guidelines()
        demonstrate_integration_scenario()
        
        print("\n" + "="*70)
        print("🎉 PLAYBOOK DEMONSTRATION COMPLETE!")
        print("\n💡 KEY INSIGHTS:")
        print("   • Each agent has distinct strategic guidance")
        print("   • Playbooks provide realistic negotiation behaviors")
        print("   • Orchestrator ensures structured, fair process")
        print("   • Mediator provides neutral conflict resolution")
        print("   • Integration creates sophisticated negotiation dynamics")
        
        print("\n🚀 READY TO TEST WITH OLLAMA:")
        print("   Run: uv run python test_ollama_integration.py")
        
    except FileNotFoundError as e:
        print(f"❌ Playbook file not found: {e}")
        print("   Ensure all playbook files are in src/knowledge_base/")
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")

if __name__ == "__main__":
    main() 