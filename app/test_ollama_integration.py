#!/usr/bin/env python3
"""
Test script for Ollama integration with Multi-Agent Negotiation System
Validates local Ollama connectivity and runs a basic negotiation scenario
"""

import sys
import os
import requests
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ollama_connectivity():
    """Test if Ollama is running and accessible."""
    print("🔍 Testing Ollama connectivity...")
    
    try:
        # Test basic Ollama API connectivity
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running and accessible")
            
            # Check available models
            model_names = [model["name"] for model in models.get("models", [])]
            print(f"📋 Available models: {model_names}")
            
            # Check for our target models
            if "llama4:16x17b" in model_names:
                print("✅ llama4:16x17b is available")
                return "llama4:16x17b"
            elif any("llama3.1" in name and "70b" in name for name in model_names):
                llama31_model = next(name for name in model_names if "llama3.1" in name and "70b" in name)
                print(f"✅ Found Llama 3.1 70B model: {llama31_model}")
                return llama31_model
            elif any("llama" in name for name in model_names):
                llama_model = next(name for name in model_names if "llama" in name)
                print(f"⚠️  Using fallback Llama model: {llama_model}")
                return llama_model
            else:
                print("❌ No suitable Llama models found")
                print("   Run: ollama pull llama3.1:70b")
                print("   Or:  ollama pull llama4:16x17b")
                return None
                
        else:
            print(f"❌ Ollama API responded with status {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("   Start Ollama with: ollama serve")
        return None
    except Exception as e:
        print(f"❌ Error testing Ollama connectivity: {e}")
        return None

def test_embedding_model():
    """Test embedding model availability (skipped - not needed)."""
    print("\n🔍 Testing embedding model availability...")
    print("✅ Embedding model not required - skipping")
    return True

def test_simple_llm_call():
    """Test a simple LLM call to verify Ollama integration."""
    print("\n🔍 Testing simple LLM call...")
    
    try:
        from crewai import LLM
        
        # Try the specific model first
        try:
            llm = LLM(
                model="ollama/llama4:16x17b",
                base_url="http://localhost:11434"
            )
            response = llm.call("Hello! Please respond with 'Ollama integration working' if you can understand this.")
            print("✅ llama4:16x17b LLM call successful")
            print(f"   Response: {response[:100]}...")
            return True
        except Exception:
            # Fallback to llama3.1
            try:
                llm = LLM(
                    model="ollama/llama3.1:70b",
                    base_url="http://localhost:11434"
                )
                response = llm.call("Hello! Please respond with 'Ollama integration working' if you can understand this.")
                print("✅ llama3.1:70b LLM call successful")
                print(f"   Response: {response[:100]}...")
                return True
            except Exception as e:
                print(f"❌ LLM call failed: {e}")
                return False
                
    except ImportError as e:
        print(f"❌ Failed to import CrewAI LLM: {e}")
        return False

def test_crew_instantiation():
    """Test that the negotiation crew can be created with Ollama."""
    print("\n🔍 Testing crew instantiation with Ollama...")
    
    try:
        from crew_ai.crew import create_negotiation_crew
        
        crew = create_negotiation_crew()
        print("✅ Negotiation crew instantiated successfully with Ollama")
        
        # Test context setting
        test_context = {
            "contract_type": "software_license",
            "buyer_budget": 10000.0,
            "seller_price": 12000.0,
            "requirements": {
                "license_duration": "1 year",
                "support_level": "basic"
            }
        }
        
        crew.set_negotiation_context(test_context)
        print("✅ Negotiation context set successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to instantiate crew with Ollama: {e}")
        return False

def test_simple_negotiation():
    """Run a minimal negotiation scenario to test the full pipeline."""
    print("\n🔍 Testing simple negotiation scenario...")
    
    try:
        from crew_ai.crew import create_negotiation_crew
        
        # Create crew
        crew = create_negotiation_crew()
        
        # Run a simple negotiation
        print("🚀 Starting negotiation test...")
        result = crew.run_negotiation(
            contract_type="test_contract",
            buyer_budget=5000.0,
            seller_price=6000.0,
            requirements={
                "duration": "6 months",
                "support": "basic"
            }
        )
        
        if result.get("success"):
            print("✅ Simple negotiation completed successfully!")
            print(f"   Negotiation ID: {result['context']['negotiation_id']}")
            return True
        else:
            print(f"❌ Negotiation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error running simple negotiation: {e}")
        return False

def main():
    """Run all Ollama integration tests."""
    print("🚀 Multi-Agent Negotiation System - Ollama Integration Test")
    print("="*70)
    
    tests = [
        ("Ollama Connectivity", test_ollama_connectivity),
        ("Embedding Model (Skipped)", test_embedding_model),
        ("Simple LLM Call", test_simple_llm_call),
        ("Crew Instantiation", test_crew_instantiation),
        ("Simple Negotiation", test_simple_negotiation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Ollama integration is working correctly.")
        print("\n🚀 Ready to run full negotiation scenarios!")
        print("   Try: uv run python -c \"from src.crew_ai.crew import run_sample_negotiation; print(run_sample_negotiation())\"")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        print("\n🔧 Setup Instructions:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Pull models: ollama pull llama4:16x17b (or llama3.1:70b)")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 