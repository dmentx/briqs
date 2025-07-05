# Ollama Setup Guide for Multi-Agent Negotiation System

This guide walks you through setting up and running the multi-agent negotiation system with Ollama for local inference.

## 📋 Prerequisites

1. **Ollama installed** - Download from [ollama.ai](https://ollama.ai)
2. **Python 3.12+** with `uv` package manager
3. **Sufficient system resources** (16GB+ RAM recommended for llama4:16x17b)

## 🚀 Quick Setup

### 1. Start Ollama Server
```bash
# Start Ollama in the background
ollama serve
```

### 2. Install Required Models
```bash
# Install the target model (requires significant disk space)
ollama pull llama4:16x17b

# OR install a smaller fallback model
ollama pull llama3.1:70b
```

### 3. Verify Installation
```bash
# Check available models
ollama list

# Test model functionality
ollama run llama4:16x17b "Hello, how are you?"
```

## 🧪 Testing the Integration

### 1. Run Ollama Integration Test
```bash
cd app
uv run python test_ollama_integration.py
```

This comprehensive test will:
- ✅ Check Ollama connectivity
- ✅ Verify model availability
- ✅ Test LLM calls
- ✅ Validate crew instantiation
- ✅ Run a simple negotiation

### 2. View Playbook Demonstrations
```bash
uv run python demo_playbooks.py
```

This will show you:
- 🛒 Buyer negotiation strategies
- 💼 Seller optimization tactics  
- 🎭 Orchestrator process management
- ⚖️ Mediator conflict resolution
- 🎪 Complete integration scenario

## 📊 System Requirements by Model

| Model | RAM Required | Disk Space | Performance |
|-------|-------------|------------|-------------|
| llama4:16x17b | 24GB+ | ~90GB | Excellent |
| llama3.1:70b | 16GB+ | ~40GB | Very Good |
| llama3.1:8b | 8GB+ | ~5GB | Good |

## 🎯 Running Negotiations

### Basic Negotiation Example
```python
from src.crew_ai.crew import create_negotiation_crew

# Create the crew
crew = create_negotiation_crew()

# Run a negotiation
result = crew.run_negotiation(
    contract_type="software_license",
    buyer_budget=10000.0,
    seller_price=12000.0,
    requirements={
        "license_duration": "2 years",
        "support_level": "premium",
        "user_count": 100
    }
)

print(f"Negotiation successful: {result['success']}")
if result['success']:
    print(f"Final result: {result['result']}")
```

### Advanced Configuration
```python
from crewai import LLM

# Custom model configuration
custom_llm = LLM(
    model="ollama/llama4:16x17b",
    base_url="http://localhost:11434"
)

# Use in agent configuration
agent = Agent(
    role="Custom Negotiator",
    goal="Achieve optimal outcomes",
    backstory="Expert negotiator with custom model",
    llm=custom_llm
)
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. "Cannot connect to Ollama" Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

#### 2. "Model not found" Error
```bash
# Check available models
ollama list

# Pull the required model
ollama pull llama4:16x17b
```

#### 3. Memory Issues
- Reduce model size: Use `llama3.1:8b` instead of larger models
- Increase system swap space
- Close other memory-intensive applications

#### 4. Slow Performance
- Ensure adequate RAM for model
- Use SSD storage for model files
- Consider GPU acceleration if available

### Performance Optimization

#### 1. Model Selection
- **Production**: `llama4:16x17b` for best quality
- **Development**: `llama3.1:8b` for speed
- **Balanced**: `llama3.1:70b` for good quality/speed ratio

#### 2. System Optimization
```bash
# Set Ollama environment variables for performance
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

#### 3. Memory Management
```python
# Configure crew with memory optimization
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    max_iter=2  # Reduce iterations for faster execution
)
```

## 📈 Monitoring and Debugging

### 1. Ollama Logs
```bash
# View Ollama logs
ollama logs

# Monitor resource usage
htop
```

### 2. CrewAI Debugging
```python
# Enable verbose logging
crew = create_negotiation_crew()
crew.crew().verbose = True

# Check agent status
for agent in crew.crew().agents:
    print(f"Agent: {agent.role}, LLM: {agent.llm}")
```

### 3. Performance Metrics
```python
import time

start_time = time.time()
result = crew.run_negotiation(...)
execution_time = time.time() - start_time

print(f"Negotiation completed in {execution_time:.2f} seconds")
```

## 🎯 Next Steps

1. **Experiment with scenarios**: Try different contract types and parameters
2. **Customize playbooks**: Modify knowledge bases for specific domains
3. **Scale testing**: Run multiple concurrent negotiations
4. **Integration**: Connect to external APIs or databases
5. **Evaluation**: Measure negotiation quality and efficiency

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [CrewAI Documentation](https://docs.crewai.com)
- [Model Performance Comparisons](https://ollama.ai/models)
- [Hardware Requirements Guide](https://github.com/ollama/ollama#system-requirements)

## 🆘 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review Ollama logs: `ollama logs`
3. Test with smaller models first
4. Verify system resources meet requirements

---

🎉 **Ready to negotiate!** Run `uv run python test_ollama_integration.py` to get started. 