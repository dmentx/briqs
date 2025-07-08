"""
CrewAI Knowledge Graph Tool Integration

This module integrates our Knowledge Graph MCP server with CrewAI agents using MCPServerAdapter.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphMCPTool:
    """CrewAI integration for Knowledge Graph MCP Server."""
    
    def __init__(self, python_path: str = "python", server_script_path: Optional[str] = None):
        self.python_path = python_path
        self.server_script_path = server_script_path or self._get_server_path()
        self._adapter: Optional[MCPServerAdapter] = None
        self._tools: List = []
        
    def _get_server_path(self) -> str:
        """Auto-detect the path to our MCP server script."""
        current_dir = Path(__file__).parent.parent
        server_path = current_dir / "knowledge_graph" / "mcp_server.py"
        
        if not server_path.exists():
            raise FileNotFoundError(f"MCP server script not found at {server_path}")
        
        return str(server_path)
    
    def get_server_params(self) -> StdioServerParameters:
        """Get server parameters for the Knowledge Graph MCP server."""
        return StdioServerParameters(
            command=self.python_path,
            args=[self.server_script_path],
            env={
                "PYTHONPATH": str(Path(__file__).parent.parent.parent),
                **os.environ
            }
        )
    
    def create_context_manager(self):
        """Create a context manager for managed tool access."""
        server_params = self.get_server_params()
        return MCPServerAdapter(server_params)

# Convenience functions
def create_knowledge_graph_context():
    """Create a context manager for Knowledge Graph MCP tools."""
    kg_tool = KnowledgeGraphMCPTool()
    return kg_tool.create_context_manager()

def test_mcp_connection():
    """Test the MCP server connection and tool availability."""
    try:
        kg_tool = KnowledgeGraphMCPTool()
        
        with kg_tool.create_context_manager() as tools:
            tool_names = [tool.name for tool in tools]
            
            return {
                "status": "success",
                "connected": True,
                "available_tools": tool_names,
                "tool_count": len(tool_names)
            }
            
    except Exception as e:
        return {
            "status": "error", 
            "connected": False,
            "error": str(e),
            "available_tools": [],
            "tool_count": 0
        }

if __name__ == "__main__":
    print("🔍 Testing Knowledge Graph MCP Connection")
    result = test_mcp_connection()
    print(f"Status: {result['status']}")
    if result['connected']:
        print(f"Tools: {', '.join(result['available_tools'])}")
    else:
        print(f"Error: {result['error']}") 