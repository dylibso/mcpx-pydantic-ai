import unittest
from unittest.mock import Mock
from typing import Dict, Any
import os


from mcpx_pydantic_ai import Agent

os.environ["ANTHROPIC_API_KEY"] = "something"


class MockTool:
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockResponse:
    def __init__(self, content):
        self.content = [Mock(text=content)]


class MockClient:
    def __init__(self):
        self.tools = {
            "test_tool": MockTool(
                "test_tool",
                "A test tool",
                {
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"},
                    }
                },
            )
        }
        self.called_tool = None
        self.called_params = None
        self.config = Mock(profile="default")

    def call_tool(self, tool: str, params: Dict[str, Any]) -> MockResponse:
        self.called_tool = tool
        self.called_params = params
        return MockResponse("mock response")

    def _make_pydantic_function(self, tool):
        def test(input: dict):
            return self.call_tool(tool.name, input).content[0].text

        return test

    def mcp_sse(self, profile=None, expires_in=None):
        mock_mcp = Mock()
        mock_mcp.is_sse = True
        mock_mcp.is_stdio = False
        mock_mcp.config = Mock(url="http://mock-url.com")
        return mock_mcp

    def _fix_profile(self, profile):
        return profile


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.mock_client = MockClient()
        self.agent = Agent(
            model="claude-3-5-sonnet-latest",
            client=self.mock_client,
            system_prompt="test prompt",
        )

    def test_init_with_custom_client(self):
        """Test agent initialization with custom client"""
        self.assertEqual(self.agent.client, self.mock_client)
        self.assertEqual(
            len(self.agent._function_tools), 1
        )  # Should have our mock tool


if __name__ == "__main__":
    unittest.main()
