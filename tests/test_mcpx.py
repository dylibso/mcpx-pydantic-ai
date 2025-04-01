import unittest
from unittest.mock import Mock, patch
from mcpx_pydantic_ai import Agent, _convert_type
from typing import Dict, Any
import os

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

    def call_tool(self, tool: str, params: Dict[str, Any]) -> MockResponse:
        self.called_tool = tool
        self.called_params = params
        return MockResponse("mock response")

    def set_profile(self, profile: str):
        self.profile = profile

    def _make_pydantic_function(self, tool):
        def test(input: dict):
            return self.call_tool(tool.name, input).content[0].text
        return test


class TestTypeConversion(unittest.TestCase):
    def test_convert_basic_types(self):
        self.assertEqual(_convert_type("string"), str)
        self.assertEqual(_convert_type("boolean"), bool)
        self.assertEqual(_convert_type("number"), float)
        self.assertEqual(_convert_type("integer"), int)
        self.assertEqual(_convert_type("object"), dict)
        self.assertEqual(_convert_type("array"), list)

    def test_convert_invalid_type(self):
        with self.assertRaises(TypeError):
            _convert_type("invalid_type")


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

    def test_init_with_ignore_tools(self):
        """Test agent initialization with ignored tools"""
        agent = Agent(
            model="claude-3-5-sonnet-latest",
            client=self.mock_client,
            ignore_tools=["test_tool"],
            system_prompt="test prompt",
        )
        self.assertEqual(
            len(agent._function_tools), 0
        )  # Should have no tools due to ignore

    def test_set_profile(self):
        """Test setting profile updates client profile"""
        self.agent.set_profile("test_profile")
        self.assertEqual(self.mock_client.profile, "test_profile")

    def test_register_custom_tool(self):
        """Test registering a custom tool with custom function"""
        custom_mock = Mock(return_value="custom response")

        self.agent.register_tool(
            MockTool(
                "custom_tool",
                "A custom tool",
                {"properties": {"param": {"type": "string"}}},
            ),
            custom_mock,
        )

        # Verify tool was registered
        self.assertIn("custom_tool", self.agent._function_tools)

        # Test tool execution
        tool_func = self.agent._function_tools["custom_tool"].function
        result = tool_func({"param": "test"})

        custom_mock.assert_called_once_with({"param": "test"})
        self.assertEqual(result, "custom response")

    def test_tool_execution(self):
        """Test executing a registered tool"""
        # Our mock tool should be registered automatically
        tool_func = self.agent._function_tools["test_tool"].function

        result = tool_func({"param1": "test", "param2": 123})

        self.assertEqual(self.mock_client.called_tool, "test_tool")
        self.assertEqual(
            self.mock_client.called_params, {"param1": "test", "param2": 123}
        )
        self.assertEqual(result, "mock response")

    def test_reset_tools(self):
        """Test resetting tools"""
        # Add a custom tool
        self.agent.register_tool(
            MockTool(
                "custom_tool",
                "A custom tool",
                {"properties": {"param": {"type": "string"}}},
            ),
            Mock(),
        )

        # Reset tools
        self.agent.reset_tools()

        # Only custom tool should remain
        self.assertEqual(len(self.agent._function_tools), 1)
        self.assertIn("custom_tool", self.agent._function_tools)
        self.assertNotIn("test_tool", self.agent._function_tools)

    @patch("mcpx_pydantic_ai.pydantic_ai.Agent.run_sync")
    def test_run_sync_updates_tools(self, mock_run_sync):
        """Test that run_sync updates tools by default"""
        mock_run_sync.return_value = "test response"

        result = self.agent.run_sync("test prompt")

        self.assertEqual(result, "test response")
        mock_run_sync.assert_called_once()

    @patch("mcpx_pydantic_ai.pydantic_ai.Agent.run_sync")
    def test_run_sync_without_tool_update(self, mock_run_sync):
        """Test that run_sync can skip tool updates"""
        mock_run_sync.return_value = "test response"

        result = self.agent.run_sync("test prompt", update_tools=False)

        self.assertEqual(result, "test response")
        mock_run_sync.assert_called_once()


if __name__ == "__main__":
    unittest.main()
