import mcp_run
import pydantic_ai
from pydantic import BaseModel, Field

from typing import TypedDict
import traceback

__all__ = ["BaseModel", "Field", "Agent"]


def _convert_type(t):
    if t == "string":
        return str
    elif t == "boolean":
        return bool
    elif t == "number":
        return float
    elif t == "integer":
        return int
    elif t == "object":
        return dict
    elif t == "array":
        return list
    raise TypeError(f"Unhandled conversion type: {t}")


class Agent(pydantic_ai.Agent):
    """
    A Pydantic Agent using tools from mcp.run
    """

    client: mcp_run.Client
    _original_tools: list

    def __init__(self, *args, client: mcp_run.Client | None = None, **kw):
        self.client = client or mcp_run.Client()
        self._original_tools = kw.get("tools", [])
        super().__init__(*args, **kw)
        self._update_tools()

    def _update_tools(self):
        if not self.client.install_cache.needs_refresh():
            return

        self._function_tools = {}
        for t in self._original_tools.copy():
            self._register_tool(t)

        for tool in self.client.tools.values():

            def wrap(tool):
                props = tool.input_schema["properties"]
                t = {k: _convert_type(v["type"]) for k, v in props.items()}
                InputType = TypedDict("Input", t)

                def f(input: InputType):
                    try:
                        res = self.client.call_tool(tool=tool.name, params=input)
                        return res.content[0].text
                    except Exception as exc:
                        print(
                            f"XXX ERROR call to tool {tool.name} failed: {traceback.format_exception(exc)}"
                        )
                        return f"ERROR call to tool {tool.name} failed: {traceback.format_exception(exc)}"

                return f

            self._register_tool(
                pydantic_ai.Tool(
                    wrap(tool),
                    name=tool.name,
                    description=tool.description,
                )
            )

    def run(self, *args, **kw):
        self._update_tools()
        return super().run(*args, **kw)

    def run_sync(self, *args, **kw):
        self._update_tools()
        return super().run_sync(*args, **kw)

    async def run_async(self, *args, **kw):
        self._update_tools()
        return await super().run_async(*args, **kw)

    def run_stream(self, *args, **kw):
        self._update_tools()
        return super().run_stream(*args, **kw)
