import mcpx
import pydantic_ai
from pydantic import BaseModel, Field

from typing import TypedDict
import traceback


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
    raise TypeError(f"Unhandled conversion type: {t}")


class Agent(pydantic_ai.Agent):
    client: mcpx.Client
    _original_tools: list

    def __init__(self, *args, client: mcpx.Client | None = None, **kw):
        self.client = client or mcpx.Client()
        self._original_tools = {t.name: t for t in kw.get("tools", {})}
        super().__init__(*args, **kw)
        self._update_tools()

    def _update_tools(self):
        self._function_tools = {}
        for t in self._original_tools.copy().values():
            self._register_tool(t)

        for tool in self.client.tools.values():

            def wrap(tool):
                props = tool.input_schema["properties"]
                t = {k: _convert_type(v["type"]) for k, v in props.items()}
                InputType = TypedDict("Input", t)

                def f(input: InputType):
                    try:
                        res = self.client.call(tool=tool.name, input=input)
                        return res.content[0].text
                    except Exception as exc:
                        return f"ERROR: Tool call failed, {traceback.format_exception(exc)}"

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

    async def run_async(self, *args, **kw):
        self._update_tools()
        return await super().run_async(*args, **kw)

    def run_stream(self, *args, **kw):
        self._update_tools()
        return super().run_stream(*args, **kw)
