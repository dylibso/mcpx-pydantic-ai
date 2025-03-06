import mcp_run
import pydantic_ai
import pydantic
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from typing import TypedDict, List, Set, AsyncIterator, Any
import traceback

__all__ = ["BaseModel", "Field", "Agent", "mcp_run", "pydantic_ai", "pydantic"]


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


def openai_compatible_model(url: str, model: str, api_key: str | None = None):
    """
    Returns an OpenAI compatible model from the provided `url`, `model` name and optional `api_key`
    """
    provider = OpenAIProvider(base_url=url, api_key=api_key)
    return OpenAIModel(model, provider=provider)


class Agent(pydantic_ai.Agent):
    """
    A Pydantic Agent using tools from mcp.run
    """

    client: mcp_run.Client
    ignore_tools: Set[str]
    _original_tools: list
    _registered_tools: List[str]

    def __init__(
        self,
        *args,
        client: mcp_run.Client | None = None,
        ignore_tools: List[str] | None = None,
        **kw,
    ):
        self.client = client or mcp_run.Client()
        self._original_tools = kw.get("tools", [])
        self._registered_tools = []
        self.ignore_tools = set(ignore_tools or [])
        super().__init__(*args, **kw)
        self._update_tools()

        for t in self._original_tools:
            self._registered_tools.append(t.name)

    def set_profile(self, profile: str):
        self.client.set_profile(profile)
        self._update_tools()

    def register_tool(self, tool: mcp_run.Tool, f=None):
        if tool.name in self.ignore_tools:
            return

        def wrap(tool, inner):
            props = tool.input_schema["properties"]
            t = {k: _convert_type(v["type"]) for k, v in props.items()}
            InputType = TypedDict("Input", t)

            if inner is not None:

                def f(input: InputType):
                    try:
                        return inner(input)
                    except Exception as exc:
                        return f"ERROR call to tool {tool.name} failed: {traceback.format_exception(exc)}"
            else:

                def f(input: InputType):
                    try:
                        res = self.client.call_tool(tool=tool.name, params=input)
                        return res.content[0].text
                    except Exception as exc:
                        return f"ERROR call to tool {tool.name} failed: {traceback.format_exception(exc)}"

            return f

        self._register_tool(
            pydantic_ai.Tool(
                wrap(tool, f),
                name=tool.name,
                description=tool.description,
            )
        )

        if f is not None:
            self._registered_tools.append(tool.name)

    def reset_tools(self):
        for k in list(self._function_tools.keys()):
            if k not in self._registered_tools:
                del self._function_tools[k]

    def _update_tools(self):
        self.reset_tools()
        for tool in self.client.tools.values():
            self.register_tool(tool)

    async def run(self, *args, update_tools: bool = True, **kw):
        if update_tools:
            self._update_tools()
        return await super().run(*args, **kw)

    def run_sync(self, *args, update_tools: bool = True, **kw):
        if update_tools:
            self._update_tools()
        return super().run_sync(*args, **kw)

    async def run_async(self, *args, update_tools: bool = True, **kw):
        if update_tools:
            self._update_tools()
        return await super().run_async(*args, **kw)

    def run_stream(
        self,
        *args,
        update_tools: bool = True,
        **kw,
    ) -> AsyncIterator[Any]:
        if update_tools:
            self._update_tools()
        return super().run_stream(*args, **kw)

    def iter(
        self,
        *args,
        update_tools: bool = True,
        **kw,
    ) -> AsyncIterator[Any]:
        if update_tools:
            self._update_tools()
        return super().iter(*args, **kw)
