import mcp_run
import pydantic_ai
import pydantic
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP, MCPServerStdio
from datetime import timedelta
from mcp_run import MCPClient

from typing import TypedDict, List, Set, AsyncIterator, Any
import traceback

__all__ = ["BaseModel", "Field", "Agent", "mcp_run", "pydantic_ai", "pydantic"]


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

    def __init__(
        self,
        *args,
        client: mcp_run.Client | None = None,
        mcp_client: MCPClient | None = None,
        expires_in: timedelta | None = None,
        **kw,
    ):
        self.client = client or mcp_run.Client()
        mcp = mcp_client or self.client.mcp_sse(profile=self.client.config.profile)
        mcp_servers = kw.get("mcp_servers", [])
        if mcp.is_sse:
            mcp_servers.append(MCPServerHTTP(url=mcp.config.url))
        elif mcp.is_stdio:
            mcp_servers.append(
                MCPServerStdio(
                    command=mcp.config.command,
                    args=mcp.config.args,
                    env=mcp.config.env,
                    cwd=mcp.config.cwd,
                )
            )
        kw["mcp_servers"] = mcp_servers
        super().__init__(*args, **kw)

    def set_profile(self, profile: str):
        self.client.set_profile(profile)
