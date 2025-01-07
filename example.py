from mcpx_pydantic_ai import Agent, BaseModel, Field

import code
from typing import List
import readline


class ImageList(BaseModel):
    images: List[str] = Field("List of image URLs")


class VowelCount(BaseModel):
    count: int = Field("Number of vowels")
    source: str = Field("Source string")


def new_agent(result_type, model="claude-3-5-sonnet-latest"):
    """
    Create a new agent for the given return type
    """
    return Agent(model, result_type=result_type)


def run(agent, msg):
    """
    Send a message to an agent and return the result
    """
    result = agent.run_sync(msg)
    return result.data


agent = new_agent(ImageList | VowelCount)

while True:
    msg = input("> ")
    if msg == "exit":
        break
    elif len(msg) == 0:
        continue
    res = run(agent, msg)
    print(">>", res)
