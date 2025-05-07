from mcpx_pydantic_ai import Agent, BaseModel, Field

from typing import List
import readline
import asyncio


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


history = []
types = ImageList | VowelCount


async def run(msg):
    """
    Send a message to an agent and return the result
    """
    global history, types
    agent = new_agent(types)
    async with agent.run_mcp_servers():
        async with agent.run_stream(msg, message_history=history) as result:
            return await result.get_data()


while True:
    msg = input("> ")
    readline.add_history(msg)
    if msg == "exit":
        break
    elif len(msg) == 0:
        continue
    elif msg.startswith("/type"):
        msg = msg[5:].strip()
        name = msg.split(" ", maxsplit=1)[0]
        s = msg[len(name) + 1 :]
        exec(f"""
class Type_{name}(BaseModel):
    {s}
        """)
        t = eval(f"Type_{name}")
        types = types | t
        continue
    res = asyncio.run(run(msg))
    print(">>", res)
