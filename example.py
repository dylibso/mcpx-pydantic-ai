from mcpx_pydantic_ai import Agent, BaseModel, Field

from typing import List
import readline

from pydantic_ai import capture_run_messages


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


def run(agent, msg):
    """
    Send a message to an agent and return the result
    """
    global history
    with capture_run_messages() as messages:
        result = agent.run_sync(msg, message_history=history)
        history.extend(messages)
    return result.data


types = ImageList | VowelCount
agent = new_agent(types)

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
        agent = new_agent(types | t)
        continue
    res = run(agent, msg)
    print(">>", res)
