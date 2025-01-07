from mcpx_pydantic_ai import Agent, BaseModel, Field
import readline

import code

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

# Start interactive shell
code.interact(local=locals())
