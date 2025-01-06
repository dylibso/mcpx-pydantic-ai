# mcpx-pydantic-ai

A Python library for using mcp.run tools with [pydantic-ai](https://github.com/pydantic/pydantic-ai)

## Example

```python
agent = Agent("claude-3-5-sonnet-latest", result_type=int)
results = agent.run_sync(
    "find the largest prime under 1000 that ends with the digit '3'"
)
print(results.data)
```
