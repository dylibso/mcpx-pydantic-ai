from mcpx_pydantic_ai import Agent


def main():
    agent = Agent("claude-3-5-sonnet-latest", result_type=int)
    results = agent.run_sync(
        "find the largest prime under 1000 that ends with the digit '3'"
    )
    print(results.data)


if __name__ == "__main__":
    main()
