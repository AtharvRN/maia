import os

from .adapters import AnthropicAdapter, LocalAdapter, OpenAIAdapter
from .agent import Agent, SimpleRetry


def infer_provider(model: str, base_url: str | None = None) -> str:
    """Infer the provider name based on the model prefix."""
    if base_url:
        return "local"
    if model.startswith("gpt"):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("local-"):
        return "local"
    raise ValueError(f"Unsupported model prefix: {model}")


def create_agent(model: str, **kwargs) -> Agent:
    """
    Public entry point for creating an Agent.
    (api_key and organization can be setup as env. vars)
    Examples:
        create_agent("gpt-4o", api_key="...", organization="...")
        create_agent("claude-3-5-sonnet", api_key="...")
        create_agent("local-google/gemma-3-27b-it", base_url="http://localhost:11434/v1")
        create_agent("Qwen/Qwen3-VL-8B-Instruct", base_url="http://localhost:8000/v1")
    """
    base_url = kwargs.pop("base_url", None)
    provider: str = infer_provider(model, base_url=base_url)
    retry = SimpleRetry(
        max_attempts=kwargs.pop("max_attempts", 5), base=kwargs.pop("base", 60)
    )
    max_output_tokens: int = kwargs.pop("max_output_tokens", 4096)

    if provider == "openai":
        adapter = OpenAIAdapter(
            api_key=kwargs.pop("api_key", os.getenv("OPENAI_API_KEY")),
            organization=kwargs.pop("organization", os.getenv("OPENAI_ORGANIZATION")),
            base_url=base_url,
            model=model,
        )
    elif provider == "anthropic":
        adapter = AnthropicAdapter(
            api_key=kwargs.pop("api_key", os.getenv("ANTHROPIC_API_KEY")),
            model=model,
        )
    elif provider == "local":
        adapter = LocalAdapter(
            base_url=base_url or "http://localhost:11434/v1",
            model=model.removeprefix("local-"),
            api_key=kwargs.pop("api_key", "123"),
        )
    else:
        raise ValueError(f"Unrecognized provider: {provider}")

    return Agent(adapter=adapter, retry=retry, max_output_tokens=max_output_tokens)
