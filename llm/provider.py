"""Call an LLM through any OpenAI-compatible chat completions API.

Configured via environment variables:
  LLM_API_KEY   Required. API key for the provider.
  LLM_MODEL     Required. Model name, as expected by the provider.
  LLM_BASE_URL  Optional. Override to target a non-OpenAI, OpenAI-compatible
                endpoint (e.g. a local server). Defaults to OpenAI's API.

Run with: uv run python -m llm.provider "your prompt"
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


API_KEY_ENV = "LLM_API_KEY"
MODEL_ENV = "LLM_MODEL"
BASE_URL_ENV = "LLM_BASE_URL"


def load_client() -> tuple[OpenAI, str]:
    """Build a client and resolve the model name from the environment.

    Returns:
        The configured client and the model name to use for requests.

    Raises:
        RuntimeError: A required environment variable is missing.
    """
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is not set")

    model = os.getenv(MODEL_ENV)
    if not model:
        raise RuntimeError(f"{MODEL_ENV} is not set")

    # The SDK's default timeout (600s) lets one stalled request block for up
    # to 10 minutes with no error — during a multi-question eval sweep that
    # looks indistinguishable from a genuine hang. 60s is generous for a
    # single completion; eval.py already catches and records per-question
    # exceptions, so a timeout here fails one question and moves on instead
    # of blocking the whole run.
    client = OpenAI(api_key=api_key, base_url=os.getenv(BASE_URL_ENV), timeout=60.0)
    return client, model


def generate(prompt: str, *, system: str | None = None) -> str:
    """Send a single prompt and return the model's text response.

    Args:
        prompt: The user message.
        system: Optional system prompt.

    Returns:
        The response text.
    """
    client, model = load_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content


def generate_with_tools(messages: list[dict], tools: list[dict], *, tool_choice: str = "auto"):
    """Send a conversation with tool/function definitions and return the raw reply.

    Args:
        messages: Full chat history in OpenAI message format (system/user/
            assistant/tool roles).
        tools: OpenAI-format function-calling tool definitions.
        tool_choice: "auto" lets the model decide whether to call a tool;
            "required" forces it to call one every turn.

    Returns:
        The assistant's ChatCompletionMessage. Inspect `.content` and
        `.tool_calls`; pass it through `message_to_dict` before appending it
        to `messages` for a follow-up request.
    """
    client, model = load_client()
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice, temperature=0
    )
    return response.choices[0].message


def message_to_dict(message) -> dict:
    """Convert an assistant response message into history-appendable dict form."""
    result: dict = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python -m llm.provider PROMPT")
    print(generate(sys.argv[1]))
