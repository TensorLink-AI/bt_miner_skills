"""Chutes AI LLM inference via OpenAI-compatible API.

Supports both plain chat completions and tool/function calling for
Basilica GPU operations.
"""

import json
import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI, NotFoundError

from ralph_loop.config import CHUTES_API_KEY, CHUTES_BASE_URL, CHUTES_MODEL, MAX_TOKENS_PER_TURN

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 5  # seconds; doubles each attempt


@dataclass
class LLMResponse:
    """Structured response from the LLM, supporting both text and tool calls."""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


def get_client() -> OpenAI:
    """Create an OpenAI client pointing at Chutes AI."""
    if not CHUTES_API_KEY:
        raise ValueError(
            "CHUTES_API_KEY environment variable is required. "
            "Get your key at https://chutes.ai"
        )
    return OpenAI(base_url=CHUTES_BASE_URL, api_key=CHUTES_API_KEY)


def chat(
    messages: list[dict],
    model: str = CHUTES_MODEL,
    max_tokens: int = MAX_TOKENS_PER_TURN,
) -> str:
    """Send a plain chat completion request. Returns text only.

    Kept for backwards compatibility.
    """
    response = chat_with_tools(messages, tools=None, model=model, max_tokens=max_tokens)
    return response.content


def chat_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = CHUTES_MODEL,
    max_tokens: int = MAX_TOKENS_PER_TURN,
) -> LLMResponse:
    """Send a chat completion request with optional tool definitions.

    Returns an LLMResponse with text content and/or tool calls.
    """
    client = get_client()
    logger.info(
        "Sending request to Chutes AI (model=%s, messages=%d, tools=%d)",
        model, len(messages), len(tools) if tools else 0,
    )

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    if tools:
        kwargs["tools"] = tools
        # Allow the model to choose between text and tool calls
        kwargs["tool_choice"] = "auto"

    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            result = LLMResponse(content=message.content or "")

            # Extract tool calls if present
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        arguments = {}

                    result.tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                    })

            logger.info(
                "Got response (%d chars, %d tool calls)",
                len(result.content), len(result.tool_calls),
            )
            return result

        except NotFoundError as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Chute not found for %s (attempt %d/%d), retrying in %ds...",
                    model, attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Chute not found for %s after %d attempts.", model, _MAX_RETRIES,
                )

    raise last_err  # type: ignore[misc]
