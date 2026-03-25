"""Chutes AI LLM inference via OpenAI-compatible API."""

import logging

from openai import OpenAI

from ralph_loop.config import CHUTES_API_KEY, CHUTES_BASE_URL, CHUTES_MODEL, MAX_TOKENS_PER_TURN

logger = logging.getLogger(__name__)


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
    """Send a chat completion request to Chutes AI and return the response text."""
    client = get_client()
    logger.info("Sending request to Chutes AI (model=%s, messages=%d)", model, len(messages))

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )

    content = response.choices[0].message.content
    logger.info("Got response (%d chars)", len(content) if content else 0)
    return content or ""
