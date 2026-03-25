"""Chutes AI LLM inference via OpenAI-compatible API."""

import logging
import time

from openai import OpenAI, NotFoundError

from ralph_loop.config import CHUTES_API_KEY, CHUTES_BASE_URL, CHUTES_MODEL, MAX_TOKENS_PER_TURN

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 5  # seconds; doubles each attempt


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
    """Send a chat completion request to Chutes AI and return the response text.

    Retries with exponential backoff on 404 errors, which occur transiently
    when a chute instance is not yet available.
    """
    client = get_client()
    logger.info("Sending request to Chutes AI (model=%s, messages=%d)", model, len(messages))

    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            logger.info("Got response (%d chars)", len(content) if content else 0)
            return content or ""
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
