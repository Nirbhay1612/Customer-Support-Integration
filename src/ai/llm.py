"""
LLM provider management for the AI Customer Support Integration System.

Supports Groq and OpenAI with reusable clients, input validation,
structured logging, and provider-independent error handling.

Security:
- API keys are never logged.
- Provider errors are translated into application exceptions.
- Connection readiness checks do not make network requests.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Any

from groq import Groq
from openai import OpenAI

from config import settings
from src.utils.exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =========================================================
# Constants
# =========================================================

REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024

# =========================================================
# Reusable Clients
# =========================================================

_groq_client: Groq | None = None
_openai_client: OpenAI | None = None

_client_lock = Lock()

# =========================================================
# Groq Client
# =========================================================

def get_groq_client() -> Groq:
    """Return a reusable, lazily initialized Groq client."""

    global _groq_client

    if _groq_client is not None:
        return _groq_client

    with _client_lock:
        if _groq_client is not None:
            return _groq_client

        api_key = settings.GROQ_API_KEY

        if not isinstance(api_key, str) or not api_key.strip():
            raise LLMConfigurationError(
                "GROQ_API_KEY is missing from the configuration."
            )

        try:
            _groq_client = Groq(
                api_key=api_key.strip(),
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )

            logger.info("Groq client initialized.")
            return _groq_client

        except Exception as exc:
            logger.error("Groq client initialization failed.")
            raise LLMError(
                "Unable to initialize the Groq client."
            ) from exc

# =========================================================
# OpenAI Client
# =========================================================

def get_openai_client() -> OpenAI:
    """Return a reusable, lazily initialized OpenAI client."""

    global _openai_client

    if _openai_client is not None:
        return _openai_client

    with _client_lock:
        if _openai_client is not None:
            return _openai_client

        api_key = settings.OPENAI_API_KEY

        if not isinstance(api_key, str) or not api_key.strip():
            raise LLMConfigurationError(
                "OPENAI_API_KEY is missing from the configuration."
            )

        try:
            _openai_client = OpenAI(
                api_key=api_key.strip(),
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )

            logger.info("OpenAI client initialized.")
            return _openai_client

        except Exception as exc:
            logger.error("OpenAI client initialization failed.")
            raise LLMError(
                "Unable to initialize the OpenAI client."
            ) from exc

# =========================================================
# LLM Configuration
# =========================================================

def get_llm_config() -> dict[str, Any]:
    """Return the active LLM provider and model configuration."""

    provider = settings.LLM_PROVIDER

    if not isinstance(provider, str):
        raise LLMConfigurationError(
            "The LLM provider configuration is invalid."
        )

    provider = provider.strip().lower()

    if not provider:
        raise LLMConfigurationError(
            "The LLM provider configuration cannot be empty."
        )

    if provider == "groq":
        model = settings.LLM_MODEL

    elif provider == "openai":
        model = settings.OPENAI_MODEL

    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: {provider}"
        )

    if not isinstance(model, str) or not model.strip():
        raise LLMConfigurationError(
            f"No model is configured for provider '{provider}'."
        )

    return {
        "provider": provider,
        "model": model.strip(),
    }

# =========================================================
# Input Validation
# =========================================================

def _validate_inputs(
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> None:
    """Validate generation inputs before making an API request."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt cannot be empty.")

    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message cannot be empty.")

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
    ):
        raise ValueError("temperature must be a number.")

    if (
        not math.isfinite(temperature)
        or not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE
    ):
        raise ValueError(
            "temperature must be finite and between 0 and 2."
        )

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer.")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

# =========================================================
# Response Handling
# =========================================================

def _extract_content(response: Any, provider: str) -> str:
    """Extract non-empty text from a chat completion response."""

    try:
        choices = response.choices

        if not choices:
            raise LLMError(
                f"{provider} returned no completion choices."
            )

        content = choices[0].message.content

    except LLMError:
        raise

    except (AttributeError, IndexError, TypeError, KeyError) as exc:
        raise LLMError(
            f"{provider} returned an unexpected response."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMError(
            f"{provider} returned an empty text response."
        )

    return content.strip()

# =========================================================
# Provider Error Translation
# =========================================================

def _translate_provider_error(
    exc: Exception,
    provider: str,
) -> LLMError:
    """
    Translate common provider failures into application exceptions.

    Avoid including raw provider error messages because they may
    contain implementation details or sensitive request metadata.
    """

    error_name = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)

    if (
        "ratelimit" in error_name
        or status_code == 429
    ):
        return LLMRateLimitError(
            f"{provider} rate limit reached. Please retry later."
        )

    if (
        "timeout" in error_name
        or isinstance(exc, TimeoutError)
    ):
        return LLMTimeoutError(
            f"{provider} request timed out."
        )

    if (
        "authentication" in error_name
        or "permissiondenied" in error_name
        or status_code in {401, 403}
    ):
        return LLMConfigurationError(
            f"{provider} authentication or permission failed. "
            "Check the API credentials and access permissions."
        )

    return LLMError(
        f"Failed to generate an AI response using {provider}."
    )

# =========================================================
# Shared Completion Function
# =========================================================

def _create_chat_completion(
    provider: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate text using the requested configured provider."""

    config = get_llm_config()

    if config["provider"] != provider:
        raise LLMConfigurationError(
            f"Provider mismatch: requested '{provider}', "
            f"configured '{config['provider']}'."
        )

    if provider == "groq":
        client = get_groq_client()

    elif provider == "openai":
        client = get_openai_client()

    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: {provider}"
        )

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = _extract_content(response, provider.title())

        logger.debug(
            "LLM response generated successfully using %s.",
            provider,
        )

        return content

    except LLMError:
        raise

    except Exception as exc:
        translated_error = _translate_provider_error(
            exc,
            provider.title(),
        )

        # Do not log raw provider exception messages or request data.
        logger.warning(
            "LLM request failed for provider '%s' (%s).",
            provider,
            type(exc).__name__,
        )

        raise translated_error from exc

# =========================================================
# Groq Response
# =========================================================

def _generate_groq_response(
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate a response using Groq."""

    return _create_chat_completion(
        provider="groq",
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# =========================================================
# OpenAI Response
# =========================================================

def _generate_openai_response(
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate a response using OpenAI."""

    return _create_chat_completion(
        provider="openai",
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# =========================================================
# Public Response Generation
# =========================================================

def generate_llm_response(
    system_prompt: str,
    user_message: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Generate a response using the configured LLM provider."""

    _validate_inputs(
        system_prompt,
        user_message,
        temperature,
        max_tokens,
    )

    config = get_llm_config()

    if config["provider"] == "groq":
        return _generate_groq_response(
            system_prompt,
            user_message,
            temperature,
            max_tokens,
        )

    return _generate_openai_response(
        system_prompt,
        user_message,
        temperature,
        max_tokens,
    )

# =========================================================
# Configuration Readiness Check
# =========================================================

def check_llm_connection() -> bool:
    """
    Check local provider configuration readiness.

    This function does not send a network request and does not
    verify API connectivity, credentials, model availability,
    account permissions, or provider service health.
    """

    try:
        config = get_llm_config()

        if config["provider"] == "groq":
            return bool(
                isinstance(settings.GROQ_API_KEY, str)
                and settings.GROQ_API_KEY.strip()
            )

        if config["provider"] == "openai":
            return bool(
                isinstance(settings.OPENAI_API_KEY, str)
                and settings.OPENAI_API_KEY.strip()
            )

        return False

    except LLMConfigurationError:
        logger.warning("LLM configuration is invalid.")
        return False
