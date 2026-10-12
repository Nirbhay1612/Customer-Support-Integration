"""
Response handling layer for the AI Customer Support chatbot.

Formats workflow results, cleans generated text, sanitizes
customer-facing data, and prepares consistent UI/API responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.workflows.support_workflow import SupportResponse

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_MAX_RESPONSE_LENGTH = 5000

FALLBACK_MESSAGE = (
    "I'm sorry, but I couldn't generate a useful "
    "response for your request."
)

WRAPPER_PATTERN = re.compile(
    r"^\s*(?:(?:\*{1,2})\s*)?"
    r"(?:assistant|answer|response)"
    r"(?:\s*(?:\*{1,2}))?\s*:\s*",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------
# Standard Chatbot Response
# ---------------------------------------------------------

@dataclass
class ChatbotResponse:
    """Standard response returned to the application or UI."""

    message: str
    conversation_id: str
    intent: str | None = None
    escalated: bool = False
    sources: list[str] = field(default_factory=list)
    order_data: dict[str, Any] | None = None
    customer_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the response into a serializable dictionary."""

        return {
            "response": self.message,
            "answer": self.message,
            "conversation_id": self.conversation_id,
            "intent": self.intent,
            "escalated": self.escalated,
            "sources": list(self.sources),
            "order_data": self.order_data,
            "customer_data": self.customer_data,
            "metadata": dict(self.metadata),
        }

# ---------------------------------------------------------
# Response Handler
# ---------------------------------------------------------

class ResponseHandler:
    """Convert workflow results into customer-facing responses."""

    def __init__(
        self,
        max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH,
    ) -> None:
        if (
            isinstance(max_response_length, bool)
            or not isinstance(max_response_length, int)
            or max_response_length <= 0
        ):
            raise ValueError(
                "max_response_length must be a positive integer."
            )

        self.max_response_length = max_response_length

    # -----------------------------------------------------
    # Main Response Handling
    # -----------------------------------------------------

    def handle(
        self,
        result: SupportResponse,
    ) -> ChatbotResponse:
        """Convert a SupportResponse into a ChatbotResponse."""

        if not isinstance(result, SupportResponse):
            raise TypeError(
                "result must be a SupportResponse instance."
            )

        message = self.clean_response(result.response)

        if not message:
            message = FALLBACK_MESSAGE

        if (
            not isinstance(result.conversation_id, str)
            or not result.conversation_id.strip()
        ):
            raise ValueError(
                "Workflow conversation ID must be a non-empty string."
            )

        if result.intent is not None and not isinstance(
            result.intent, str
        ):
            raise TypeError("Workflow intent must be a string or None.")

        if not isinstance(result.escalated, bool):
            raise TypeError("Workflow escalated must be a boolean.")

        return ChatbotResponse(
            message=message,
            conversation_id=result.conversation_id.strip(),
            intent=result.intent,
            escalated=result.escalated,
            sources=self.clean_sources(result.sources),
            order_data=self._sanitize_order_data(result.order_data),
            customer_data=self._sanitize_customer_data(
                result.customer_data
            ),
            metadata={
                "response_type": (
                    "escalated" if result.escalated else "standard"
                ),
            },
        )

    # -----------------------------------------------------
    # AI Response Cleaning
    # -----------------------------------------------------

    def clean_response(self, response: str) -> str:
        """Clean AI-generated text before displaying it."""

        if not isinstance(response, str):
            return ""

        cleaned = response.replace("\x00", "").strip()

        # Normalize Windows and Unix line endings.
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize excessive blank lines and horizontal whitespace.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        # Remove a common leading LLM wrapper label.
        cleaned = WRAPPER_PATTERN.sub("", cleaned, count=1).strip()

        # Enforce the configured response length.
        if len(cleaned) > self.max_response_length:
            truncated = cleaned[: self.max_response_length]

            # Prefer a word boundary where possible.
            if " " in truncated:
                truncated = truncated.rsplit(" ", 1)[0]

            cleaned = truncated.rstrip() + "..."

        return cleaned.strip()

    # -----------------------------------------------------
    # Source Handling
    # -----------------------------------------------------

    @staticmethod
    def clean_sources(
        sources: list[str] | None,
    ) -> list[str]:
        """Remove invalid, empty, and duplicate source names."""

        if sources is None:
            return []

        if not isinstance(sources, list):
            return []

        cleaned_sources: list[str] = []
        seen: set[str] = set()

        for source in sources:
            if not isinstance(source, str):
                continue

            source = source.strip()

            if not source or source in seen:
                continue

            seen.add(source)
            cleaned_sources.append(source)

        return cleaned_sources

    # -----------------------------------------------------
    # Customer Data Sanitization
    # -----------------------------------------------------

    @staticmethod
    def _sanitize_customer_data(
        customer_data: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Keep only explicitly approved customer fields.

        This is a presentation-level filter, not an authorization
        mechanism. Access to customer records must be checked upstream.
        """

        if customer_data is None or not isinstance(customer_data, dict):
            return None

        allowed_fields = {
            "id",
            "customer_id",
            "name",
            "email",
        }

        sanitized = {
            key: value
            for key, value in customer_data.items()
            if key in allowed_fields
            and isinstance(value, (str, int, float, bool, type(None)))
        }

        return sanitized or None

    # -----------------------------------------------------
    # Escalation Formatting
    # -----------------------------------------------------

    @staticmethod
    def format_escalation_message(
        message: str,
        ticket_id: str | None = None,
    ) -> str:
        """Append a consistent escalation notice."""

        base_message = (
            message.strip() if isinstance(message, str) else ""
        )

        if not base_message:
            base_message = FALLBACK_MESSAGE

        if isinstance(ticket_id, str) and ticket_id.strip():
            return (
                f"{base_message}\n\n"
                f"Support ticket: {ticket_id.strip()}"
            )

        return (
            f"{base_message}\n\n"
            "Our support team will assist you further."
        )

    # -----------------------------------------------------
    # Order Formatting
    # -----------------------------------------------------

    @staticmethod
    def format_order_summary(
        order_data: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Extract approved customer-facing order information."""

        if order_data is None or not isinstance(order_data, dict):
            return None

        allowed_fields = {
            "id",
            "order_id",
            "status",
            "total_amount",
            "currency",
            "tracking_number",
            "estimated_delivery",
        }

        safe_value_types = (str, int, float, bool, type(None))

        summary = {
            key: value
            for key, value in order_data.items()
            if key in allowed_fields
            and isinstance(value, safe_value_types)
        }

        return summary or None

    @staticmethod
    def _sanitize_order_data(
        order_data: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Sanitize order information before exposing it in responses."""

        return ResponseHandler.format_order_summary(order_data)

    # -----------------------------------------------------
    # UI/API Response
    # -----------------------------------------------------

    def prepare_for_ui(
        self,
        response: ChatbotResponse,
    ) -> dict[str, Any]:
        """Prepare a sanitized response for Streamlit or another UI."""

        if not isinstance(response, ChatbotResponse):
            raise TypeError(
                "response must be a ChatbotResponse instance."
            )

        order_summary = self.format_order_summary(response.order_data)

        customer_data = self._sanitize_customer_data(
            response.customer_data
        )

        return {
            "response": response.message,
            "answer": response.message,
            "conversation_id": response.conversation_id,
            "intent": response.intent,
            "escalated": response.escalated,
            "sources": self.clean_sources(response.sources),
            "order": order_summary,
            "customer": customer_data,
            "metadata": dict(response.metadata),
        }

    def prepare_for_api(
        self,
        response: ChatbotResponse,
    ) -> dict[str, Any]:
        """Prepare a consistent API response envelope."""

        return {
            "success": True,
            "data": self.prepare_for_ui(response),
        }

# ---------------------------------------------------------
# Default Response Handler
# ---------------------------------------------------------

_response_handler = ResponseHandler()

# ---------------------------------------------------------
# Public Convenience Functions
# ---------------------------------------------------------

def handle_response(
    result: SupportResponse,
) -> ChatbotResponse:
    """Convert a workflow response into a chatbot response."""

    return _response_handler.handle(result)

def clean_response(response: str) -> str:
    """Clean AI-generated response text."""

    return _response_handler.clean_response(response)

def prepare_response_for_ui(
    result: SupportResponse,
) -> dict[str, Any]:
    """Prepare a workflow result for the UI."""

    chatbot_response = _response_handler.handle(result)

    return _response_handler.prepare_for_ui(chatbot_response)
