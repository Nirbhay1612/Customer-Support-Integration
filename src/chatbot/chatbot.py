
"""
Main chatbot orchestration layer.

Validates incoming messages, manages conversation identifiers,
invokes the support workflow, and returns a consistent response.
"""

from __future__ import annotations

import uuid
from threading import Lock
from typing import Any

from src.workflows.support_workflow import (
    SupportRequest,
    SupportResponse,
    SupportWorkflow,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


# =========================================================
# Constants
# =========================================================

MAX_MESSAGE_LENGTH = 5000
MAX_IDENTIFIER_LENGTH = 200


# =========================================================
# Exceptions
# =========================================================

class ChatbotError(Exception):
    """Base exception for chatbot-related errors."""


# =========================================================
# Customer Support Chatbot
# =========================================================

class CustomerSupportChatbot:
    """Application-facing interface for customer support."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    MAX_IDENTIFIER_LENGTH = MAX_IDENTIFIER_LENGTH

    def __init__(
        self,
        workflow: SupportWorkflow | None = None,
    ) -> None:
        """Initialize the chatbot with an optional workflow."""
        self.workflow = (
            workflow if workflow is not None else SupportWorkflow()
        )

    # -----------------------------------------------------
    # Process Message
    # -----------------------------------------------------

    def process_message(
        self,
        message: str,
        conversation_id: str | None = None,
        customer_id: str | None = None,
        order_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a customer message and return a structured result."""

        try:
            normalized_message = self._validate_message(message)

            conversation_id = (
                self._validate_identifier(
                    conversation_id,
                    "conversation_id",
                )
                or self._generate_conversation_id()
            )

            customer_id = self._validate_identifier(
                customer_id, "customer_id"
            )
            order_id = self._validate_identifier(
                order_id, "order_id"
            )
            session_id = self._validate_identifier(
                session_id, "session_id"
            )

        except ValueError as exc:
            logger.warning(
                "Invalid chatbot request (%s).",
                type(exc).__name__,
            )
            raise ChatbotError(
                "Please provide a valid customer support request."
            ) from exc

        request = SupportRequest(
            message=normalized_message,
            customer_id=customer_id,
            order_id=order_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )

        try:
            result = self.workflow.process(request)

        except ValueError as exc:
            logger.warning(
                "Support workflow rejected a request (%s).",
                type(exc).__name__,
            )
            raise ChatbotError(
                "The customer support request could not be processed."
            ) from exc

        except Exception as exc:
            logger.error(
                "Customer support workflow failed (%s).",
                type(exc).__name__,
            )
            raise ChatbotError(
                "Unable to process the customer request. "
                "Please try again later."
            ) from exc

        try:
            return self._format_response(result)

        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            logger.error(
                "Invalid response returned by support workflow (%s).",
                type(exc).__name__,
            )
            raise ChatbotError(
                "The support workflow returned an invalid response."
            ) from exc

    # -----------------------------------------------------
    # Short Process Alias
    # -----------------------------------------------------

    def process(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Convenience alias for process_message()."""

        return self.process_message(
            message=message,
            conversation_id=conversation_id,
        )

    # -----------------------------------------------------
    # Start Conversation
    # -----------------------------------------------------

    def start_conversation(
        self,
        session_id: str | None = None,
    ) -> str:
        """Generate and return a new conversation identifier."""

        try:
            self._validate_identifier(session_id, "session_id")

        except ValueError as exc:
            logger.warning(
                "Invalid session identifier (%s).",
                type(exc).__name__,
            )
            raise ChatbotError(
                "Please provide a valid session identifier."
            ) from exc

        return self._generate_conversation_id()

    # -----------------------------------------------------
    # Response Text
    # -----------------------------------------------------

    def get_response_text(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> str:
        """Return only the customer-facing response text."""

        result = self.process_message(
            message=message,
            conversation_id=conversation_id,
        )
        return result["response"]

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    @classmethod
    def _validate_message(cls, message: object) -> str:
        """Validate and normalize an incoming customer message."""

        if not isinstance(message, str):
            raise ValueError("Message must be a string.")

        normalized = message.strip()

        if not normalized:
            raise ValueError("Message cannot be empty.")

        if len(normalized) > cls.MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message cannot exceed "
                f"{cls.MAX_MESSAGE_LENGTH} characters."
            )

        return normalized

    @classmethod
    def _validate_identifier(
        cls,
        value: str | None,
        name: str,
    ) -> str | None:
        """Validate an optional external identifier."""

        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string.")

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{name} cannot be empty.")

        if len(normalized) > cls.MAX_IDENTIFIER_LENGTH:
            raise ValueError(
                f"{name} cannot exceed "
                f"{cls.MAX_IDENTIFIER_LENGTH} characters."
            )

        return normalized

    # -----------------------------------------------------
    # Conversation ID
    # -----------------------------------------------------

    @staticmethod
    def _generate_conversation_id() -> str:
        """Generate a unique conversation identifier."""
        return f"conv_{uuid.uuid4().hex}"

    # -----------------------------------------------------
    # Response Formatting
    # -----------------------------------------------------

    @staticmethod
    def _format_response(
        result: SupportResponse,
    ) -> dict[str, Any]:
        """Convert a workflow response into a stable dictionary."""

        if not isinstance(result, SupportResponse):
            raise TypeError(
                "Workflow result must be a SupportResponse instance."
            )

        if (
            not isinstance(result.response, str)
            or not result.response.strip()
        ):
            raise ValueError("Workflow response text cannot be empty.")

        if (
            not isinstance(result.intent, str)
            or not result.intent.strip()
        ):
            raise ValueError("Workflow intent cannot be empty.")

        if (
            not isinstance(result.conversation_id, str)
            or not result.conversation_id.strip()
        ):
            raise ValueError(
                "Workflow conversation ID must be a non-empty string."
            )

        if not isinstance(result.escalated, bool):
            raise TypeError(
                "Workflow escalated flag must be a boolean."
            )

        if not isinstance(result.sources, list):
            raise TypeError("Workflow sources must be a list.")

        response_text = result.response.strip()

        return {
            "response": response_text,
            "answer": response_text,
            "intent": result.intent.strip(),
            "conversation_id": result.conversation_id.strip(),
            "escalated": result.escalated,
            "order_data": result.order_data,
            "customer_data": result.customer_data,
            "sources": result.sources,
        }


# =========================================================
# Default Chatbot Instance
# =========================================================

_chatbot: CustomerSupportChatbot | None = None
_chatbot_lock = Lock()


def get_chatbot() -> CustomerSupportChatbot:
    """Return a lazily initialized default chatbot instance."""

    global _chatbot

    if _chatbot is None:
        with _chatbot_lock:
            if _chatbot is None:
                _chatbot = CustomerSupportChatbot()

    return _chatbot


# =========================================================
# Public Convenience Function
# =========================================================

def chat(
    message: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Process a message using the default chatbot instance."""

    return get_chatbot().process_message(
        message=message,
        conversation_id=conversation_id,
    )
