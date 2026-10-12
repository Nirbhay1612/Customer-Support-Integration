"""
Main customer-support workflow.

Orchestrates intent classification, customer and order lookup,
knowledge retrieval, response generation, escalation, and
best-effort conversation persistence.

Business actions and access-control decisions must be enforced
by the application and authorized integrations, never by the LLM.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from src.ai.intent_classifier import SupportIntent, classify_intent
from src.ai.response_generator import (
    generate_order_response,
    generate_rag_response,
    generate_response,
)
from src.database.connection import get_db
from src.database.repository import (
    ConversationRepository,
    CustomerRepository,
)
from src.integrations.customer_api import (
    CustomerAPIClient,
    CustomerAPIError,
)
from src.integrations.order_api import (
    OrderAPIClient,
    OrderAPIError,
)
from src.rag.retriever import RetrievalResult, Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =========================================================
# Constants

# =========================================================

MAX_MESSAGE_LENGTH = 5000
MAX_IDENTIFIER_LENGTH = 256

ORDER_INTENTS = {
    SupportIntent.ORDER_STATUS.value,
    SupportIntent.SHIPPING.value,
    SupportIntent.CANCELLATION.value,
}

RAG_INTENTS = {
    SupportIntent.GENERAL_FAQ.value,
    SupportIntent.SHIPPING.value,
    SupportIntent.RETURN.value,
    SupportIntent.REFUND.value,
    SupportIntent.PAYMENT.value,
    SupportIntent.CANCELLATION.value,
    SupportIntent.PRODUCT_INFORMATION.value,
    SupportIntent.TECHNICAL_ISSUE.value,
}

ESCALATION_INTENTS = {
    SupportIntent.HUMAN_SUPPORT.value,
    SupportIntent.COMPLAINT.value,
}

ORDER_UNAVAILABLE_RESPONSE = (
    "I couldn't retrieve your order information right now. "
    "Please verify your order ID and try again, or contact "
    "our support team for assistance."
)

# =========================================================
# Data Models

# =========================================================

@dataclass

class SupportRequest:
    """Input received by the support workflow."""

    message: str
    customer_id: str | None = None
    order_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None

@dataclass

class SupportResponse:
    """Standardized output returned by the support workflow."""

    response: str
    intent: str
    conversation_id: str
    escalated: bool = False
    order_data: dict[str, Any] | None = None
    customer_data: dict[str, Any] | None = None
    sources: list[str] = field(default_factory=list)

# =========================================================
# Main Workflow

# =========================================================

class SupportWorkflow:
    """Orchestrate the complete customer-support process."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        customer_api: CustomerAPIClient | None = None,
        order_api: OrderAPIClient | None = None,
    ) -> None:
        """
        Initialize the workflow with optional dependencies.

        Dependencies can be injected for testing or configured
        by the application composition layer.
        """
        self.retriever = retriever
        self.customer_api = customer_api
        self.order_api = order_api

    def process(self, request: SupportRequest) -> SupportResponse:
        """Process a customer-support request."""

        self._validate_request(request)

        message = request.message.strip()
        conversation_id = (
            request.conversation_id or self._generate_conversation_id()
        )

        logger.info("Support request received.")

        # 1. Classify intent.
        intent = classify_intent(message)

        # 2. Retrieve customer information when configured.
        customer_data = self._get_customer(request.customer_id)

        # 3. Retrieve order information for order-related intents.
        order_data = None

        if intent in ORDER_INTENTS:
            order_data = self._get_order(request.order_id)

        # 4. Retrieve knowledge-base context when appropriate.
        retrieved_context = ""
        sources: list[str] = []

        if self._requires_rag(intent):
            retrieved_context, sources = self._retrieve_context(message)

        # 5. Generate the response from available context.
        response = self._generate_response(
            message=message,
            intent=intent,
            context=retrieved_context,
            customer_data=customer_data,
            order_data=order_data,
            order_id=request.order_id,
        )

        # 6. Determine whether human support is appropriate.
        escalated = self._should_escalate(
            intent=intent,
            message=message,
        )

        if escalated:
            response = (
                f"{response}\n\n"
                "I've flagged this issue for human support so our "
                "support team can assist you further."
            )

        # 7. Persist conversation on a best-effort basis.
        self._save_conversation(
            conversation_id=conversation_id,
            request=request,
            response=response,
            intent=intent,
        )

        logger.info(
            "Support request processed. Intent: %s, escalated: %s",
            intent,
            escalated,
        )

        return SupportResponse(
            response=response,
            intent=intent,
            conversation_id=conversation_id,
            escalated=escalated,
            order_data=order_data,
            customer_data=customer_data,
            sources=sources,
        )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_request(request: SupportRequest) -> None:
        """Validate a support request and its optional identifiers."""

        if not isinstance(request, SupportRequest):
            raise TypeError(
                "request must be a SupportRequest instance."
            )

        if not isinstance(request.message, str):
            raise ValueError("message must be a string.")

        message = request.message.strip()

        if not message:
            raise ValueError("message cannot be empty.")

        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                "message exceeds the maximum allowed length."
            )

        optional_fields = {
            "customer_id": request.customer_id,
            "order_id": request.order_id,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
        }

        for field_name, value in optional_fields.items():
            if value is None:
                continue

            if not isinstance(value, str):
                raise ValueError(
                    f"{field_name} must be a string or None."
                )

            normalized = value.strip()

            if not normalized:
                raise ValueError(
                    f"{field_name} cannot be empty when provided."
                )

            if len(normalized) > MAX_IDENTIFIER_LENGTH:
                raise ValueError(
                    f"{field_name} exceeds the maximum length."
                )

    @staticmethod
    def _generate_conversation_id() -> str:
        """Generate a unique conversation identifier."""
        return f"conv_{uuid.uuid4().hex}"

    # -----------------------------------------------------
    # Customer Lookup
    # -----------------------------------------------------

    def _get_customer(
        self,
        customer_id: str | None,
    ) -> dict[str, Any] | None:
        """Retrieve customer data through the configured API."""

        if not customer_id or self.customer_api is None:
            return None

        try:
            customer = self.customer_api.get_customer(customer_id)

            if customer is not None and not isinstance(customer, dict):
                logger.warning(
                    "Customer API returned an unexpected data type."
                )
                return None

            return customer

        except CustomerAPIError as exc:
            logger.warning(
                "Customer API lookup failed (%s).",
                type(exc).__name__,
            )
            return None

        except Exception as exc:
            logger.error(
                "Unexpected customer lookup failure (%s).",
                type(exc).__name__,
            )
            return None

    # -----------------------------------------------------
    # Order Lookup
    # -----------------------------------------------------

    def _get_order(
        self,
        order_id: str | None,
    ) -> dict[str, Any] | None:
        """Retrieve order data without guessing missing identifiers."""

        if not order_id:
            logger.info("Order lookup skipped: no order ID provided.")
            return None

        if self.order_api is None:
            logger.warning(
                "Order lookup unavailable: no API client configured."
            )
            return None

        try:
            order = self.order_api.get_order(order_id)

            if order is not None and not isinstance(order, dict):
                logger.warning(
                    "Order API returned an unexpected data type."
                )
                return None

            return order

        except OrderAPIError as exc:
            logger.warning(
                "Order API lookup failed (%s).",
                type(exc).__name__,
            )
            return None

        except Exception as exc:
            logger.error(
                "Unexpected order lookup failure (%s).",
                type(exc).__name__,
            )
            return None

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    @staticmethod
    def _requires_rag(intent: str) -> bool:
        """Determine whether the intent benefits from retrieval."""
        return intent in RAG_INTENTS

    def _retrieve_context(
        self,
        message: str,
    ) -> tuple[str, list[str]]:
        """Retrieve knowledge chunks and their source identifiers."""

        if self.retriever is None:
            logger.warning(
                "Knowledge retrieval skipped: no retriever configured."
            )
            return "", []

        try:
            results: Any = self.retriever.get_context(message)

        except Exception as exc:
            logger.error(
                "Knowledge-base retrieval failed (%s).",
                type(exc).__name__,
            )
            return "", []

        # The project's Retriever.get_context() returns a formatted string.
        # Also accept structured results for compatibility with other retrievers.
        if isinstance(results, str):
            context = results.strip()
            return context, (["Knowledge Base"] if context else [])

        if not isinstance(results, (list, tuple)):
            logger.warning("Retriever returned an unexpected result type.")
            return "", []

        context_parts: list[str] = []
        sources: list[str] = []

        for result in results:
            if not isinstance(result, RetrievalResult):
                logger.warning("Skipping invalid retrieval result.")
                continue
            if not isinstance(result.content, str):
                continue

            content = result.content.strip()
            if not content:
                continue
            context_parts.append(content)

            metadata = result.metadata
            if not isinstance(metadata, dict):
                metadata = {}

            source = (
                metadata.get("filename")
                or metadata.get("source")
                or metadata.get("Source")
                or "Knowledge Base"
            )
            if not isinstance(source, str):
                source = "Knowledge Base"
            source = source.strip() or "Knowledge Base"

            if source not in sources:
                sources.append(source)

        return "\\n\\n".join(context_parts), sources

    # -----------------------------------------------------
    # Response Generation
    # -----------------------------------------------------

    def _generate_response(
        self,
        message: str,
        intent: str,
        context: str,
        customer_data: dict[str, Any] | None,
        order_data: dict[str, Any] | None,
        order_id: str | None = None,
    ) -> str:
        """Generate a response using available verified context."""

        # Order responses must use retrieved order data.
        if intent in ORDER_INTENTS:
            if order_data is None:
                if not order_id:
                    return (
                        "Please share your order ID so I can help "
                        "you with your order."
                    )

                return ORDER_UNAVAILABLE_RESPONSE

            order_context = self._format_order_context(order_data)

            if not order_context:
                logger.warning(
                    "Order lookup returned no usable order fields."
                )
                return ORDER_UNAVAILABLE_RESPONSE

            return generate_order_response(
                message=message,
                order_context=order_context,
            )

        # Use retrieved knowledge when available.
        if context:
            return generate_rag_response(
                question=message,
                context=context,
            )

        # Keep intent-specific guidance useful when retrieval or the LLM is
        # unavailable. Do not invent a return window or eligibility policy.
        if intent == SupportIntent.RETURN.value:
            return (
                "I can help you with your return. Please share your order ID "
                "and briefly describe the item or issue, so our support team "
                "can check the applicable return options. If you already have "
                "a return request, please include its reference number."
            )

        # Preserve the existing general-response fallback.
        customer_context = self._format_customer_context(customer_data)

        return generate_response(
            message=message,
            customer_context=customer_context,
        )

    @staticmethod
    def _format_order_context(
        order_data: dict[str, Any],
    ) -> str:
        """Format selected order fields for response generation."""

        allowed_fields = (
            "id",
            "order_id",
            "status",
            "total_amount",
            "currency",
            "tracking_number",
            "estimated_delivery",
        )

        lines: list[str] = []

        for key in allowed_fields:
            value = order_data.get(key)

            if isinstance(value, (str, int, float)) and not isinstance(
                value, bool
            ):
                if isinstance(value, str) and not value.strip():
                    continue

                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    @staticmethod
    def _format_customer_context(
        customer_data: dict[str, Any] | None,
    ) -> str:
        """Format a limited set of customer fields for response context."""

        if not customer_data:
            return ""

        safe_fields = {
            "name": customer_data.get("name"),
            "email": customer_data.get("email"),
            "customer_id": (
                customer_data.get("customer_id")
                or customer_data.get("id")
            ),
        }

        return "\n".join(
            f"{key}: {value}"
            for key, value in safe_fields.items()
            if isinstance(value, str) and value.strip()
        )

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    @staticmethod
    def _should_escalate(intent: str, message: str) -> bool:
        """Determine whether human support is appropriate."""

        if intent in ESCALATION_INTENTS:
            return True

        escalation_keywords = {
            "fraud",
            "scam",
            "legal",
            "chargeback",
            "account hacked",
        }

        normalized_message = message.casefold()

        return any(
            keyword in normalized_message
            for keyword in escalation_keywords
        )

    # -----------------------------------------------------
    # Conversation Persistence
    # -----------------------------------------------------

    @staticmethod
    def _save_conversation(
        conversation_id: str,
        request: SupportRequest,
        response: str,
        intent: str,
    ) -> None:
        """Persist conversation history on a best-effort basis."""

        try:
            with get_db() as db:
                customer_id: int | None = None

                if request.customer_id:
                    customer_repo = CustomerRepository(db)

                    customer = customer_repo.get_by_external_id(
                        request.customer_id
                    )

                    if customer is not None:
                        customer_id = customer.id

                conversation_repo = ConversationRepository(db)

                conversation_repo.create(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    session_id=request.session_id,
                    user_message=request.message,
                    assistant_response=response,
                    intent=intent,
                )

        except Exception as exc:
            logger.error(
                "Failed to persist support conversation (%s).",
                type(exc).__name__,
            )

# =========================================================
# Default Workflow Instance

# =========================================================

_support_workflow: SupportWorkflow | None = None
_workflow_lock = Lock()

def _get_default_workflow() -> SupportWorkflow:
    """Lazily initialize the default workflow."""

    global _support_workflow

    if _support_workflow is None:
        with _workflow_lock:
            if _support_workflow is None:
                _support_workflow = SupportWorkflow()

    return _support_workflow

# =========================================================
# Public Convenience Function

# =========================================================

def process_support_request(
    message: str,
    customer_id: str | None = None,
    order_id: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
) -> SupportResponse:
    """Process a customer-support request using the default workflow."""

    request = SupportRequest(
        message=message,
        customer_id=customer_id,
        order_id=order_id,
        session_id=session_id,
        conversation_id=conversation_id,
    )

    return _get_default_workflow().process(request)
