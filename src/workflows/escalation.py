"""
Customer-support escalation workflow.

Evaluates escalation rules and optionally creates a support ticket when
escalate() is called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.ai.intent_classifier import SupportIntent
from src.integrations.ticket_service import TicketService, TicketServiceError
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_PRIORITIES = {"LOW", "MEDIUM", "HIGH"}

HUMAN_SUPPORT_PHRASES = {
    "talk to a human",
    "talk to an agent",
    "speak to an agent",
    "speak to a human",
    "human support",
    "customer support agent",
    "connect me to support",
    "connect me with an agent",
    "i want an agent",
    "i need an agent",
}

HIGH_PRIORITY_KEYWORDS = {
    "fraud",
    "scam",
    "account hacked",
    "unauthorized payment",
    "unauthorized transaction",
    "chargeback",
    "legal issue",
    "identity theft",
}

ESCALATION_KEYWORDS = {
    "fraud",
    "scam",
    "hacked",
    "chargeback",
    "legal",
    "threat",
    "unauthorized payment",
    "unauthorized transaction",
    "identity theft",
}

MAX_MESSAGE_LENGTH = 10_000
MAX_IDENTIFIER_LENGTH = 200


@dataclass
class EscalationRequest:
    """Input data used to evaluate a support escalation."""

    message: str
    intent: str
    customer_id: str | None = None
    conversation_id: str | None = None
    order_id: str | None = None
    previous_failures: int = 0
    customer_requested_human: bool = False


@dataclass
class EscalationResult:
    """Standardized result returned by the escalation workflow."""

    should_escalate: bool
    reason: str | None = None
    priority: str = "MEDIUM"
    category: str = "GENERAL_SUPPORT"
    ticket_id: str | None = None
    ticket_created: bool = False
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def escalate(self) -> bool:
        """Backward-compatible alias for should_escalate."""
        return self.should_escalate


class EscalationWorkflow:
    """Evaluate escalation rules and optionally create support tickets."""

    HUMAN_SUPPORT_PHRASES = HUMAN_SUPPORT_PHRASES
    HIGH_PRIORITY_KEYWORDS = HIGH_PRIORITY_KEYWORDS
    ESCALATION_KEYWORDS = ESCALATION_KEYWORDS

    def __init__(
        self,
        ticket_service: TicketService | None = None,
        max_failures: int = 2,
    ) -> None:
        if (
            isinstance(max_failures, bool)
            or not isinstance(max_failures, int)
            or max_failures < 1
        ):
            raise ValueError("max_failures must be a positive integer.")

        self.ticket_service = ticket_service
        self.max_failures = max_failures

    def evaluate(self, request: EscalationRequest) -> EscalationResult:
        """Evaluate escalation rules without making external API calls."""
        self._validate_request(request)
        message = request.message.casefold().strip()
        intent = request.intent.casefold().strip()

        # Rule 1: Explicit requests for human assistance.
        if request.customer_requested_human or self._contains_phrase(
            message, self.HUMAN_SUPPORT_PHRASES
        ):
            return self._build_result(
                should_escalate=True,
                reason="CUSTOMER_REQUESTED_HUMAN",
                priority="MEDIUM",
                category="HUMAN_SUPPORT",
            )

        # Rule 2: Sensitive or potentially high-impact issues.
        if self._contains_keyword(message, self.HIGH_PRIORITY_KEYWORDS):
            return self._build_result(
                should_escalate=True,
                reason="HIGH_PRIORITY_ISSUE",
                priority="HIGH",
                category="SENSITIVE_SUPPORT",
            )

        # Rule 3: Other specialist escalation cases.
        if self._contains_keyword(message, self.ESCALATION_KEYWORDS):
            return self._build_result(
                should_escalate=True,
                reason="ESCALATION_KEYWORD",
                priority="HIGH",
                category="SPECIALIST_SUPPORT",
            )

        # Rule 4: Intent-based escalation.
        escalation_intents = {
            SupportIntent.HUMAN_SUPPORT.value.casefold(),
            SupportIntent.COMPLAINT.value.casefold(),
        }
        if intent in escalation_intents:
            return self._build_result(
                should_escalate=True,
                reason="ESCALATION_INTENT",
                priority="MEDIUM",
                category="CUSTOMER_SUPPORT",
            )

        # Rule 5: Repeated unsuccessful support attempts.
        if request.previous_failures >= self.max_failures:
            return self._build_result(
                should_escalate=True,
                reason="REPEATED_SUPPORT_FAILURE",
                priority="MEDIUM",
                category="CUSTOMER_SUPPORT",
            )

        return self._build_result(should_escalate=False)

    def escalate(self, request: EscalationRequest) -> EscalationResult:
        """Evaluate a request and create a ticket when escalation is needed."""
        result = self.evaluate(request)

        if not result.should_escalate:
            result.message = (
                "This request does not currently require human escalation."
            )
            return result

        if self.ticket_service is None:
            result.message = (
                "Human support is required, but the ticket service "
                "is not configured."
            )
            return result

        if not request.customer_id:
            result.message = (
                "Human support is required. A verified customer ID "
                "is needed before a support ticket can be created."
            )
            return result

        try:
            ticket = self.ticket_service.create_ticket(
                customer_id=request.customer_id,
                subject=self._build_subject(result),
                description=self._build_description(request, result),
                priority=result.priority,
                category=result.category,
            )
            ticket_id = self._extract_ticket_id(ticket)

            if ticket_id is None:
                logger.warning(
                    "Ticket service returned no usable ticket identifier."
                )
                result.ticket_created = False
                result.message = (
                    "Human support is required, but the ticket service "
                    "did not return a valid ticket identifier."
                )
                result.metadata["ticket_creation_failed"] = True
                return result

            result.ticket_id = ticket_id
            result.ticket_created = True
            result.message = (
                "Your issue has been escalated to our support team."
            )
            result.metadata["ticket_id"] = ticket_id

            logger.info(
                "Support escalation completed; ticket_created=%s",
                result.ticket_created,
            )
            return result

        except TicketServiceError:
            logger.warning("Support escalation ticket creation failed.")
        except Exception:
            # Protect the workflow boundary from unexpected integration errors.
            logger.exception(
                "Unexpected error while creating an escalation ticket."
            )

        result.ticket_created = False
        result.message = (
            "Human support is required, but the support ticket "
            "could not be created automatically."
        )
        result.metadata["ticket_creation_failed"] = True
        return result

    @staticmethod
    def _validate_request(request: EscalationRequest) -> None:
        """Validate request fields before applying escalation rules."""
        if not isinstance(request, EscalationRequest):
            raise TypeError("request must be an EscalationRequest.")

        if not isinstance(request.message, str) or not request.message.strip():
            raise ValueError("Escalation message cannot be empty.")
        if len(request.message) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                "Escalation message exceeds the maximum length."
            )

        if not isinstance(request.intent, str) or not request.intent.strip():
            raise ValueError("Escalation intent cannot be empty.")

        if (
            isinstance(request.previous_failures, bool)
            or not isinstance(request.previous_failures, int)
            or request.previous_failures < 0
        ):
            raise ValueError(
                "previous_failures must be a non-negative integer."
            )

        if not isinstance(request.customer_requested_human, bool):
            raise ValueError(
                "customer_requested_human must be a boolean."
            )

        for field_name in ("customer_id", "conversation_id", "order_id"):
            value = getattr(request, field_name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{field_name} must be a non-empty string or None."
                    )
                if len(value.strip()) > MAX_IDENTIFIER_LENGTH:
                    raise ValueError(
                        f"{field_name} exceeds the maximum length."
                    )

    @staticmethod
    def _contains_phrase(message: str, phrases: set[str]) -> bool:
        """Match complete phrases without case sensitivity."""
        normalized = message.casefold()
        return any(
            re.search(
                rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)",
                normalized,
            )
            is not None
            for phrase in phrases
        )

    @staticmethod
    def _contains_keyword(message: str, keywords: set[str]) -> bool:
        """Match complete words or phrases rather than partial words."""
        normalized = message.casefold()
        return any(
            re.search(
                rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)",
                normalized,
            )
            is not None
            for keyword in keywords
        )

    @staticmethod
    def _build_result(
        should_escalate: bool,
        reason: str | None = None,
        priority: str = "MEDIUM",
        category: str = "GENERAL_SUPPORT",
    ) -> EscalationResult:
        """Build a consistent escalation result."""
        if priority not in VALID_PRIORITIES:
            raise ValueError("Invalid escalation priority.")
        return EscalationResult(
            should_escalate=should_escalate,
            reason=reason,
            priority=priority,
            category=category,
        )

    @staticmethod
    def _build_subject(result: EscalationResult) -> str:
        """Create a concise support-ticket subject."""
        category = result.category.replace("_", " ").title()
        return f"AI Support Escalation - {category}"

    @staticmethod
    def _build_description(
        request: EscalationRequest,
        result: EscalationResult,
    ) -> str:
        """Build a structured ticket description."""
        lines = [
            "AI Customer Support Escalation",
            "",
            f"Reason: {result.reason or 'Not specified'}",
            f"Priority: {result.priority}",
            f"Category: {result.category}",
            "",
            f"Customer ID: {request.customer_id or 'Not provided'}",
            f"Order ID: {request.order_id or 'Not provided'}",
            f"Conversation ID: {request.conversation_id or 'Not provided'}",
            f"Previous failures: {request.previous_failures}",
            "",
            "Customer Message:",
            request.message,
        ]
        return "\n".join(lines)

    @staticmethod
    def _extract_ticket_id(ticket: Any) -> str | None:
        """Extract a usable ticket ID from common API response formats."""
        if not isinstance(ticket, dict):
            return None

        ticket_id = (
            ticket.get("id")
            or ticket.get("ticket_id")
            or ticket.get("external_id")
        )
        if ticket_id is None or isinstance(ticket_id, bool):
            return None
        if not isinstance(ticket_id, (str, int, float)):
            return None

        normalized_id = str(ticket_id).strip()
        return normalized_id or None


_escalation_workflow = EscalationWorkflow()


def evaluate_escalation(
    message: str,
    intent: str,
    customer_id: str | None = None,
    conversation_id: str | None = None,
    order_id: str | None = None,
    previous_failures: int = 0,
    customer_requested_human: bool = False,
) -> EscalationResult:
    """Evaluate whether a support request should be escalated."""
    request = EscalationRequest(
        message=message,
        intent=intent,
        customer_id=customer_id,
        conversation_id=conversation_id,
        order_id=order_id,
        previous_failures=previous_failures,
        customer_requested_human=customer_requested_human,
    )
    return _escalation_workflow.evaluate(request)


def escalate_support_request(
    message: str,
    intent: str,
    customer_id: str | None = None,
    conversation_id: str | None = None,
    order_id: str | None = None,
    previous_failures: int = 0,
    customer_requested_human: bool = False,
) -> EscalationResult:
    """Evaluate a request and optionally create a support ticket."""
    request = EscalationRequest(
        message=message,
        intent=intent,
        customer_id=customer_id,
        conversation_id=conversation_id,
        order_id=order_id,
        previous_failures=previous_failures,
        customer_requested_human=customer_requested_human,
    )
    return _escalation_workflow.escalate(request)
