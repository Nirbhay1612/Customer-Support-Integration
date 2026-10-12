"""
Automation workflows for the AI Customer Support Integration System.

Coordinates customer notifications, ticket updates, escalation notifications,
and verified webhook processing. Business actions remain outside the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from src.integrations.email_service import (
    EmailService,
    EmailServiceError,
    send_email,
)
from src.integrations.ticket_service import TicketService
from src.integrations.webhook import WebhookError, WebhookEvent, WebhookService
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_EMAIL_LENGTH = 254
MAX_SUBJECT_LENGTH = 200
MAX_MESSAGE_LENGTH = 10_000
MAX_IDENTIFIER_LENGTH = 200
MAX_WEBHOOK_PAYLOAD_BYTES = 1_000_000

VALID_TICKET_STATUSES = {
    "open",
    "in_progress",
    "pending",
    "resolved",
    "closed",
}


@dataclass
class AutomationResult:
    """Standard result returned by automation operations."""

    success: bool
    action: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the result as a dictionary."""
        return {
            "success": self.success,
            "action": self.action,
            "message": self.message,
            "data": self.data.copy(),
        }


class SupportAutomation:
    """Coordinate customer-support automation actions."""

    def __init__(
        self,
        email_service: EmailService | None = None,
        ticket_service: TicketService | None = None,
        webhook_service: WebhookService | None = None,
    ) -> None:
        self.email_service = email_service
        self.ticket_service = ticket_service
        self.webhook_service = webhook_service

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
        max_length: int = MAX_MESSAGE_LENGTH,
    ) -> str:
        """Validate required text and enforce a maximum length."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} cannot be empty.")

        normalized = value.strip()
        if len(normalized) > max_length:
            raise ValueError(
                f"{field_name} exceeds the maximum length of {max_length}."
            )
        return normalized

    @staticmethod
    def _validate_email(email: str | None) -> str:
        """Validate a customer email address."""
        normalized = SupportAutomation._validate_text(
            email, "Customer email", MAX_EMAIL_LENGTH  # type: ignore[arg-type]
        )
        if not re.fullmatch(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+", normalized):
            raise ValueError("Customer email address is invalid.")
        return normalized

    @staticmethod
    def _validate_identifier(value: str, field_name: str) -> str:
        """Validate an identifier."""
        return SupportAutomation._validate_text(
            value, field_name, MAX_IDENTIFIER_LENGTH
        )

    @staticmethod
    def _failure(action: str, message: str) -> AutomationResult:
        """Build a consistent failed-operation result."""
        return AutomationResult(False, action, message)

    def _send_customer_email(
        self,
        recipient: str,
        subject: str,
        message: str,
    ) -> bool:
        """
        Send via an injected service when configured, otherwise use the
        module-level helper for backwards compatibility with existing mocks.
        """
        if self.email_service is not None:
            return self.email_service.send_email(
                recipient=recipient,
                subject=subject,
                body=message,
            )
        return send_email(
            recipient=recipient,
            subject=subject,
            body=message,
        )

    def notify_customer(
        self,
        email: str | None = None,
        subject: str = "",
        message: str = "",
        *,
        customer_email: str | None = None,
    ) -> AutomationResult:
        """Send a customer email; supports both email parameter names."""
        raw_email = customer_email if customer_email is not None else email
        validated_email = self._validate_email(raw_email)
        subject = self._validate_text(
            subject, "Email subject", MAX_SUBJECT_LENGTH
        )
        message = self._validate_text(message, "Notification message")

        try:
            sent = self._send_customer_email(
                recipient=validated_email,
                subject=subject,
                message=message,
            )
            if sent is False:
                return self._failure(
                    "CUSTOMER_NOTIFICATION",
                    "Customer notification could not be sent.",
                )
            return AutomationResult(
                True,
                "CUSTOMER_NOTIFICATION",
                "Customer notification sent successfully.",
                {"email": validated_email, "subject": subject},
            )
        except Exception:
            logger.warning("Customer notification failed.", exc_info=True)
            return self._failure(
                "CUSTOMER_NOTIFICATION",
                "Customer notification could not be sent.",
            )

    def notify_order_update(
        self,
        email: str | None = None,
        order_id: str = "",
        status: str = "",
        message: str | None = None,
        *,
        customer_email: str | None = None,
    ) -> AutomationResult:
        """Notify a customer about an order status update."""
        raw_email = customer_email if customer_email is not None else email
        validated_email = self._validate_email(raw_email)
        order_id = self._validate_identifier(order_id, "Order ID")
        status = self._validate_text(status, "Order status", 100)
        body = message or (
            f"Your order {order_id} has been updated. "
            f"Current status: {status}."
        )
        body = self._validate_text(body, "Notification message")
        return self.notify_customer(
            customer_email=validated_email,
            subject=f"Order Update - {order_id}",
            message=body,
        )

    def notify_ticket_created(
        self,
        email: str | None = None,
        ticket_id: str = "",
        subject: str = "",
        *,
        customer_email: str | None = None,
    ) -> AutomationResult:
        """Notify a customer that a support ticket was created."""
        raw_email = customer_email if customer_email is not None else email
        validated_email = self._validate_email(raw_email)
        ticket_id = self._validate_identifier(ticket_id, "Ticket ID")
        subject = self._validate_text(
            subject, "Ticket subject", MAX_SUBJECT_LENGTH
        )
        return self.notify_customer(
            customer_email=validated_email,
            subject=f"Support Ticket Created - {ticket_id}",
            message=(
                f"Your support ticket {ticket_id} has been created "
                "successfully.\n\n"
                f"Subject: {subject}\n\n"
                "Our support team will review your request."
            ),
        )

    def update_ticket_status(
        self,
        ticket_id: str,
        status: str,
    ) -> AutomationResult:
        """Update the status of a ticket using the configured ticket service."""
        ticket_id = self._validate_identifier(ticket_id, "Ticket ID")
        status = self._validate_text(status, "Ticket status", 50)
        normalized = status.casefold().replace(" ", "_")

        if normalized not in VALID_TICKET_STATUSES:
            raise ValueError(f"Unsupported ticket status: {status}.")

        if self.ticket_service is None:
            return self._failure(
                "TICKET_STATUS_UPDATE",
                "Ticket service is not configured.",
            )

        try:
            result = self.ticket_service.update_ticket(
                ticket_id=ticket_id,
                status=normalized,
            )
            return AutomationResult(
                True,
                "TICKET_STATUS_UPDATE",
                "Ticket status updated successfully.",
                {
                    "ticket_id": ticket_id,
                    "status": normalized,
                    "result": result,
                },
            )
        except Exception:
            logger.warning("Ticket status update failed.", exc_info=True)
            return self._failure(
                "TICKET_STATUS_UPDATE",
                "Ticket status could not be updated.",
            )

    def notify_escalation(
        self,
        email: str | None = None,
        ticket_id: str | None = None,
        reason: str = "Support request requires human assistance.",
        *,
        customer_email: str | None = None,
    ) -> AutomationResult:
        """Notify a customer that their support request was escalated."""
        raw_email = customer_email if customer_email is not None else email
        validated_email = self._validate_email(raw_email)
        reason = self._validate_text(reason, "Escalation reason", 1_000)

        if ticket_id is not None:
            ticket_id = self._validate_identifier(ticket_id, "Ticket ID")
            ticket_text = f"Ticket ID: {ticket_id}\n"
        else:
            ticket_text = (
                "Your support team will provide ticket details when available.\n"
            )

        return self.notify_customer(
            customer_email=validated_email,
            subject="Support Request Escalated",
            message=(
                "Your support request has been escalated to our support team.\n\n"
                f"Reason: {reason}\n{ticket_text}\n"
                "A support representative will review your request."
            ),
        )

    def process_webhook(
        self,
        payload: str | bytes | dict[str, Any] | None,
        signature: str | None = None,
    ) -> AutomationResult:
        """Verify and route a webhook without executing untrusted instructions."""
        if payload is None or not isinstance(payload, (str, bytes, dict)):
            raise TypeError(
                "Webhook payload must be a string, bytes, or dictionary."
            )

        if self.webhook_service is None:
            return self._failure(
                "WEBHOOK_PROCESSING",
                "Webhook service is not configured.",
            )

        if not isinstance(signature, str) or not signature.strip():
            return self._failure(
                "WEBHOOK_PROCESSING",
                "Webhook signature is required.",
            )

        try:
            if isinstance(payload, dict):
                # If callers pass a dict, the signature must have been created
                # over this exact canonical JSON representation. HTTP handlers
                # should normally pass the original raw request body instead.
                webhook_payload: str | bytes = json.dumps(
                    payload,
                    separators=(",", ":"),
                    sort_keys=True,
                    ensure_ascii=False,
                )
            else:
                webhook_payload = payload

            payload_size = (
                len(webhook_payload.encode("utf-8"))
                if isinstance(webhook_payload, str)
                else len(webhook_payload)
            )
            if payload_size > MAX_WEBHOOK_PAYLOAD_BYTES:
                return self._failure(
                    "WEBHOOK_PROCESSING",
                    "Webhook payload exceeds the maximum permitted size.",
                )

            event = self.webhook_service.process_webhook(
                payload=webhook_payload,
                signature=signature.strip(),
            )
            if event is None:
                return self._failure(
                    "WEBHOOK_PROCESSING",
                    "Webhook service returned no event.",
                )

            return self._handle_webhook_event(event)

        except (WebhookError, ValueError, TypeError):
            logger.warning("Webhook validation or processing failed.")
            return self._failure(
                "WEBHOOK_PROCESSING",
                "Webhook could not be validated or processed.",
            )
        except Exception:
            logger.exception("Unexpected webhook processing failure.")
            return self._failure(
                "WEBHOOK_PROCESSING",
                "An unexpected error prevented webhook processing.",
            )

    def _handle_webhook_event(
        self,
        event: WebhookEvent,
    ) -> AutomationResult:
        """Dispatch a verified event to its registered handler."""
        event_type = self._get_event_type(event)
        data = self._get_event_data(event)

        handlers = {
            "order.updated": self._handle_order_updated,
            "order.shipped": self._handle_order_shipped,
            "order.delivered": self._handle_order_delivered,
            "ticket.created": self._handle_ticket_created,
            "ticket.updated": self._handle_ticket_updated,
        }

        handler = handlers.get(event_type)
        if handler is None:
            return AutomationResult(
                True,
                "WEBHOOK_RECEIVED",
                "Webhook received; no handler is registered for this event.",
                {"event_type": event_type},
            )
        return handler(data)

    @staticmethod
    def _get_event_type(event: WebhookEvent) -> str:
        """Extract an event type while supporting compatible event objects."""
        value = getattr(event, "event_type", None) or getattr(event, "type", None)
        return (
            value.strip().casefold()
            if isinstance(value, str) and value.strip()
            else "unknown"
        )

    @staticmethod
    def _get_event_data(event: WebhookEvent) -> dict[str, Any]:
        """Prefer the normalized event data, falling back to its payload."""
        data = getattr(event, "data", None)
        if isinstance(data, dict):
            return data

        payload = getattr(event, "payload", None)
        if isinstance(payload, dict):
            nested = payload.get("data")
            return nested if isinstance(nested, dict) else payload
        return {} if payload is None else {"payload": payload}

    @staticmethod
    def _handle_order_updated(data: dict[str, Any]) -> AutomationResult:
        """Handle a verified order.updated event."""
        return AutomationResult(
            True, "ORDER_UPDATED", "Order update event processed.", data.copy()
        )

    @staticmethod
    def _handle_order_shipped(data: dict[str, Any]) -> AutomationResult:
        """Handle a verified order.shipped event."""
        return AutomationResult(
            True, "ORDER_SHIPPED", "Order shipped event processed.", data.copy()
        )

    @staticmethod
    def _handle_order_delivered(data: dict[str, Any]) -> AutomationResult:
        """Handle a verified order.delivered event."""
        return AutomationResult(
            True, "ORDER_DELIVERED", "Order delivered event processed.", data.copy()
        )

    @staticmethod
    def _handle_ticket_created(data: dict[str, Any]) -> AutomationResult:
        """Handle a verified ticket.created event."""
        return AutomationResult(
            True, "TICKET_CREATED", "Ticket creation event processed.", data.copy()
        )

    @staticmethod
    def _handle_ticket_updated(data: dict[str, Any]) -> AutomationResult:
        """Handle a verified ticket.updated event."""
        return AutomationResult(
            True, "TICKET_UPDATED", "Ticket update event processed.", data.copy()
        )


_automation: SupportAutomation | None = None
_automation_lock = RLock()


def get_automation() -> SupportAutomation:
    """Return the shared SupportAutomation instance."""
    global _automation
    with _automation_lock:
        if _automation is None:
            _automation = SupportAutomation()
        return _automation


def configure_automation(
    email_service: EmailService | None = None,
    ticket_service: TicketService | None = None,
    webhook_service: WebhookService | None = None,
) -> SupportAutomation:
    """Configure the shared automation instance and its dependencies."""
    global _automation
    with _automation_lock:
        _automation = SupportAutomation(
            email_service=email_service,
            ticket_service=ticket_service,
            webhook_service=webhook_service,
        )
        return _automation


def notify_customer(
    email: str | None = None,
    subject: str = "",
    message: str = "",
    *,
    customer_email: str | None = None,
) -> AutomationResult:
    """Send a customer notification through the shared automation."""
    return get_automation().notify_customer(
        email=email,
        customer_email=customer_email,
        subject=subject,
        message=message,
    )


def notify_order_update(
    email: str | None = None,
    order_id: str = "",
    status: str = "",
    message: str | None = None,
    *,
    customer_email: str | None = None,
) -> AutomationResult:
    """Send an order update notification."""
    return get_automation().notify_order_update(
        email=email,
        customer_email=customer_email,
        order_id=order_id,
        status=status,
        message=message,
    )


def notify_ticket_created(
    email: str | None = None,
    ticket_id: str = "",
    subject: str = "",
    *,
    customer_email: str | None = None,
) -> AutomationResult:
    """Send a ticket-created notification."""
    return get_automation().notify_ticket_created(
        email=email,
        customer_email=customer_email,
        ticket_id=ticket_id,
        subject=subject,
    )


def update_ticket_status(ticket_id: str, status: str) -> AutomationResult:
    """Update a ticket using the shared automation."""
    return get_automation().update_ticket_status(
        ticket_id=ticket_id,
        status=status,
    )


def notify_escalation(
    email: str | None = None,
    ticket_id: str | None = None,
    reason: str = "Support request requires human assistance.",
    *,
    customer_email: str | None = None,
) -> AutomationResult:
    """Send an escalation notification."""
    return get_automation().notify_escalation(
        email=email,
        customer_email=customer_email,
        ticket_id=ticket_id,
        reason=reason,
    )


def process_webhook(
    payload: str | bytes | dict[str, Any] | None,
    signature: str | None = None,
) -> AutomationResult:
    """Verify and route a webhook through the shared automation."""
    return get_automation().process_webhook(
        payload=payload,
        signature=signature,
    )
