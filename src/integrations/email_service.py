"""
Email integration for the AI Customer Support Integration System.

Features:
- SMTP configuration validation
- TLS-enabled SMTP delivery
- Compatible recipient keyword aliases
- Safe message construction and header-injection prevention
- Consistent delivery errors and health checks
"""

from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage
from math import isfinite
from threading import Lock
from typing import Final, Any

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_EMAIL_LENGTH: Final[int] = 254
MAX_SUBJECT_LENGTH: Final[int] = 200
MAX_BODY_LENGTH: Final[int] = 50_000
MAX_IDENTIFIER_LENGTH: Final[int] = 200
EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$"
)
_UNSET = object()


class EmailServiceError(RuntimeError):
    """Raised when SMTP configuration or email delivery fails."""


class EmailService:
    """Send support emails through a configured SMTP server."""

    def __init__(
        self,
        smtp_host: str | None | Any = _UNSET,
        smtp_port: int | None | Any = _UNSET,
        username: str | None | Any = _UNSET,
        password: str | None | Any = _UNSET,
        timeout: int | float = 10,
    ) -> None:
        # Omitted arguments use application settings. Explicit None is treated
        # as missing configuration, which makes configuration errors testable.
        self.smtp_host = (
            getattr(settings, "SMTP_HOST", None)
            if smtp_host is _UNSET else smtp_host
        )
        self.smtp_port = (
            getattr(settings, "SMTP_PORT", None)
            if smtp_port is _UNSET else smtp_port
        )
        self.username = (
            getattr(settings, "SMTP_USERNAME", None)
            if username is _UNSET else username
        )
        self.password = (
            getattr(settings, "SMTP_PASSWORD", None)
            if password is _UNSET else password
        )

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("SMTP timeout must be a positive finite number.")

        self.timeout = float(timeout)
        self._lock = Lock()
        self._validate_configuration()

    @staticmethod
    def _validate_email_address(address: str, field_name: str) -> str:
        """Validate an email address and prevent header injection."""
        if not isinstance(address, str) or not address.strip():
            raise ValueError(f"{field_name} cannot be empty.")

        address = address.strip()
        if len(address) > MAX_EMAIL_LENGTH:
            raise ValueError(f"{field_name} is too long.")
        if "\r" in address or "\n" in address:
            raise ValueError(f"{field_name} contains invalid characters.")
        if not EMAIL_PATTERN.fullmatch(address):
            raise ValueError(f"{field_name} is not a valid email address.")
        return address

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
        max_length: int,
        *,
        allow_empty: bool = False,
    ) -> str:
        """Validate text fields and enforce their length limits."""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string.")
        normalized = value.strip()
        if not normalized and not allow_empty:
            raise ValueError(f"{field_name} cannot be empty.")
        if len(normalized) > max_length:
            raise ValueError(
                f"{field_name} exceeds the maximum length of {max_length}."
            )
        return normalized

    def _validate_configuration(self) -> tuple[str, int, str, str]:
        """Return validated SMTP configuration or raise EmailServiceError."""
        host = self.smtp_host
        port = self.smtp_port
        username = self.username
        password = self.password

        if not isinstance(host, str) or not host.strip():
            raise EmailServiceError("SMTP_HOST is missing or invalid.")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise EmailServiceError("SMTP_PORT is missing or invalid.")
        if not isinstance(username, str) or not username.strip():
            raise EmailServiceError("SMTP_USERNAME is missing or invalid.")
        if not isinstance(password, str) or not password:
            raise EmailServiceError("SMTP_PASSWORD is missing or invalid.")

        return host.strip(), port, username.strip(), password

    def _open_smtp_connection(self, host: str, port: int) -> smtplib.SMTP:
        """Open SMTP connection and negotiate STARTTLS."""
        server = smtplib.SMTP(host, port, timeout=self.timeout)
        try:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            return server
        except Exception:
            try:
                server.close()
            except Exception:
                pass
            raise

    @staticmethod
    def _close_smtp_connection(server: smtplib.SMTP) -> None:
        """Close an SMTP connection without hiding a completed send."""
        try:
            server.quit()
        except Exception:
            try:
                server.close()
            except Exception:
                pass

    def send_email(
        self,
        recipient: str | None = None,
        subject: str = "",
        body: str = "",
        sender: str | None = None,
        *,
        to_email: str | None = None,
    ) -> bool:
        """Send an email using recipient or the legacy to_email keyword."""
        if recipient is not None and to_email is not None:
            raise ValueError("Provide recipient or to_email, not both.")

        recipient_value = to_email if to_email is not None else recipient
        if recipient_value is None:
            raise ValueError("Recipient email address is required.")
        recipient_value = self._validate_email_address(
            recipient_value, "Recipient email"
        )

        subject = self._validate_text(
            subject, "Email subject", MAX_SUBJECT_LENGTH
        )
        body = self._validate_text(
            body, "Email body", MAX_BODY_LENGTH, allow_empty=True
        )
        if "\r" in subject or "\n" in subject:
            raise ValueError("Email subject contains invalid characters.")

        host, port, username, password = self._validate_configuration()
        sender_address = self._validate_email_address(
            username if sender is None else sender, "Sender email"
        )

        message = EmailMessage()
        message["From"] = sender_address
        message["To"] = recipient_value
        message["Subject"] = subject
        message.set_content(body)

        server: smtplib.SMTP | None = None
        try:
            with self._lock:
                server = self._open_smtp_connection(host, port)
                server.login(username, password)
                server.send_message(message)
            logger.info("Email sent successfully.")
            return True
        except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
            logger.warning(
                "Email delivery failed; error type: %s",
                type(exc).__name__,
            )
            raise EmailServiceError(
                "Unable to send email. Check SMTP configuration and server availability."
            ) from exc
        finally:
            if server is not None:
                self._close_smtp_connection(server)

    def send_customer_notification(
        self,
        recipient: str | None = None,
        subject: str = "",
        message: str = "",
        *,
        customer_email: str | None = None,
    ) -> bool:
        """Send a customer notification using either recipient keyword."""
        if recipient is not None and customer_email is not None:
            raise ValueError("Provide recipient or customer_email, not both.")
        recipient_value = (
            customer_email if customer_email is not None else recipient
        )
        return self.send_email(
            recipient=recipient_value,
            subject=subject,
            body=message,
        )

    def send_ticket_notification(
        self,
        recipient: str,
        ticket_id: str,
        issue_summary: str,
    ) -> bool:
        """Notify a customer that their support ticket was created."""
        ticket_id = self._validate_text(
            ticket_id, "Ticket ID", MAX_IDENTIFIER_LENGTH
        )
        issue_summary = self._validate_text(
            issue_summary, "Issue summary", 5_000
        )
        return self.send_email(
            recipient=recipient,
            subject=f"Support Ticket Created - {ticket_id}",
            body=(
                "Hello,\n\nYour support request has been received successfully.\n\n"
                f"Ticket ID: {ticket_id}\n\nIssue summary:\n{issue_summary}\n\n"
                "Our support team will review your request and assist you as soon as possible.\n\n"
                "Thank you,\nCustomer Support Team"
            ),
        )

    def send_escalation_notification(
        self, recipient: str, ticket_id: str
    ) -> bool:
        """Notify a customer that their support ticket was escalated."""
        ticket_id = self._validate_text(
            ticket_id, "Ticket ID", MAX_IDENTIFIER_LENGTH
        )
        return self.send_email(
            recipient=recipient,
            subject=f"Support Request Escalated - {ticket_id}",
            body=(
                "Hello,\n\nYour support request has been escalated to our "
                "customer support team.\n\n"
                f"Ticket ID: {ticket_id}\n\n"
                "A support representative will review your request.\n\n"
                "Thank you,\nCustomer Support Team"
            ),
        )

    def health_check(self) -> bool:
        """Return whether SMTP can connect and authenticate."""
        server: smtplib.SMTP | None = None
        try:
            host, port, username, password = self._validate_configuration()
            with self._lock:
                server = self._open_smtp_connection(host, port)
                server.login(username, password)
            return True
        except (
            smtplib.SMTPException,
            EmailServiceError,
            OSError,
            ssl.SSLError,
        ) as exc:
            logger.warning(
                "Email service health check failed; error type: %s",
                type(exc).__name__,
            )
            return False
        finally:
            if server is not None:
                self._close_smtp_connection(server)


_email_service: EmailService | None = None
_email_service_lock = Lock()


def get_email_service() -> EmailService:
    """Return the shared EmailService instance, initialized once."""
    global _email_service
    if _email_service is None:
        with _email_service_lock:
            if _email_service is None:
                _email_service = EmailService()
    return _email_service


def send_email(
    recipient: str | None = None,
    subject: str = "",
    body: str = "",
    *,
    to_email: str | None = None,
) -> bool:
    """Send email through the shared service instance."""
    return get_email_service().send_email(
        recipient=recipient,
        subject=subject,
        body=body,
        to_email=to_email,
    )
