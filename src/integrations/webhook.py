"""
Webhook utilities for the AI Customer Support Integration System.

Validates HMAC-SHA256 signatures, parses JSON payloads, and normalizes
events received from external business systems.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from threading import Lock
from typing import Any, Final

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_WEBHOOK_PAYLOAD_SIZE: Final[int] = 1_000_000
MAX_EVENT_TYPE_LENGTH: Final[int] = 200
MAX_EVENT_ID_LENGTH: Final[int] = 200
SIGNATURE_PREFIX: Final[str] = "sha256="


class WebhookError(ValueError):
    """Base exception for webhook-related errors."""


class InvalidWebhookSignature(WebhookError):
    """Raised when webhook signature verification fails."""


class WebhookEvent:
    """Normalized representation of an incoming webhook event."""

    def __init__(
        self,
        event_type: str,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> None:
        self.event_type = event_type
        self.payload = payload
        self.event_id = event_id

    @property
    def data(self) -> dict[str, Any]:
        """Return the event's data object, or an empty object if malformed."""
        value = self.payload.get("data", self.payload)
        return value if isinstance(value, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the normalized event."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "payload": self.payload,
            "data": self.data,
        }


class WebhookService:
    """Validate signatures and parse incoming webhook events."""

    def __init__(self, webhook_secret: str | None = None) -> None:
        configured_secret = (
            webhook_secret
            if webhook_secret is not None
            else getattr(settings, "WEBHOOK_SECRET", None)
        )
        if configured_secret is not None and not isinstance(configured_secret, str):
            raise WebhookError("WEBHOOK_SECRET must be a string.")

        self.webhook_secret = configured_secret
        # Backwards-compatible alias used by existing callers/tests.
        self.secret = configured_secret

    @staticmethod
    def _get_payload_bytes(payload: bytes | str) -> bytes:
        """Normalize payload to bytes and enforce basic size limits."""
        if isinstance(payload, str):
            try:
                payload_bytes = payload.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise WebhookError("Webhook payload encoding is invalid.") from exc
        elif isinstance(payload, bytes):
            payload_bytes = payload
        else:
            raise WebhookError("Webhook payload must be bytes or a string.")

        if not payload_bytes:
            raise WebhookError("Webhook payload is empty.")
        if len(payload_bytes) > MAX_WEBHOOK_PAYLOAD_SIZE:
            raise WebhookError("Webhook payload exceeds the maximum allowed size.")
        return payload_bytes

    def generate_signature(self, payload: bytes | str) -> str:
        """Generate a signature for trusted integrations and tests."""
        secret = self.webhook_secret
        if not isinstance(secret, str) or not secret:
            raise WebhookError("WEBHOOK_SECRET is not configured.")

        payload_bytes = self._get_payload_bytes(payload)
        digest = hmac.new(
            secret.encode("utf-8"), payload_bytes, hashlib.sha256
        ).hexdigest()
        return f"{SIGNATURE_PREFIX}{digest}"

    def validate_signature(self, payload: bytes | str, signature: str) -> bool:
        """Return True only when the supplied HMAC-SHA256 signature is valid."""
        secret = self.webhook_secret
        if not isinstance(secret, str) or not secret:
            raise WebhookError("WEBHOOK_SECRET is not configured.")
        if not isinstance(signature, str) or not signature:
            return False

        payload_bytes = self._get_payload_bytes(payload)
        supplied = signature.strip()
        if not supplied.startswith(SIGNATURE_PREFIX):
            return False

        digest = supplied[len(SIGNATURE_PREFIX):]
        if len(digest) != 64:
            return False
        try:
            bytes.fromhex(digest)
        except ValueError:
            return False

        expected = hmac.new(
            secret.encode("utf-8"), payload_bytes, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, digest.lower())

    def verify_signature(self, payload: bytes | str, signature: str) -> bool:
        """
        Verify a signature. Return True when valid and raise
        InvalidWebhookSignature when invalid.
        """
        if not self.validate_signature(payload, signature):
            logger.warning("Webhook signature validation failed.")
            raise InvalidWebhookSignature("Webhook signature validation failed.")
        return True

    @staticmethod
    def parse_payload(payload: bytes | str) -> dict[str, Any]:
        """Parse a JSON object without performing signature verification."""
        payload_bytes = WebhookService._get_payload_bytes(payload)
        try:
            decoded = payload_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WebhookError("Webhook payload must use UTF-8 encoding.") from exc

        try:
            data: Any = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise WebhookError("Webhook payload is not valid JSON.") from exc

        if not isinstance(data, dict):
            raise WebhookError("Webhook payload must be a JSON object.")
        return data

    def create_event(self, payload: dict[str, Any]) -> WebhookEvent:
        """Normalize a parsed payload into a WebhookEvent."""
        if not isinstance(payload, dict):
            raise WebhookError("Webhook payload must be a JSON object.")

        raw_type = payload.get("event") or payload.get("event_type") or payload.get("type")
        if not isinstance(raw_type, str):
            raise WebhookError("Webhook event type is missing or invalid.")
        event_type = raw_type.strip()
        if not event_type or len(event_type) > MAX_EVENT_TYPE_LENGTH:
            raise WebhookError("Webhook event type is missing or exceeds the maximum length.")

        raw_id = payload.get("event_id") or payload.get("id")
        if raw_id is None:
            event_id = None
        elif isinstance(raw_id, (dict, list, bool)) or not isinstance(raw_id, (str, int, float)):
            raise WebhookError("Webhook event ID must be a string or number.")
        else:
            event_id = str(raw_id).strip() or None
            if event_id is not None and len(event_id) > MAX_EVENT_ID_LENGTH:
                raise WebhookError("Webhook event ID exceeds the maximum length.")

        return WebhookEvent(
            event_type=event_type,
            event_id=event_id,
            payload=payload,
        )

    def parse_webhook(self, payload: bytes | str, signature: str) -> WebhookEvent:
        """Verify, parse, and normalize a webhook (compatibility API)."""
        payload_bytes = self._get_payload_bytes(payload)
        self.verify_signature(payload_bytes, signature)
        parsed = self.parse_payload(payload_bytes)
        event = self.create_event(parsed)
        logger.info(
            "Webhook verified: type=%s, id_present=%s",
            event.event_type,
            event.event_id is not None,
        )
        return event

    def process_webhook(self, payload: bytes | str, signature: str) -> WebhookEvent:
        """Public processing method; alias to parse_webhook."""
        return self.parse_webhook(payload, signature)


_webhook_service: WebhookService | None = None
_webhook_service_lock = Lock()


def get_webhook_service() -> WebhookService:
    """Return the shared, thread-safe webhook service."""
    global _webhook_service
    if _webhook_service is None:
        with _webhook_service_lock:
            if _webhook_service is None:
                _webhook_service = WebhookService()
    return _webhook_service


def process_webhook(payload: bytes | str, signature: str) -> WebhookEvent:
    """Verify and normalize a webhook using the shared service."""
    return get_webhook_service().process_webhook(payload=payload, signature=signature)
