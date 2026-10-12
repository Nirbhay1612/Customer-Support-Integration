"""
Ticket API integration for the AI Customer Support Integration System.

Uses requests' module-level methods so callers and unit tests can mock
HTTP operations directly. Raises explicit exceptions for not-found and
general API failures.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 10.0


class TicketServiceError(RuntimeError):
    """Base exception for ticket-service failures."""


class TicketNotFoundError(TicketServiceError):
    """Raised when the requested ticket does not exist."""


class TicketService:
    """Client for customer-support ticket endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        configured_url = getattr(settings, "TICKET_API_URL", None)
        configured_key = getattr(settings, "TICKET_API_KEY", None)

        self.base_url = (base_url if base_url is not None else configured_url or "").rstrip("/")
        self.api_key = api_key if api_key is not None else configured_key or ""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        self.timeout = float(timeout)

    def _get_headers(self) -> dict[str, str]:
        """Return headers for authenticated JSON API requests."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise TicketServiceError("Ticket API base URL is not configured.")
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _decode_response(response: requests.Response) -> Any:
        """Decode JSON, producing a useful service error for invalid payloads."""
        try:
            return response.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as exc:
            raise TicketServiceError("Ticket API returned invalid JSON.") from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = self._url(path)
        request_method = getattr(requests, method.lower(), None)
        if request_method is None:
            raise ValueError(f"Unsupported HTTP method: {method}")

        try:
            response = request_method(
                url,
                headers=self._get_headers(),
                json=json,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Ticket API request failed (%s).", type(exc).__name__)
            raise TicketServiceError("Unable to complete the ticket API request.") from exc

        if response.status_code == 404:
            raise TicketNotFoundError("Ticket was not found.")
        if response.status_code >= 500:
            raise TicketServiceError("Ticket API returned a server error.")
        if response.status_code >= 400:
            raise TicketServiceError(
                f"Ticket API request failed with HTTP {response.status_code}."
            )

        return self._decode_response(response)

    def create_ticket(
        self,
        customer_id: str,
        subject: str,
        description: str,
        priority: str = "normal",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Create a ticket for a customer."""
        required = {
            "customer_id": customer_id,
            "subject": subject,
            "description": description,
        }
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty.")

        payload: dict[str, Any] = {
            "customer_id": customer_id.strip(),
            "subject": subject.strip(),
            "description": description.strip(),
            "priority": priority.strip() if isinstance(priority, str) else priority,
        }
        payload.update(extra_fields)
        result = self._request("post", "/tickets", json=payload)
        if not isinstance(result, dict):
            raise TicketServiceError("Ticket API returned an unexpected response.")
        return result

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Retrieve a ticket by ID."""
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise ValueError("ticket_id cannot be empty.")
        result = self._request("get", f"/tickets/{quote(ticket_id.strip(), safe='')}")
        if not isinstance(result, dict):
            raise TicketServiceError("Ticket API returned an unexpected response.")
        return result

    def update_ticket(self, ticket_id: str, **updates: Any) -> dict[str, Any]:
        """Update fields on an existing ticket."""
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise ValueError("ticket_id cannot be empty.")
        if not updates:
            raise ValueError("At least one ticket field must be provided.")
        result = self._request(
            "patch",
            f"/tickets/{quote(ticket_id.strip(), safe='')}",
            json=updates,
        )
        if not isinstance(result, dict):
            raise TicketServiceError("Ticket API returned an unexpected response.")
        return result

    def get_customer_tickets(self, customer_id: str) -> list[dict[str, Any]]:
        """Return tickets belonging to a customer."""
        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError("customer_id cannot be empty.")
        result = self._request(
            "get",
            "/tickets",
            params={"customer_id": customer_id.strip()},
        )
        if isinstance(result, dict):
            # Support APIs that wrap collection results in a `tickets` key.
            result = result.get("tickets", [])
        if not isinstance(result, list):
            raise TicketServiceError("Ticket API returned an unexpected response.")
        if not all(isinstance(ticket, dict) for ticket in result):
            raise TicketServiceError("Ticket API returned an invalid ticket list.")
        return result

    def health_check(self) -> bool:
        """Check whether the ticket API is reachable."""
        try:
            self._request("get", "/health")
            return True
        except TicketServiceError:
            return False
