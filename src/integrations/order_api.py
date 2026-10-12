"""
Order API integration.

Provides validated methods for order lookup, tracking, customer order
search, and explicitly requested order actions.
"""

from __future__ import annotations

from math import isfinite
from threading import Lock
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from config import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)

MAX_ID_LENGTH = 200
MAX_REASON_LENGTH = 500
MAX_SEARCH_RESULTS = 10_000


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------


class OrderAPIError(RuntimeError):
    """Raised when an order API operation fails."""


class OrderNotFoundError(OrderAPIError):
    """Raised when an order resource is not found."""


class OrderActionError(OrderAPIError):
    """Raised when an order action is rejected or cannot be completed."""


# ---------------------------------------------------------
# Order API Client
# ---------------------------------------------------------


class OrderAPIClient:
    """Client for order lookup, tracking, and explicit order actions."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 10,
    ) -> None:
        resolved_url = (
            base_url
            if base_url is not None
            else getattr(settings, "ORDER_API_URL", None)
        )

        if not isinstance(resolved_url, str) or not resolved_url.strip():
            raise ValueError("ORDER_API_URL must be configured.")

        resolved_url = resolved_url.strip().rstrip("/")

        try:
            parsed_url = urlsplit(resolved_url)
            port = parsed_url.port
            valid_port = port is None or 1 <= port <= 65535
        except ValueError as exc:
            raise ValueError("Order API base URL is invalid.") from exc

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
            or not valid_port
        ):
            raise ValueError(
                "Order API base URL must be a valid HTTP(S) URL."
            )

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Timeout must be a positive finite number.")

        resolved_key = (
            api_key
            if api_key is not None
            else getattr(settings, "ORDER_API_KEY", None)
        )

        if resolved_key is not None and not isinstance(resolved_key, str):
            raise ValueError("API key must be a string or None.")

        self.base_url = resolved_url
        self.api_key = resolved_key.strip() if resolved_key else None
        self.timeout = float(timeout)

        self._session = requests.Session()
        self._request_lock = Lock()
        self._closed = False

    # -----------------------------------------------------
    # Headers and Validation
    # -----------------------------------------------------

    def _get_headers(self) -> dict[str, str]:
        """Build request headers without logging credentials."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    @staticmethod
    def _validate_id(value: str, field_name: str) -> str:
        """Validate and normalize an identifier."""
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")

        value = value.strip()

        if not value:
            raise ValueError(f"{field_name} must be a non-empty string.")

        if len(value) > MAX_ID_LENGTH:
            raise ValueError(
                f"{field_name} exceeds the maximum permitted length."
            )

        return value

    @staticmethod
    def _read_json(response: requests.Response) -> Any:
        """Parse JSON and normalize decoding errors."""
        try:
            return response.json()
        except ValueError as exc:
            raise OrderAPIError(
                "Order API returned an invalid JSON response."
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> requests.Response:
        """Send a request to a relative API path."""
        if not isinstance(method, str) or not method.strip():
            raise ValueError("HTTP method cannot be empty.")

        if not isinstance(path, str) or not path.strip():
            raise ValueError("Request path cannot be empty.")

        path = path.strip()

        # Only internally constructed relative paths are expected here.
        parsed_path = urlsplit(path)
        if (
            path.startswith(("//", "\\"))
            or parsed_path.scheme
            or parsed_path.netloc
            or parsed_path.query
            or parsed_path.fragment
            or "\\" in path
            or any(part == ".." for part in path.split("/"))
        ):
            raise ValueError("Request path must be a safe relative path.")

        url = f"{self.base_url}/{path.lstrip('/')}"

        with self._request_lock:
            if self._closed:
                raise OrderAPIError("Order API client is closed.")

            request_method = method.strip().upper()
            request_kwargs = {
                "headers": self._get_headers(),
                "timeout": self.timeout,
                "allow_redirects": False,
            }
            if request_method in {"POST", "PATCH", "PUT"}:
                request_kwargs["json"] = payload

            # Use module-level requests methods so callers and tests can mock
            # the individual HTTP verbs consistently.
            if request_method == "GET":
                return requests.get(url, **request_kwargs)
            if request_method == "POST":
                return requests.post(url, **request_kwargs)
            if request_method == "PATCH":
                return requests.patch(url, **request_kwargs)
            if request_method == "PUT":
                return requests.put(url, **request_kwargs)
            if request_method == "DELETE":
                return requests.delete(url, **request_kwargs)

            request_method = method.strip().upper()
            request_kwargs: dict[str, Any] = {
                "headers": self._get_headers(),
                "timeout": self.timeout,
                "allow_redirects": False,
            }

            if request_method in {"POST", "PUT", "PATCH"}:
                request_kwargs["json"] = payload

            # Use the public requests helpers so calls are easy to mock in tests
            # and consistent with the other integration clients.
            if request_method == "GET":
                return requests.get(url, **request_kwargs)
            if request_method == "POST":
                return requests.post(url, **request_kwargs)
            if request_method == "PUT":
                return requests.put(url, **request_kwargs)
            if request_method == "PATCH":
                return requests.patch(url, **request_kwargs)
            if request_method == "DELETE":
                return requests.delete(url, **request_kwargs)

            return self._session.request(
                method=request_method,
                url=url,
                **request_kwargs,
            )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        not_found_message: str = "Requested order resource was not found.",
        action: bool = False,
    ) -> Any:
        """Send an HTTP request and normalize common API errors."""
        try:
            response = self._request(method, path, payload=payload)

            if response.status_code == 404:
                raise OrderNotFoundError(not_found_message)

            if action and response.status_code in {400, 409, 422}:
                raise OrderActionError(
                    "The order action was rejected by the API."
                )

            response.raise_for_status()

            if response.status_code == 204 or not response.content:
                return {}

            return self._read_json(response)

        except (OrderNotFoundError, OrderActionError, OrderAPIError):
            raise

        except requests.Timeout as exc:
            logger.warning("Order API request timed out.")
            error_class = OrderActionError if action else OrderAPIError
            raise error_class("Order API request timed out.") from exc

        except requests.HTTPError as exc:
            status = (
                exc.response.status_code
                if exc.response is not None
                else None
            )
            logger.warning(
                "Order API returned an HTTP error; status=%s.",
                status if status is not None else "unknown",
            )
            error_class = OrderActionError if action else OrderAPIError
            raise error_class(
                "Order API returned an unsuccessful HTTP response."
            ) from exc

        except requests.RequestException as exc:
            logger.warning(
                "Order API request failed; error_type=%s.",
                type(exc).__name__,
            )
            error_class = OrderActionError if action else OrderAPIError
            raise error_class("Order API request failed.") from exc

    # -----------------------------------------------------
    # Get Order
    # -----------------------------------------------------

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Retrieve an order by ID."""
        order_id = self._validate_id(order_id, "Order ID")
        encoded_id = quote(order_id, safe="")

        data = self._request_json(
            "GET",
            f"orders/{encoded_id}",
            not_found_message="Order was not found.",
        )

        if not isinstance(data, dict):
            raise OrderAPIError(
                "Order API returned an unexpected response."
            )

        return data

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Retrieve an order and validate its current status field."""
        order = self.get_order(order_id)
        status = order.get("status")

        if not isinstance(status, str) or not status.strip():
            raise OrderAPIError(
                "Order response does not contain a valid status."
            )

        # Return the order payload, preserving status and other order fields.
        order["status"] = status.strip()
        return order

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------

    def get_tracking_info(self, order_id: str) -> dict[str, Any]:
        """Retrieve tracking information for an order."""
        order_id = self._validate_id(order_id, "Order ID")
        encoded_id = quote(order_id, safe="")

        data = self._request_json(
            "GET",
            f"orders/{encoded_id}/tracking",
            not_found_message="Tracking information was not found.",
        )

        if not isinstance(data, dict):
            raise OrderAPIError(
                "Tracking API returned an unexpected response."
            )

        return data

    # -----------------------------------------------------
    # Customer Orders
    # -----------------------------------------------------

    def get_customer_orders(
        self,
        customer_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve orders associated with a customer."""
        customer_id = self._validate_id(customer_id, "Customer ID")
        encoded_id = quote(customer_id, safe="")

        data = self._request_json(
            "GET",
            f"customers/{encoded_id}/orders",
            not_found_message="Customer orders were not found.",
        )

        if isinstance(data, list):
            orders = data
        elif isinstance(data, dict):
            orders = data.get("orders", data.get("results", []))
        else:
            orders = None

        if not isinstance(orders, list):
            raise OrderAPIError(
                "Customer orders API returned an unexpected response."
            )

        if len(orders) > MAX_SEARCH_RESULTS:
            raise OrderAPIError(
                "Customer orders API returned too many entries."
            )

        if not all(isinstance(order, dict) for order in orders):
            raise OrderAPIError(
                "Customer orders API returned invalid order entries."
            )

        return orders

    # -----------------------------------------------------
    # Cancel Order
    # -----------------------------------------------------

    def cancel_order(
        self,
        order_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Request order cancellation.

        Call only after the application has authenticated the customer,
        verified authorization, and obtained explicit confirmation.
        """
        order_id = self._validate_id(order_id, "Order ID")
        encoded_id = quote(order_id, safe="")
        payload: dict[str, Any] = {}

        if reason is not None:
            if not isinstance(reason, str):
                raise TypeError("Cancellation reason must be a string.")

            reason = reason.strip()

            if not reason:
                raise ValueError("Cancellation reason cannot be empty.")

            if len(reason) > MAX_REASON_LENGTH:
                raise ValueError(
                    "Cancellation reason cannot exceed 500 characters."
                )

            payload["reason"] = reason

        data = self._request_json(
            "POST",
            f"orders/{encoded_id}/cancel",
            payload=payload,
            not_found_message="Order was not found.",
            action=True,
        )

        if not isinstance(data, dict):
            raise OrderAPIError(
                "Cancellation API returned an unexpected response."
            )

        logger.info("Order cancellation request completed.")
        return data

    # -----------------------------------------------------
    # Health Check
    # -----------------------------------------------------

    def health_check(self) -> bool:
        """
        Check basic endpoint reachability.

        A response below HTTP 500 indicates reachability, not necessarily
        successful authentication or successful order operations.
        """
        try:
            with self._request_lock:
                if self._closed:
                    return False

                response = requests.get(
                    self.base_url,
                    headers=self._get_headers(),
                    timeout=self.timeout,
                    allow_redirects=False,
                )

            return response.status_code < 500

        except requests.RequestException:
            logger.warning("Order API health check failed.")
            return False

    # -----------------------------------------------------
    # Resource Cleanup
    # -----------------------------------------------------

    def close(self) -> None:
        """Close the HTTP session and release its resources."""
        with self._request_lock:
            if not self._closed:
                self._session.close()
                self._closed = True

    def __enter__(self) -> OrderAPIClient:
        """Support use as a context manager."""
        if self._closed:
            raise OrderAPIError("Order API client is closed.")
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Close the client when leaving a context manager."""
        self.close()


# ---------------------------------------------------------
# Global Client
# ---------------------------------------------------------

_order_api_client: OrderAPIClient | None = None
_order_api_lock = Lock()


def get_order_api() -> OrderAPIClient:
    """Return the shared order API client, initializing it once."""
    global _order_api_client

    if _order_api_client is None:
        with _order_api_lock:
            if _order_api_client is None:
                _order_api_client = OrderAPIClient()

    return _order_api_client


# ---------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------


def get_order(order_id: str) -> dict[str, Any]:
    """Retrieve order information."""
    return get_order_api().get_order(order_id)


def get_order_status(order_id: str) -> dict[str, Any]:
    """Retrieve an order's current status."""
    return get_order_api().get_order_status(order_id)


def get_tracking_info(order_id: str) -> dict[str, Any]:
    """Retrieve order tracking information."""
    return get_order_api().get_tracking_info(order_id)


def get_customer_orders(customer_id: str) -> list[dict[str, Any]]:
    """Retrieve a customer's orders."""
    return get_order_api().get_customer_orders(customer_id)


def cancel_order(
    order_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Explicitly request order cancellation."""
    return get_order_api().cancel_order(order_id, reason)
