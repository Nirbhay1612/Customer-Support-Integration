"""

Customer API integration.

Provides validated methods for retrieving and searching customer

information through an external customer-management API.

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

MAX_CUSTOMER_ID_LENGTH = 200

MAX_SEARCH_QUERY_LENGTH = 500

# ---------------------------------------------------------

# Exceptions

# ---------------------------------------------------------

class CustomerAPIError(RuntimeError):

    """Base exception for customer API errors."""

class CustomerNotFoundError(CustomerAPIError):

    """Raised when a customer cannot be found."""

# ---------------------------------------------------------

# Customer API Client

# ---------------------------------------------------------

class CustomerAPIClient:

    """Client for interacting with an external customer API."""

    def __init__(

        self,

        base_url: str | None = None,

        api_key: str | None = None,

        timeout: float = 10,

    ) -> None:

        resolved_url = (

            base_url

            if base_url is not None

            else getattr(settings, "CUSTOMER_API_URL", None)

        )

        if not isinstance(resolved_url, str) or not resolved_url.strip():

            raise ValueError("CUSTOMER_API_URL is not configured.")

        resolved_url = resolved_url.strip().rstrip("/")

        try:

            parsed_url = urlsplit(resolved_url)

            valid_port = parsed_url.port is None or (

                1 <= parsed_url.port <= 65535

            )

        except ValueError as exc:

            raise ValueError(

                "Customer API base URL is invalid."

            ) from exc

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

                "Customer API base URL must be a valid HTTP(S) URL."

            )

        if (

            isinstance(timeout, bool)

            or not isinstance(timeout, (int, float))

            or not isfinite(timeout)

            or timeout <= 0

        ):

            raise ValueError("Timeout must be a positive number.")

        resolved_key = (

            api_key

            if api_key is not None

            else getattr(settings, "CUSTOMER_API_KEY", None)

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

        """Build HTTP request headers without logging credentials."""

        headers = {

            "Accept": "application/json",

            "Content-Type": "application/json",

        }

        if self.api_key:

            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    @staticmethod

    def _validate_customer_id(customer_id: str) -> str:

        """Validate and normalize a customer ID."""

        if not isinstance(customer_id, str):

            raise TypeError("customer_id must be a string.")

        customer_id = customer_id.strip()

        if not customer_id:

            raise ValueError(

                "customer_id must be a non-empty string."

            )

        if len(customer_id) > MAX_CUSTOMER_ID_LENGTH:

            raise ValueError(

                "customer_id exceeds the maximum permitted length."

            )

        return customer_id

    @staticmethod

    def _read_json(response: requests.Response) -> Any:

        """Parse a JSON response and normalize decoding errors."""

        try:

            return response.json()

        except ValueError as exc:

            raise CustomerAPIError(

                "Customer API returned invalid JSON."

            ) from exc

    def _request(

        self,

        url: str,

        *,

        params: dict[str, str] | None = None,

    ) -> requests.Response:

        """Perform a serialized GET request using the shared session."""

        with self._request_lock:

            if self._closed:

                raise CustomerAPIError(

                    "Customer API client is closed."

                )

            return requests.get(

                url,

                headers=self._get_headers(),

                params=params,

                timeout=self.timeout,

            )

    def _request_json(

        self,

        url: str,

        *,

        params: dict[str, str] | None = None,

        not_found_message: str = "Requested customer was not found.",

    ) -> Any:

        """Perform a GET request with consistent error handling."""

        try:

            response = self._request(url, params=params)

            if response.status_code == 404:

                raise CustomerNotFoundError(not_found_message)

            response.raise_for_status()

            if response.status_code == 204 or not response.content:

                return {}

            return self._read_json(response)

        except CustomerNotFoundError:

            raise

        except CustomerAPIError:

            raise

        except requests.Timeout as exc:

            logger.warning("Customer API request timed out.")

            raise CustomerAPIError(

                "Customer API request timed out."

            ) from exc

        except requests.HTTPError as exc:

            status_code = (

                exc.response.status_code

                if exc.response is not None

                else None

            )

            logger.warning(

                "Customer API returned an HTTP error; status=%s.",

                status_code if status_code is not None else "unknown",

            )

            raise CustomerAPIError(

                "Customer API returned an unsuccessful HTTP response."

            ) from exc

        except requests.RequestException as exc:

            logger.warning(

                "Customer API request failed; error_type=%s.",

                type(exc).__name__,

            )

            raise CustomerAPIError(

                "Unable to complete the customer API request."

            ) from exc

    # -----------------------------------------------------

    # Get Customer

    # -----------------------------------------------------

    def get_customer(self, customer_id: str) -> dict[str, Any]:

        """Retrieve a customer by ID."""

        customer_id = self._validate_customer_id(customer_id)

        encoded_id = quote(customer_id, safe="")

        url = f"{self.base_url}/customers/{encoded_id}"

        data = self._request_json(

            url,

            not_found_message="Customer was not found.",

        )

        if not isinstance(data, dict):

            raise CustomerAPIError(

                "Customer API returned an unexpected response."

            )

        logger.info("Customer record retrieved successfully.")

        return data

    # -----------------------------------------------------

    # Search Customers

    # -----------------------------------------------------

    def search_customers(self, query: str) -> list[dict[str, Any]]:

        """Search customers using the external API."""

        if not isinstance(query, str):

            raise TypeError("Search query must be a string.")

        query = query.strip()

        if not query:

            return []

        if len(query) > MAX_SEARCH_QUERY_LENGTH:

            raise ValueError(

                "Search query exceeds the maximum permitted length."

            )

        data = self._request_json(

            f"{self.base_url}/customers",

            params={"search": query},

        )

        if isinstance(data, list):

            customers = data

        elif isinstance(data, dict):

            customers = data.get(

                "customers",

                data.get("results", []),

            )

        else:

            customers = None

        if not isinstance(customers, list):

            raise CustomerAPIError(

                "Customer API returned an invalid search response."

            )

        if not all(isinstance(item, dict) for item in customers):

            raise CustomerAPIError(

                "Customer API returned invalid customer entries."

            )

        logger.info(

            "Customer search completed; results=%d.",

            len(customers),

        )

        return customers

    # -----------------------------------------------------

    # Health Check

    # -----------------------------------------------------

    def health_check(self) -> bool:

        """

        Check basic endpoint reachability.

        This does not guarantee authentication or API operations work.

        HTTP 4xx responses may still indicate that the server is reachable.

        """

        try:

            response = self._request(self.base_url)

            return response.status_code < 500

        except (requests.RequestException, CustomerAPIError):

            logger.warning("Customer API health check failed.")

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

    def __enter__(self) -> CustomerAPIClient:

        """Support use as a context manager."""

        if self._closed:

            raise CustomerAPIError("Customer API client is closed.")

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

_customer_api_client: CustomerAPIClient | None = None

_customer_api_lock = Lock()

def get_customer_api() -> CustomerAPIClient:

    """Return a lazily initialized, thread-safe customer API client."""

    global _customer_api_client

    if _customer_api_client is None:

        with _customer_api_lock:

            if _customer_api_client is None:

                _customer_api_client = CustomerAPIClient()

    return _customer_api_client

# ---------------------------------------------------------

# Convenience Functions

# ---------------------------------------------------------

def get_customer(customer_id: str) -> dict[str, Any]:

    """Retrieve customer information."""

    return get_customer_api().get_customer(customer_id)

def search_customers(query: str) -> list[dict[str, Any]]:

    """Search for customers."""

    return get_customer_api().search_customers(query)
