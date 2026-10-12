"""
Tests for integration services.

Covers:
- Customer API
- Order API
- Email service
- Ticket service
- Webhook validation and parsing
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.integrations.customer_api import (
    CustomerAPIClient,
    CustomerAPIError,
    CustomerNotFoundError,
)
from src.integrations.email_service import (
    EmailService,
    EmailServiceError,
)
from src.integrations.order_api import (
    OrderAPIClient,
    OrderAPIError,
    OrderNotFoundError,
)
from src.integrations.ticket_service import (
    TicketNotFoundError,
    TicketService,
    TicketServiceError,
)
from src.integrations.webhook import (
    InvalidWebhookSignature,
    WebhookService,
)


# ============================================================
# Shared Mock Response
# ============================================================


def mock_response(
    status_code=200,
    json_data=None,
    text="",
):
    """Create a mock requests.Response object."""
    response = MagicMock(spec=requests.Response)

    response.status_code = status_code
    response.text = text

    response.json.return_value = (
        json_data if json_data is not None else {}
    )

    response.raise_for_status.side_effect = (
        None
        if status_code < 400
        else requests.HTTPError(
            f"HTTP {status_code}"
        )
    )

    return response


# ============================================================
# Customer API Tests
# ============================================================


class TestCustomerAPI:
    """Tests for CustomerAPIClient."""

    @pytest.fixture
    def client(self):
        """Create a customer API client."""
        return CustomerAPIClient(
            base_url="https://example.com/api",
            api_key="test-api-key",
        )

    def test_client_initialization(self, client):
        """Customer API client should initialize correctly."""
        assert client.base_url == (
            "https://example.com/api"
        )
        assert client.api_key == "test-api-key"

    def test_headers_include_api_key(self, client):
        """API key should be included in request headers."""
        headers = client._get_headers()

        assert isinstance(headers, dict)
        assert (
            headers.get("Authorization")
            == "Bearer test-api-key"
        )

    @patch("src.integrations.customer_api.requests.get")
    def test_get_customer(self, mock_get, client):
        """Customer should be retrieved successfully."""
        mock_get.return_value = mock_response(
            json_data={
                "id": "CUST-10001",
                "name": "Aarav Sharma",
                "email": "aarav.sharma@example.com",
            }
        )

        customer = client.get_customer("CUST-10001")

        assert customer["id"] == "CUST-10001"
        assert customer["name"] == "Aarav Sharma"

        mock_get.assert_called_once()

    @patch("src.integrations.customer_api.requests.get")
    def test_get_customer_sends_correct_url(
        self,
        mock_get,
        client,
    ):
        """Customer endpoint should use the customer ID."""
        mock_get.return_value = mock_response(
            json_data={
                "id": "CUST-10001"
            }
        )

        client.get_customer("CUST-10001")

        request_url = mock_get.call_args.args[0]

        assert request_url.endswith(
            "/customers/CUST-10001"
        )

    @patch("src.integrations.customer_api.requests.get")
    def test_get_customer_not_found(
        self,
        mock_get,
        client,
    ):
        """404 should raise CustomerNotFoundError."""
        mock_get.return_value = mock_response(
            status_code=404,
            json_data={
                "detail": "Customer not found"
            },
        )

        with pytest.raises(CustomerNotFoundError):
            client.get_customer("CUST-99999")

    @patch("src.integrations.customer_api.requests.get")
    def test_get_customer_api_error(
        self,
        mock_get,
        client,
    ):
        """Server errors should raise CustomerAPIError."""
        mock_get.return_value = mock_response(
            status_code=500,
            json_data={
                "detail": "Internal server error"
            },
        )

        with pytest.raises(CustomerAPIError):
            client.get_customer("CUST-10001")

    @patch("src.integrations.customer_api.requests.get")
    def test_search_customers(
        self,
        mock_get,
        client,
    ):
        """Customer search should return matching customers."""
        mock_get.return_value = mock_response(
            json_data=[
                {
                    "id": "CUST-10001",
                    "name": "Aarav Sharma",
                    "email": "aarav.sharma@example.com",
                }
            ]
        )

        customers = client.search_customers(
            "aarav.sharma@example.com"
        )

        assert isinstance(customers, list)
        assert len(customers) == 1
        assert customers[0]["id"] == "CUST-10001"

    @patch("src.integrations.customer_api.requests.get")
    def test_health_check_success(
        self,
        mock_get,
        client,
    ):
        """Customer API health check should succeed."""
        mock_get.return_value = mock_response(
            json_data={"status": "ok"}
        )

        assert client.health_check() is True

    @patch("src.integrations.customer_api.requests.get")
    def test_health_check_failure(
        self,
        mock_get,
        client,
    ):
        """Customer API health check should fail safely."""
        mock_get.side_effect = requests.RequestException(
            "Connection failed"
        )

        assert client.health_check() is False


# ============================================================
# Order API Tests
# ============================================================


class TestOrderAPI:
    """Tests for OrderAPIClient."""

    @pytest.fixture
    def client(self):
        """Create an order API client."""
        return OrderAPIClient(
            base_url="https://example.com/api",
            api_key="test-api-key",
        )

    def test_client_initialization(self, client):
        """Order API client should initialize correctly."""
        assert client.base_url == (
            "https://example.com/api"
        )
        assert client.api_key == "test-api-key"

    def test_headers_include_api_key(self, client):
        """Order API headers should contain authentication."""
        headers = client._get_headers()

        assert (
            headers.get("Authorization")
            == "Bearer test-api-key"
        )

    @patch("src.integrations.order_api.requests.get")
    def test_get_order(self, mock_get, client):
        """Order should be retrieved successfully."""
        mock_get.return_value = mock_response(
            json_data={
                "id": "ORD-10005",
                "customer_id": "CUST-10001",
                "status": "out_for_delivery",
                "total_amount": 4999.00,
            }
        )

        order = client.get_order("ORD-10005")

        assert order["id"] == "ORD-10005"
        assert order["status"] == "out_for_delivery"

    @patch("src.integrations.order_api.requests.get")
    def test_get_order_not_found(
        self,
        mock_get,
        client,
    ):
        """404 should raise OrderNotFoundError."""
        mock_get.return_value = mock_response(
            status_code=404,
            json_data={
                "detail": "Order not found"
            },
        )

        with pytest.raises(OrderNotFoundError):
            client.get_order("ORD-99999")

    @patch("src.integrations.order_api.requests.get")
    def test_get_order_status(
        self,
        mock_get,
        client,
    ):
        """Order status should be returned."""
        mock_get.return_value = mock_response(
            json_data={
                "id": "ORD-10005",
                "status": "shipped",
            }
        )

        result = client.get_order_status(
            "ORD-10005"
        )

        assert result["status"] == "shipped"

    @patch("src.integrations.order_api.requests.get")
    def test_get_tracking_info(
        self,
        mock_get,
        client,
    ):
        """Tracking information should be returned."""
        mock_get.return_value = mock_response(
            json_data={
                "order_id": "ORD-10005",
                "tracking_number": "TRK-IND-10005",
                "status": "out_for_delivery",
            }
        )

        result = client.get_tracking_info(
            "ORD-10005"
        )

        assert (
            result["tracking_number"]
            == "TRK-IND-10005"
        )

    @patch("src.integrations.order_api.requests.get")
    def test_get_customer_orders(
        self,
        mock_get,
        client,
    ):
        """Customer orders should be returned."""
        mock_get.return_value = mock_response(
            json_data=[
                {
                    "id": "ORD-10001",
                    "customer_id": "CUST-10001",
                },
                {
                    "id": "ORD-10005",
                    "customer_id": "CUST-10001",
                },
            ]
        )

        orders = client.get_customer_orders(
            "CUST-10001"
        )

        assert len(orders) == 2
        assert orders[0]["customer_id"] == "CUST-10001"

    @patch("src.integrations.order_api.requests.post")
    def test_cancel_order(
        self,
        mock_post,
        client,
    ):
        """Order cancellation should call the cancellation endpoint."""
        mock_post.return_value = mock_response(
            json_data={
                "id": "ORD-10003",
                "status": "cancelled",
            }
        )

        result = client.cancel_order(
            "ORD-10003"
        )

        assert result["status"] == "cancelled"

        mock_post.assert_called_once()

    @patch("src.integrations.order_api.requests.get")
    def test_order_api_error(
        self,
        mock_get,
        client,
    ):
        """Server errors should raise OrderAPIError."""
        mock_get.return_value = mock_response(
            status_code=500,
            json_data={
                "detail": "Server error"
            },
        )

        with pytest.raises(OrderAPIError):
            client.get_order("ORD-10005")

    @patch("src.integrations.order_api.requests.get")
    def test_health_check_success(
        self,
        mock_get,
        client,
    ):
        """Order API health check should succeed."""
        mock_get.return_value = mock_response(
            json_data={"status": "ok"}
        )

        assert client.health_check() is True

    @patch("src.integrations.order_api.requests.get")
    def test_health_check_failure(
        self,
        mock_get,
        client,
    ):
        """Order API health check should fail safely."""
        mock_get.side_effect = requests.RequestException(
            "Connection failed"
        )

        assert client.health_check() is False


# ============================================================
# Ticket Service Tests
# ============================================================


class TestTicketService:
    """Tests for TicketService."""

    @pytest.fixture
    def service(self):
        """Create a ticket service."""
        return TicketService(
            base_url="https://example.com/api",
            api_key="test-api-key",
        )

    def test_service_initialization(self, service):
        """Ticket service should initialize correctly."""
        assert service.base_url == (
            "https://example.com/api"
        )
        assert service.api_key == "test-api-key"

    def test_headers_include_api_key(self, service):
        """Ticket service should include authentication."""
        headers = service._get_headers()

        assert (
            headers.get("Authorization")
            == "Bearer test-api-key"
        )

    @patch("src.integrations.ticket_service.requests.post")
    def test_create_ticket(
        self,
        mock_post,
        service,
    ):
        """Ticket should be created successfully."""
        mock_post.return_value = mock_response(
            status_code=201,
            json_data={
                "id": "TKT-10001",
                "customer_id": "CUST-10001",
                "subject": "Payment issue",
                "status": "open",
                "priority": "high",
            },
        )

        ticket = service.create_ticket(
            customer_id="CUST-10001",
            subject="Payment issue",
            description="Customer payment requires review.",
            priority="high",
        )

        assert ticket["id"] == "TKT-10001"
        assert ticket["status"] == "open"

    @patch("src.integrations.ticket_service.requests.get")
    def test_get_ticket(
        self,
        mock_get,
        service,
    ):
        """Ticket should be retrieved successfully."""
        mock_get.return_value = mock_response(
            json_data={
                "id": "TKT-10001",
                "subject": "Payment issue",
                "status": "open",
            }
        )

        ticket = service.get_ticket("TKT-10001")

        assert ticket["id"] == "TKT-10001"
        assert ticket["status"] == "open"

    @patch("src.integrations.ticket_service.requests.get")
    def test_get_ticket_not_found(
        self,
        mock_get,
        service,
    ):
        """Missing ticket should raise TicketNotFoundError."""
        mock_get.return_value = mock_response(
            status_code=404,
            json_data={
                "detail": "Ticket not found"
            },
        )

        with pytest.raises(TicketNotFoundError):
            service.get_ticket("TKT-99999")

    @patch("src.integrations.ticket_service.requests.patch")
    def test_update_ticket(
        self,
        mock_patch,
        service,
    ):
        """Ticket should be updated successfully."""
        mock_patch.return_value = mock_response(
            json_data={
                "id": "TKT-10001",
                "status": "resolved",
            }
        )

        result = service.update_ticket(
            "TKT-10001",
            status="resolved",
        )

        assert result["status"] == "resolved"

    @patch("src.integrations.ticket_service.requests.get")
    def test_get_customer_tickets(
        self,
        mock_get,
        service,
    ):
        """Customer tickets should be returned."""
        mock_get.return_value = mock_response(
            json_data=[
                {
                    "id": "TKT-10001",
                    "customer_id": "CUST-10001",
                },
                {
                    "id": "TKT-10002",
                    "customer_id": "CUST-10001",
                },
            ]
        )

        tickets = service.get_customer_tickets(
            "CUST-10001"
        )

        assert len(tickets) == 2

    @patch("src.integrations.ticket_service.requests.post")
    def test_ticket_service_error(
        self,
        mock_post,
        service,
    ):
        """Server errors should raise TicketServiceError."""
        mock_post.return_value = mock_response(
            status_code=500,
            json_data={
                "detail": "Ticket service unavailable"
            },
        )

        with pytest.raises(TicketServiceError):
            service.create_ticket(
                customer_id="CUST-10001",
                subject="Test ticket",
                description="Test description",
            )


# ============================================================
# Email Service Tests
# ============================================================


class TestEmailService:
    """Tests for EmailService."""

    @pytest.fixture
    def service(self):
        """Create an email service with test configuration."""
        return EmailService(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="test-password",
        )

    def test_email_service_initialization(self, service):
        """Email service should initialize correctly."""
        assert service.smtp_host == "smtp.example.com"
        assert service.smtp_port == 587
        assert service.username == "test@example.com"

    def test_missing_configuration_raises_error(self):
        """Missing SMTP configuration should raise an error."""
        with pytest.raises(
            EmailServiceError
        ):
            EmailService(
                smtp_host=None,
                smtp_port=587,
                username=None,
                password=None,
            )

    @patch(
        "src.integrations.email_service.smtplib.SMTP"
    )
    def test_send_email(
        self,
        mock_smtp,
        service,
    ):
        """Email should be sent through SMTP."""
        smtp_instance = MagicMock()
        mock_smtp.return_value = smtp_instance

        result = service.send_email(
            to_email="customer@example.com",
            subject="Order Update",
            body="Your order has been shipped.",
        )

        assert result is True

        smtp_instance.starttls.assert_called_once()
        smtp_instance.login.assert_called_once()
        smtp_instance.send_message.assert_called_once()
        smtp_instance.quit.assert_called_once()

    @patch(
        "src.integrations.email_service.smtplib.SMTP"
    )
    def test_send_customer_notification(
        self,
        mock_smtp,
        service,
    ):
        """Customer notification should send an email."""
        smtp_instance = MagicMock()
        mock_smtp.return_value = smtp_instance

        result = service.send_customer_notification(
            customer_email="customer@example.com",
            subject="Order Update",
            message="Your order is on the way.",
        )

        assert result is True
        smtp_instance.send_message.assert_called_once()

    @patch(
        "src.integrations.email_service.smtplib.SMTP"
    )
    def test_send_email_failure(
        self,
        mock_smtp,
        service,
    ):
        """SMTP errors should raise EmailServiceError."""
        mock_smtp.side_effect = OSError(
            "SMTP connection failed"
        )

        with pytest.raises(EmailServiceError):
            service.send_email(
                to_email="customer@example.com",
                subject="Test",
                body="Test email",
            )

    @patch(
        "src.integrations.email_service.smtplib.SMTP"
    )
    def test_email_health_check(
        self,
        mock_smtp,
        service,
    ):
        """Email health check should succeed."""
        smtp_instance = MagicMock()
        mock_smtp.return_value = smtp_instance

        assert service.health_check() is True


# ============================================================
# Webhook Tests
# ============================================================


class TestWebhookService:
    """Tests for WebhookService."""

    @pytest.fixture
    def service(self):
        """Create webhook service."""
        return WebhookService(
            webhook_secret="test-webhook-secret"
        )

    @pytest.fixture
    def payload(self):
        """Return a sample webhook payload."""
        return {
            "event": "order.updated",
            "event_id": "evt-10001",
            "data": {
                "order_id": "ORD-10005",
                "status": "shipped",
            },
        }

    def test_service_initialization(self, service):
        """Webhook service should initialize."""
        assert service.secret == "test-webhook-secret"

    def test_generate_signature(
        self,
        service,
        payload,
    ):
        """Webhook signature should be generated."""
        import json

        payload_bytes = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")

        signature = service.generate_signature(
            payload_bytes
        )

        assert isinstance(signature, str)
        assert signature.startswith("sha256=")

    def test_valid_signature(
        self,
        service,
        payload,
    ):
        """Valid signatures should pass verification."""
        import json

        payload_bytes = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")

        signature = service.generate_signature(
            payload_bytes
        )

        assert service.verify_signature(
            payload_bytes,
            signature,
        ) is True

    def test_invalid_signature(
        self,
        service,
        payload,
    ):
        """Invalid signatures should be rejected."""
        import json

        payload_bytes = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")

        with pytest.raises(
            InvalidWebhookSignature
        ):
            service.verify_signature(
                payload_bytes,
                "sha256=invalid-signature",
            )

    def test_parse_webhook(
        self,
        service,
        payload,
    ):
        """Webhook payload should be parsed correctly."""
        import json

        payload_bytes = json.dumps(
            payload
        ).encode("utf-8")

        signature = service.generate_signature(
            payload_bytes
        )

        event = service.parse_webhook(
            payload_bytes,
            signature,
        )

        assert event.event_type == "order.updated"
        assert event.event_id == "evt-10001"
        assert event.data["order_id"] == "ORD-10005"

    def test_parse_webhook_supports_type_field(
        self,
        service,
    ):
        """Webhook parser should support alternative event field."""
        payload = {
            "type": "ticket.created",
            "id": "evt-20001",
            "data": {
                "ticket_id": "TKT-10001"
            },
        }

        import json

        payload_bytes = json.dumps(
            payload
        ).encode("utf-8")

        signature = service.generate_signature(
            payload_bytes
        )

        event = service.parse_webhook(
            payload_bytes,
            signature,
        )

        assert event.event_type == "ticket.created"
        assert event.event_id == "evt-20001"

    def test_process_webhook(
        self,
        service,
        payload,
    ):
        """Webhook should be processed after validation."""
        import json

        payload_bytes = json.dumps(
            payload
        ).encode("utf-8")

        signature = service.generate_signature(
            payload_bytes
        )

        result = service.process_webhook(
            payload_bytes,
            signature,
        )

        assert result.event_type == "order.updated"
        assert result.event_id == "evt-10001"

    def test_invalid_json(
        self,
        service,
    ):
        """Invalid JSON payload should be rejected."""
        payload_bytes = b"not-valid-json"

        signature = service.generate_signature(
            payload_bytes
        )

        with pytest.raises(
            Exception
        ):
            service.parse_webhook(
                payload_bytes,
                signature,
            )