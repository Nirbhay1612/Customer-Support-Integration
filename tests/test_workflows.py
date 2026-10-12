"""
Tests for workflow modules.

Covers:
- Support workflow
- Escalation workflow
- Automation workflow
"""

from unittest.mock import MagicMock, patch

import pytest

from src.workflows.support_workflow import (
    SupportRequest,
    SupportResponse,
    SupportWorkflow,
)
from src.workflows.escalation import (
    EscalationRequest,
    EscalationWorkflow,
)
from src.workflows.automation import (
    AutomationResult,
    SupportAutomation,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_retriever():
    """Create a mock RAG retriever."""
    retriever = MagicMock()

    retriever.get_context.return_value = (
        "Eligible products can be returned within 7 days "
        "of delivery."
    )

    retriever.retrieve.return_value = []

    return retriever


@pytest.fixture
def mock_customer_api():
    """Create a mock customer API."""
    client = MagicMock()

    client.get_customer.return_value = {
        "id": "CUST-10001",
        "external_id": "EXT-CUST-10001",
        "name": "Aarav Sharma",
        "email": "aarav.sharma@example.com",
    }

    return client


@pytest.fixture
def mock_order_api():
    """Create a mock order API."""
    client = MagicMock()

    client.get_order.return_value = {
        "id": "ORD-10005",
        "external_id": "EXT-ORD-10005",
        "customer_id": "CUST-10001",
        "status": "out_for_delivery",
        "total_amount": 4999.00,
        "currency": "INR",
        "tracking_number": "TRK-IND-10005",
    }

    return client


@pytest.fixture
def workflow(
    mock_retriever,
    mock_customer_api,
    mock_order_api,
):
    """Create a SupportWorkflow with mocked dependencies."""
    return SupportWorkflow(
        retriever=mock_retriever,
        customer_api=mock_customer_api,
        order_api=mock_order_api,
    )


# ============================================================
# SupportRequest Tests
# ============================================================


class TestSupportRequest:
    """Tests for SupportRequest."""

    def test_create_request(self):
        """SupportRequest should store customer information."""
        request = SupportRequest(
            message="Where is my order?",
            customer_id="CUST-10001",
            order_id="ORD-10005",
        )

        assert request.message == "Where is my order?"
        assert request.customer_id == "CUST-10001"
        assert request.order_id == "ORD-10005"

    def test_request_without_optional_fields(self):
        """Optional fields should be allowed."""
        request = SupportRequest(
            message="What is your return policy?"
        )

        assert request.message == (
            "What is your return policy?"
        )


# ============================================================
# SupportResponse Tests
# ============================================================


class TestSupportResponse:
    """Tests for SupportResponse."""

    def test_create_response(self):
        """SupportResponse should contain the generated answer."""
        response = SupportResponse(
            "Your order is out for delivery.",
            "ORDER_STATUS",
            "conv-10001",
        )
        assert response is not None
        assert response.response == (
            "Your order is out for delivery."
        )
        assert response.intent == "ORDER_STATUS"
        assert response.conversation_id == "conv-10001"


# ============================================================
# Support Workflow Tests
# ============================================================


class TestSupportWorkflow:
    """Tests for SupportWorkflow."""

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_response"
    )
    def test_process_general_question(
        self,
        mock_generate,
        mock_classify,
        workflow,
    ):
        """General customer questions should be processed."""
        mock_classify.return_value = "GENERAL_FAQ"
        mock_generate.return_value = (
            "Our support team is available to help you."
        )

        request = SupportRequest(
            message="How can I contact support?",
        )

        response = workflow.process(request)

        assert isinstance(response, SupportResponse)
        assert response.response
        assert response.conversation_id

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_order_response"
    )
    def test_process_order_status(
        self,
        mock_generate,
        mock_classify,
        workflow,
        mock_order_api,
    ):
        """Order-status requests should retrieve order data."""
        mock_classify.return_value = "ORDER_STATUS"

        mock_generate.return_value = (
            "Your order is currently out for delivery."
        )

        request = SupportRequest(
            message="Where is my order?",
            customer_id="CUST-10001",
            order_id="ORD-10005",
        )

        response = workflow.process(request)

        assert response.response == (
            "Your order is currently out for delivery."
        )

        mock_order_api.get_order.assert_called_once_with(
            "ORD-10005"
        )

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_rag_response"
    )
    def test_process_return_question(
        self,
        mock_generate,
        mock_classify,
        workflow,
        mock_retriever,
    ):
        """Return questions should use RAG context."""
        mock_classify.return_value = "RETURN"

        mock_generate.return_value = (
            "Eligible products can be returned within 7 days."
        )

        request = SupportRequest(
            message="Can I return my product?"
        )

        response = workflow.process(request)

        assert "return" in response.response.lower()

        mock_retriever.get_context.assert_called_once()

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_rag_response"
    )
    def test_shipping_question_uses_rag(
        self,
        mock_generate,
        mock_classify,
        workflow,
        mock_retriever,
    ):
        """Shipping questions should retrieve shipping context."""
        mock_classify.return_value = "SHIPPING"

        mock_generate.return_value = (
            "Standard shipping usually takes 3 to 5 business days."
        )

        request = SupportRequest(
            message="How long does shipping take?"
        )

        response = workflow.process(request)

        assert response.response
        mock_retriever.get_context.assert_called_once()

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_order_response"
    )
    def test_order_data_is_attached_to_response(
        self,
        mock_generate,
        mock_classify,
        workflow,
    ):
        """Order information should be included in the response."""
        mock_classify.return_value = "ORDER_STATUS"

        mock_generate.return_value = (
            "Your order is out for delivery."
        )

        request = SupportRequest(
            message="Where is my order?",
            order_id="ORD-10005",
        )

        response = workflow.process(request)

        assert response.order_data is not None
        assert (
            response.order_data["id"]
            == "ORD-10005"
        )

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_response"
    )
    def test_conversation_id_is_generated(
        self,
        mock_generate,
        mock_classify,
        workflow,
    ):
        """Workflow should generate a conversation ID."""
        mock_classify.return_value = "GENERAL_FAQ"
        mock_generate.return_value = "Test response."

        request = SupportRequest(
            message="Hello"
        )

        response = workflow.process(request)

        assert response.conversation_id
        assert isinstance(
            response.conversation_id,
            str,
        )

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_response"
    )
    def test_existing_conversation_id_is_preserved(
        self,
        mock_generate,
        mock_classify,
        workflow,
    ):
        """Existing conversation IDs should be preserved."""
        mock_classify.return_value = "GENERAL_FAQ"
        mock_generate.return_value = "Hello!"

        request = SupportRequest(
            message="Hello",
            conversation_id="conv-existing-123",
        )

        response = workflow.process(request)

        assert response.conversation_id == (
            "conv-existing-123"
        )

    def test_empty_message_is_rejected(self, workflow):
        """Empty support messages should be rejected."""
        request = SupportRequest(
            message=""
        )

        with pytest.raises(
            (ValueError, Exception)
        ):
            workflow.process(request)

    def test_whitespace_message_is_rejected(
        self,
        workflow,
    ):
        """Whitespace-only messages should be rejected."""
        request = SupportRequest(
            message="   "
        )

        with pytest.raises(
            (ValueError, Exception)
        ):
            workflow.process(request)

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_response"
    )
    def test_customer_lookup(
        self,
        mock_generate,
        mock_classify,
        workflow,
        mock_customer_api,
    ):
        """Customer information should be retrieved when ID is provided."""
        mock_classify.return_value = "GENERAL_FAQ"
        mock_generate.return_value = "Hello!"

        request = SupportRequest(
            message="Hello",
            customer_id="CUST-10001",
        )

        response = workflow.process(request)

        assert response.customer_data is not None
        assert (
            response.customer_data["id"]
            == "CUST-10001"
        )

        mock_customer_api.get_customer.assert_called_once_with(
            "CUST-10001"
        )

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_response"
    )
    def test_workflow_survives_customer_api_failure(
        self,
        mock_generate,
        mock_classify,
        workflow,
        mock_customer_api,
    ):
        """Customer API failures should not crash the workflow."""
        mock_classify.return_value = "GENERAL_FAQ"
        mock_generate.return_value = "How can I help?"

        mock_customer_api.get_customer.side_effect = (
            Exception("Customer API unavailable")
        )

        request = SupportRequest(
            message="Hello",
            customer_id="CUST-10001",
        )

        response = workflow.process(request)

        assert isinstance(
            response,
            SupportResponse,
        )
        assert response.response


# ============================================================
# Escalation Workflow Tests
# ============================================================


class TestEscalationWorkflow:
    """Tests for EscalationWorkflow."""

    @pytest.fixture
    def escalation(self):
        """Create escalation workflow."""
        return EscalationWorkflow()

    def test_normal_message_does_not_escalate(
        self,
        escalation,
    ):
        """Normal FAQ should not require escalation."""
        request = EscalationRequest(
            message="What are your shipping times?",
            intent="SHIPPING",
        )

        result = escalation.evaluate(request)

        assert result.escalate is False

    def test_human_support_request_escalates(
        self,
        escalation,
    ):
        """Explicit human-support requests should escalate."""
        request = EscalationRequest(
            message="I want to speak to a human agent.",
            intent="GENERAL_FAQ",
        )

        result = escalation.evaluate(request)

        assert result.escalate is True

    def test_complaint_escalates(
        self,
        escalation,
    ):
        """Complaints should be eligible for escalation."""
        request = EscalationRequest(
            message="I want to file a complaint.",
            intent="COMPLAINT",
        )

        result = escalation.evaluate(request)

        assert result.escalate is True

    def test_unauthorized_payment_escalates(
        self,
        escalation,
    ):
        """Unauthorized payment cases should escalate."""
        request = EscalationRequest(
            message="There is an unauthorized payment on my account.",
            intent="PAYMENT",
        )

        result = escalation.evaluate(request)

        assert result.escalate is True

    def test_fraud_keyword_escalates(
        self,
        escalation,
    ):
        """Fraud-related issues should escalate."""
        request = EscalationRequest(
            message="I think this transaction is fraud.",
            intent="PAYMENT",
        )

        result = escalation.evaluate(request)

        assert result.escalate is True

    def test_account_hacked_escalates(
        self,
        escalation,
    ):
        """Account-security incidents should escalate."""
        request = EscalationRequest(
            message="My account has been hacked.",
            intent="TECHNICAL_ISSUE",
        )

        result = escalation.evaluate(request)

        assert result.escalate is True

    def test_legal_issue_escalates(
        self,
        escalation,
    ):
        """Legal-related cases should escalate."""
        request = EscalationRequest(
            message="I need help with a legal issue.",
            intent="COMPLAINT",
        )

        result = escalation.evaluate(request)

        assert result.escalate is True

    def test_repeated_failures_escalate(
        self,
        escalation,
    ):
        """Repeated failed interactions should escalate."""
        request = EscalationRequest(
            message="This still isn't working.",
            intent="TECHNICAL_ISSUE",
            previous_failures=3,
        )

        result = escalation.evaluate(request)

        assert result.escalate is True

    def test_escalation_result_contains_reason(
        self,
        escalation,
    ):
        """Escalated results should explain the reason."""
        request = EscalationRequest(
            message="I want a human agent.",
            intent="HUMAN_SUPPORT",
        )

        result = escalation.evaluate(request)

        assert result.escalate is True
        assert result.reason


# ============================================================
# Automation Workflow Tests
# ============================================================


class TestSupportAutomation:
    """Tests for SupportAutomation."""

    @pytest.fixture
    def automation(self):
        """Create automation workflow."""
        return SupportAutomation()

    def test_automation_initialization(
        self,
        automation,
    ):
        """Automation workflow should initialize."""
        assert automation is not None

    @patch(
        "src.workflows.automation.send_email"
    )
    def test_notify_customer(
        self,
        mock_send_email,
        automation,
    ):
        """Customer notification should trigger email service."""
        mock_send_email.return_value = True

        result = automation.notify_customer(
            customer_email="customer@example.com",
            subject="Order Update",
            message="Your order has shipped.",
        )

        assert isinstance(
            result,
            AutomationResult,
        )
        assert result.success is True

        mock_send_email.assert_called_once()

    @patch(
        "src.workflows.automation.send_email"
    )
    def test_notify_customer_email_failure(
        self,
        mock_send_email,
        automation,
    ):
        """Email failures should be represented in the result."""
        mock_send_email.side_effect = Exception(
            "Email service unavailable"
        )

        result = automation.notify_customer(
            customer_email="customer@example.com",
            subject="Test",
            message="Test message",
        )

        assert isinstance(
            result,
            AutomationResult,
        )
        assert result.success is False

    @patch(
        "src.workflows.automation.send_email"
    )
    def test_notify_order_update(
        self,
        mock_send_email,
        automation,
    ):
        """Order updates should notify the customer."""
        mock_send_email.return_value = True

        result = automation.notify_order_update(
            customer_email="customer@example.com",
            order_id="ORD-10005",
            status="shipped",
        )

        assert result.success is True
        mock_send_email.assert_called_once()

    @patch(
        "src.workflows.automation.send_email"
    )
    def test_notify_ticket_created(
        self,
        mock_send_email,
        automation,
    ):
        """Ticket creation should trigger notification."""
        mock_send_email.return_value = True

        result = automation.notify_ticket_created(
            customer_email="customer@example.com",
            ticket_id="TKT-10001",
            subject="Payment issue",
        )

        assert result.success is True

    @patch(
        "src.workflows.automation.send_email"
    )
    def test_notify_escalation(
        self,
        mock_send_email,
        automation,
    ):
        """Escalation notification should be sent."""
        mock_send_email.return_value = True

        result = automation.notify_escalation(
            customer_email="customer@example.com",
            ticket_id="TKT-10001",
            reason="Customer requested human support.",
        )

        assert result.success is True

    def test_process_webhook_order_update(
        self,
        automation,
    ):
        """Order webhook should be recognized."""
        event = {
            "event": "order.updated",
            "event_id": "evt-10001",
            "data": {
                "order_id": "ORD-10005",
                "status": "shipped",
            },
        }

        result = automation.process_webhook(event)

        assert isinstance(
            result,
            AutomationResult,
        )

    def test_process_webhook_ticket_created(
        self,
        automation,
    ):
        """Ticket-created webhook should be processed."""
        event = {
            "event": "ticket.created",
            "event_id": "evt-20001",
            "data": {
                "ticket_id": "TKT-10001",
            },
        }

        result = automation.process_webhook(event)

        assert isinstance(
            result,
            AutomationResult,
        )

    def test_process_unknown_webhook(
        self,
        automation,
    ):
        """Unknown webhook events should be handled safely."""
        event = {
            "event": "unknown.event",
            "event_id": "evt-99999",
            "data": {},
        }

        result = automation.process_webhook(event)

        assert isinstance(
            result,
            AutomationResult,
        )

    def test_invalid_webhook_payload(
        self,
        automation,
    ):
        """Invalid webhook payload should fail safely."""
        with pytest.raises(
            (ValueError, TypeError, Exception)
        ):
            automation.process_webhook(None)


# ============================================================
# Integration-Level Workflow Tests
# ============================================================


class TestWorkflowIntegration:
    """Tests combining multiple workflow components."""

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_order_response"
    )
    def test_order_support_flow(
        self,
        mock_generate,
        mock_classify,
        workflow,
    ):
        """Customer -> intent -> order lookup -> response flow."""
        mock_classify.return_value = "ORDER_STATUS"

        mock_generate.return_value = (
            "Your order ORD-10005 is out for delivery."
        )

        request = SupportRequest(
            message="Where is my order ORD-10005?",
            customer_id="CUST-10001",
            order_id="ORD-10005",
        )

        response = workflow.process(request)

        assert response.intent == "ORDER_STATUS"
        assert response.response
        assert response.order_data["id"] == "ORD-10005"
        assert response.conversation_id

    @patch(
        "src.workflows.support_workflow.classify_intent"
    )
    @patch(
        "src.workflows.support_workflow.generate_rag_response"
    )
    def test_policy_support_flow(
        self,
        mock_generate,
        mock_classify,
        workflow,
    ):
        """Customer -> intent -> RAG -> response flow."""
        mock_classify.return_value = "REFUND"

        mock_generate.return_value = (
            "Refunds are processed according to the payment policy."
        )

        request = SupportRequest(
            message="When will I receive my refund?"
        )

        response = workflow.process(request)

        assert response.intent == "REFUND"
        assert response.response
        assert response.conversation_id