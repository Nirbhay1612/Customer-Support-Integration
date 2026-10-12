"""
Tests for AI components.

Covers:
- Prompt builders
- Support intent classification
- Response generation
- RAG response generation
- Order response generation
- Context-aware response generation
- Convenience functions
"""

import pytest

from src.ai.intent_classifier import (
    IntentClassifier,
    SupportIntent,
    classify_intent,
)

from src.ai.prompts import (
    FALLBACK_RESPONSE,
    SYSTEM_PROMPT,
    build_customer_support_prompt,
    build_escalation_prompt,
    build_intent_prompt,
    build_order_prompt,
    build_rag_prompt,
    build_ticket_summary_prompt,
)

from src.ai.response_generator import (
    ResponseGenerator,
    generate_order_response,
    generate_rag_response,
    generate_response,
)


# =========================================================
# Prompt Tests
# =========================================================


class TestPrompts:
    """Tests for centralized prompt builders."""

    def test_system_prompt_exists(self):
        assert SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)

    def test_build_rag_prompt_contains_question(self):
        question = "How long does shipping take?"
        context = "Shipping usually takes several business days."

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        assert isinstance(prompt, str)
        assert question in prompt
        assert context in prompt

    def test_build_rag_prompt_does_not_return_none(self):
        prompt = build_rag_prompt(
            question="Where is my order?",
            context="Order information is available.",
        )

        assert prompt is not None
        assert prompt.strip()

    def test_build_customer_support_prompt_contains_message(self):
        message = "I need help with my order."

        prompt = build_customer_support_prompt(
            message=message,
        )

        assert isinstance(prompt, str)
        assert message in prompt

    def test_build_customer_support_prompt_contains_context(self):
        message = "Where is my order?"
        customer_context = "Customer ID: CUST-10001"
        knowledge_context = "Orders can be tracked using the tracking number."

        prompt = build_customer_support_prompt(
            message=message,
            customer_context=customer_context,
            knowledge_context=knowledge_context,
        )

        assert message in prompt
        assert customer_context in prompt
        assert knowledge_context in prompt

    def test_build_order_prompt_contains_order_context(self):
        message = "Where is my order?"

        order_context = (
            "Order ID: ORD-10005\n"
            "Status: out_for_delivery"
        )

        prompt = build_order_prompt(
            message=message,
            order_context=order_context,
        )

        assert isinstance(prompt, str)
        assert message in prompt
        assert "ORD-10005" in prompt
        assert "out_for_delivery" in prompt

    def test_build_intent_prompt_contains_message(self):
        message = "I want to return my product."

        prompt = build_intent_prompt(
            message=message,
        )

        assert isinstance(prompt, str)
        assert message in prompt

    def test_build_escalation_prompt_contains_message(self):
        message = "I need to speak with a human agent."
        response = "I recommend contacting our support team."
        context = "Customer has already tried the available troubleshooting steps."

        prompt = build_escalation_prompt(
            message=message,
            response=response,
            context=context,
        )

        assert isinstance(prompt, str)
        assert message in prompt
        assert response in prompt
        assert context in prompt

    def test_build_ticket_summary_prompt_contains_details(self):
        message = "Customer received the wrong product."
        conversation = (
            "Customer reported that the delivered product "
            "does not match the ordered product."
        )
        context = "Order ID: ORD-10005"

        prompt = build_ticket_summary_prompt(
            message=message,
            conversation=conversation,
            context=context,
        )

        assert isinstance(prompt, str)
        assert message in prompt
        assert conversation in prompt
        assert context in prompt


# =========================================================
# Intent Enum Tests
# =========================================================


class TestSupportIntent:
    """Tests for supported intent values."""

    def test_all_expected_intents_exist(self):
        expected_intents = {
            "GENERAL_FAQ",
            "ORDER_STATUS",
            "SHIPPING",
            "RETURN",
            "REFUND",
            "PAYMENT",
            "CANCELLATION",
            "PRODUCT_INFORMATION",
            "COMPLAINT",
            "TECHNICAL_ISSUE",
            "HUMAN_SUPPORT",
            "UNKNOWN",
        }

        actual_intents = {
            intent.value
            for intent in SupportIntent
        }

        assert expected_intents == actual_intents

    def test_intent_is_string_enum(self):
        assert SupportIntent.SHIPPING.value == "SHIPPING"
        assert SupportIntent.RETURN.value == "RETURN"
        assert SupportIntent.REFUND.value == "REFUND"

    def test_unknown_intent_exists(self):
        assert SupportIntent.UNKNOWN.value == "UNKNOWN"


# =========================================================
# Intent Classifier Tests
# =========================================================


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    def test_classifier_can_be_created(self):
        classifier = IntentClassifier()

        assert classifier is not None

    def test_empty_message_returns_unknown(self):
        classifier = IntentClassifier()

        result = classifier.classify("")

        assert result == SupportIntent.UNKNOWN

    def test_whitespace_message_returns_unknown(self):
        classifier = IntentClassifier()

        result = classifier.classify("   ")

        assert result == SupportIntent.UNKNOWN

    def test_normalize_valid_intent(self):
        classifier = IntentClassifier()

        result = classifier._normalize_intent(
            "SHIPPING"
        )

        assert result == SupportIntent.SHIPPING

    def test_normalize_lowercase_intent(self):
        classifier = IntentClassifier()

        result = classifier._normalize_intent(
            "shipping"
        )

        assert result == SupportIntent.SHIPPING

    def test_normalize_intent_with_backticks(self):
        classifier = IntentClassifier()

        result = classifier._normalize_intent(
            "`RETURN`"
        )

        assert result == SupportIntent.RETURN

    def test_normalize_intent_with_prefix(self):
        classifier = IntentClassifier()

        result = classifier._normalize_intent(
            "INTENT: REFUND"
        )

        assert result == SupportIntent.REFUND

    def test_normalize_invalid_intent(self):
        classifier = IntentClassifier()

        result = classifier._normalize_intent(
            "INVALID_INTENT"
        )

        assert result == SupportIntent.UNKNOWN

    def test_normalize_empty_result(self):
        classifier = IntentClassifier()

        result = classifier._normalize_intent("")

        assert result == SupportIntent.UNKNOWN

    def test_classify_with_mocked_llm(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "SHIPPING"

        monkeypatch.setattr(
            "src.ai.intent_classifier.generate_llm_response",
            fake_llm_response,
        )

        classifier = IntentClassifier()

        result = classifier.classify(
            "How long will my delivery take?"
        )

        assert result == SupportIntent.SHIPPING

    def test_classify_return_intent(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "RETURN"

        monkeypatch.setattr(
            "src.ai.intent_classifier.generate_llm_response",
            fake_llm_response,
        )

        classifier = IntentClassifier()

        result = classifier.classify(
            "I want to return my product."
        )

        assert result == SupportIntent.RETURN

    def test_classify_invalid_llm_result_returns_unknown(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            return "SOMETHING_INVALID"

        monkeypatch.setattr(
            "src.ai.intent_classifier.generate_llm_response",
            fake_llm_response,
        )

        classifier = IntentClassifier()

        result = classifier.classify(
            "Some customer message."
        )

        assert result == SupportIntent.UNKNOWN

    def test_classify_llm_exception_returns_unknown(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            raise RuntimeError("LLM connection failed")

        monkeypatch.setattr(
            "src.ai.intent_classifier.generate_llm_response",
            fake_llm_response,
        )

        classifier = IntentClassifier()

        result = classifier.classify(
            "Where is my order?"
        )

        assert result == SupportIntent.UNKNOWN

    def test_convenience_classify_intent(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "ORDER_STATUS"

        monkeypatch.setattr(
            "src.ai.intent_classifier.generate_llm_response",
            fake_llm_response,
        )

        result = classify_intent(
            "Where is my order?"
        )

        assert result == SupportIntent.ORDER_STATUS


# =========================================================
# Response Generator Tests
# =========================================================


class TestResponseGenerator:
    """Tests for ResponseGenerator."""

    def test_response_generator_can_be_created(self):
        generator = ResponseGenerator()

        assert generator is not None

    def test_system_prompt_is_loaded(self):
        generator = ResponseGenerator()

        assert generator.system_prompt == SYSTEM_PROMPT

    def test_empty_message_returns_fallback(self):
        generator = ResponseGenerator()

        result = generator.generate("")

        assert result == FALLBACK_RESPONSE

    def test_whitespace_message_returns_fallback(self):
        generator = ResponseGenerator()

        result = generator.generate("   ")

        assert result == FALLBACK_RESPONSE

    def test_generate_with_mocked_llm(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "Your order is being processed."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        result = generator.generate(
            message="Where is my order?",
        )

        assert result == "Your order is being processed."

    def test_generate_includes_context(self, monkeypatch):
        captured = {}

        def fake_llm_response(*args, **kwargs):
            captured["user_message"] = kwargs.get(
                "user_message"
            )
            return "Response generated."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        result = generator.generate(
            message="Where is my order?",
            knowledge_context=(
                "Orders can be tracked using the "
                "tracking number."
            ),
            customer_context="Customer ID: CUST-10001",
        )

        assert result == "Response generated."
        assert "Where is my order?" in captured["user_message"]
        assert "CUST-10001" in captured["user_message"]

    def test_generate_from_rag(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return (
                "Shipping usually takes several "
                "business days."
            )

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        result = generator.generate_from_rag(
            question="How long does shipping take?",
            context=(
                "Shipping usually takes several "
                "business days."
            ),
        )

        assert result == (
            "Shipping usually takes several "
            "business days."
        )

    def test_generate_from_rag_empty_question(self):
        generator = ResponseGenerator()

        result = generator.generate_from_rag(
            question="",
            context="Some useful context.",
        )

        assert result == FALLBACK_RESPONSE

    def test_generate_from_rag_empty_context(self):
        generator = ResponseGenerator()

        result = generator.generate_from_rag(
            question="How long does shipping take?",
            context="",
        )

        assert result == FALLBACK_RESPONSE

    def test_generate_from_rag_llm_exception(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            raise RuntimeError("LLM error")

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        result = generator.generate_from_rag(
            question="How long does shipping take?",
            context="Shipping information.",
        )

        assert result == FALLBACK_RESPONSE

    def test_generate_order_response(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "Your order is out for delivery."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        order_context = (
            "Order ID: ORD-10005\n"
            "Status: out_for_delivery"
        )

        result = generator.generate_order_response(
            message="Where is my order?",
            order_context=order_context,
        )

        assert result == (
            "Your order is out for delivery."
        )

    def test_generate_order_response_empty_message(self):
        generator = ResponseGenerator()

        result = generator.generate_order_response(
            message="",
            order_context="Order ID: ORD-10005",
        )

        assert result == FALLBACK_RESPONSE

    def test_generate_order_response_empty_context(self):
        generator = ResponseGenerator()

        result = generator.generate_order_response(
            message="Where is my order?",
            order_context="",
        )

        assert "order information" in result.lower()

    def test_generate_order_response_llm_exception(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            raise RuntimeError("LLM error")

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        result = generator.generate_order_response(
            message="Where is my order?",
            order_context=(
                "Order ID: ORD-10005\n"
                "Status: shipped"
            ),
        )

        assert result == FALLBACK_RESPONSE

    def test_generate_with_context_order_priority(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            return "Your order has been shipped."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        context = {
            "knowledge": (
                "General shipping information."
            ),
            "customer": (
                "Customer ID: CUST-10001"
            ),
            "order": (
                "Order ID: ORD-10001\n"
                "Status: shipped"
            ),
        }

        result = generator.generate_with_context(
            message="Where is my order?",
            context=context,
        )

        assert result == (
            "Your order has been shipped."
        )

    def test_generate_with_context_rag(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            return "Shipping information."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        context = {
            "knowledge": (
                "Shipping usually takes several "
                "business days."
            )
        }

        result = generator.generate_with_context(
            message="How long does shipping take?",
            context=context,
        )

        assert result == "Shipping information."

    def test_generate_with_context_general(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            return "How can I help you."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        context = {
            "customer": (
                "Customer ID: CUST-10001"
            )
        }

        result = generator.generate_with_context(
            message="I need help.",
            context=context,
        )

        assert result == "How can I help you."

    def test_generate_with_context_none(
        self,
        monkeypatch,
    ):
        def fake_llm_response(*args, **kwargs):
            return "How can I help you."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        generator = ResponseGenerator()

        result = generator.generate_with_context(
            message="I need help.",
            context=None,
        )

        assert result == "How can I help you."

    def test_clean_response_removes_whitespace(self):
        result = ResponseGenerator._clean_response(
            "   Hello customer.   "
        )

        assert result == "Hello customer."

    def test_clean_response_empty_returns_fallback(self):
        result = ResponseGenerator._clean_response("")

        assert result == FALLBACK_RESPONSE

    def test_clean_response_none_returns_fallback(self):
        result = ResponseGenerator._clean_response("")

        assert result == FALLBACK_RESPONSE


# =========================================================
# Convenience Function Tests
# =========================================================


class TestResponseGeneratorConvenienceFunctions:
    """Tests for response-generator convenience functions."""

    def test_generate_response(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "Test response."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        result = generate_response(
            message="Hello",
        )

        assert result == "Test response."

    def test_generate_rag_response(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "Shipping information."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        result = generate_rag_response(
            question="How long does shipping take?",
            context=(
                "Shipping usually takes several "
                "business days."
            ),
        )

        assert result == "Shipping information."

    def test_generate_order_response(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "Your order is shipped."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        result = generate_order_response(
            message="What is my order status?",
            order_context=(
                "Order ID: ORD-10001\n"
                "Status: shipped"
            ),
        )

        assert result == "Your order is shipped."


# =========================================================
# Basic Integration Tests
# =========================================================


class TestAIIntegration:
    """Small integration-level tests for AI modules."""

    def test_prompt_to_response_flow(self, monkeypatch):
        def fake_llm_response(*args, **kwargs):
            return "Your request has been received."

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_llm_response,
        )

        message = "I need help with my order."

        prompt = build_customer_support_prompt(
            message=message,
            customer_context="Customer ID: CUST-10001",
            knowledge_context="Support is available.",
        )

        assert message in prompt

        generator = ResponseGenerator()

        response = generator.generate(
            message=message,
            customer_context="Customer ID: CUST-10001",
            knowledge_context="Support is available.",
        )

        assert response
        assert isinstance(response, str)

    def test_intent_and_response_flow(self, monkeypatch):
        def fake_intent_response(*args, **kwargs):
            return "ORDER_STATUS"

        def fake_response(*args, **kwargs):
            return "Your order is currently being processed."

        monkeypatch.setattr(
            "src.ai.intent_classifier.generate_llm_response",
            fake_intent_response,
        )

        monkeypatch.setattr(
            "src.ai.response_generator.generate_llm_response",
            fake_response,
        )

        classifier = IntentClassifier()

        intent = classifier.classify(
            "Where is my order?"
        )

        assert intent == SupportIntent.ORDER_STATUS

        generator = ResponseGenerator()

        response = generator.generate(
            message="Where is my order?",
        )

        assert response == (
            "Your order is currently being processed."
        )


# =========================================================
# Pytest Sanity Test
# =========================================================


def test_ai_modules_import_successfully():
    """Ensure the main AI modules can be imported."""

    assert SupportIntent is not None
    assert IntentClassifier is not None
    assert ResponseGenerator is not None
    assert build_rag_prompt is not None
    assert build_customer_support_prompt is not None
    assert build_order_prompt is not None
    assert build_intent_prompt is not None