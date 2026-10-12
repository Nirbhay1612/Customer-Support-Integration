"""

AI response generation for the Customer Support System.

Combines customer messages, retrieved knowledge, and verified

business context to generate customer-facing responses.

"""

from __future__ import annotations

from src.ai.llm import generate_llm_response

from src.ai.prompts import (

    SYSTEM_PROMPT,

    FALLBACK_RESPONSE,

    build_customer_support_prompt,

    build_order_prompt,

    build_rag_prompt,

)

from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_MESSAGE_LENGTH = 5000

MAX_CONTEXT_LENGTH = 20000

GENERAL_TEMPERATURE = 0.2

RAG_TEMPERATURE = 0.1

ORDER_TEMPERATURE = 0.1

GENERAL_MAX_TOKENS = 800

RAG_MAX_TOKENS = 800

ORDER_MAX_TOKENS = 600

ORDER_FALLBACK_RESPONSE = (

    "I couldn't retrieve the required order information. "

    "Please try again or contact our support team."

)

class ResponseGenerator:

    """Generate customer-facing responses using the configured LLM."""

    def __init__(self) -> None:

        self.system_prompt = SYSTEM_PROMPT

    # -----------------------------------------------------

    # Input Validation

    # -----------------------------------------------------

    @staticmethod

    def _is_valid_text(value: object) -> bool:

        """Return True when value is a non-empty string."""

        return isinstance(value, str) and bool(value.strip())

    @staticmethod

    def _validate_message(message: object) -> str:

        """Validate and normalize an incoming customer message."""

        if not isinstance(message, str):

            raise ValueError("Message must be a string.")

        normalized = message.strip()

        if not normalized:

            raise ValueError("Message cannot be empty.")

        if len(normalized) > MAX_MESSAGE_LENGTH:

            raise ValueError(

                f"Message cannot exceed {MAX_MESSAGE_LENGTH} characters."

            )

        return normalized

    @staticmethod

    def _validate_context(context: object, field_name: str) -> str:

        """Validate and normalize a context string."""

        if not isinstance(context, str):

            raise ValueError(f"{field_name} must be a string.")

        normalized = context.strip()

        if len(normalized) > MAX_CONTEXT_LENGTH:

            raise ValueError(

                f"{field_name} cannot exceed "

                f"{MAX_CONTEXT_LENGTH} characters."

            )

        return normalized

    @staticmethod

    def _clean_response(response: object) -> str:

        """Return a usable response or the configured fallback."""

        if not isinstance(response, str):

            return FALLBACK_RESPONSE

        cleaned = response.strip()

        if not cleaned:

            return FALLBACK_RESPONSE

        return cleaned

    # -----------------------------------------------------

    # General Response

    # -----------------------------------------------------

    def generate(

        self,

        message: str,

        knowledge_context: str = "",

        customer_context: str = "",

    ) -> str:

        """Generate a support response using available context."""

        try:

            message = self._validate_message(message)

            knowledge_context = self._validate_context(

                knowledge_context,

                "knowledge_context",

            )

            customer_context = self._validate_context(

                customer_context,

                "customer_context",

            )

            prompt = build_customer_support_prompt(

                message=message,

                customer_context=customer_context,

                knowledge_context=knowledge_context,

            )

            response = generate_llm_response(

                system_prompt=self.system_prompt,

                user_message=prompt,

                temperature=GENERAL_TEMPERATURE,

                max_tokens=GENERAL_MAX_TOKENS,

            )

            cleaned = self._clean_response(response)

            if cleaned == FALLBACK_RESPONSE:

                logger.warning("General response was empty or invalid.")

            return cleaned

        except Exception as exc:

            logger.error(

                "Customer-support response generation failed (%s).",

                type(exc).__name__,

            )

            return FALLBACK_RESPONSE

    # -----------------------------------------------------

    # RAG Response

    # -----------------------------------------------------

    def generate_from_rag(

        self,

        question: str,

        context: str,

    ) -> str:

        """Generate an answer grounded in retrieved knowledge."""

        try:

            question = self._validate_message(question)

            context = self._validate_context(context, "context")

            if not context:

                logger.warning(

                    "RAG response skipped because context is empty."

                )

                return FALLBACK_RESPONSE

            prompt = build_rag_prompt(

                question=question,

                context=context,

            )

            response = generate_llm_response(

                system_prompt=self.system_prompt,

                user_message=prompt,

                temperature=RAG_TEMPERATURE,

                max_tokens=RAG_MAX_TOKENS,

            )

            cleaned = self._clean_response(response)

            if cleaned == FALLBACK_RESPONSE:

                logger.warning("RAG response was empty or invalid.")

            return cleaned

        except Exception as exc:

            logger.error(

                "RAG response generation failed (%s).",

                type(exc).__name__,

            )

            return FALLBACK_RESPONSE

    # -----------------------------------------------------

    # Order Response

    # -----------------------------------------------------

    def generate_order_response(

        self,

        message: str,

        order_context: str,

    ) -> str:

        """Generate a response using supplied order information."""

        try:

            message = self._validate_message(message)

            order_context = self._validate_context(

                order_context,

                "order_context",

            )

            if not order_context:

                logger.warning(

                    "Order response skipped because order context is empty."

                )

                return ORDER_FALLBACK_RESPONSE

            prompt = build_order_prompt(

                message=message,

                order_context=order_context,

            )

            response = generate_llm_response(

                system_prompt=self.system_prompt,

                user_message=prompt,

                temperature=ORDER_TEMPERATURE,

                max_tokens=ORDER_MAX_TOKENS,

            )

            cleaned = self._clean_response(response)

            if cleaned == FALLBACK_RESPONSE:

                return ORDER_FALLBACK_RESPONSE

            return cleaned

        except Exception as exc:
            logger.error(
                "Order response generation failed (%s).",
                type(exc).__name__,
            )
            return FALLBACK_RESPONSE

    # -----------------------------------------------------

    # Context-Aware Response

    # -----------------------------------------------------

    def generate_with_context(

        self,

        message: str,

        context: dict[str, str] | None = None,

    ) -> str:

        """

        Generate a response using application context.

        Supported keys:

            knowledge: Retrieved knowledge-base content.

            customer: Verified customer information.

            order: Verified order information.

        """

        if context is None:

            context = {}

        if not isinstance(context, dict):

            logger.warning("Invalid context type supplied.")

            return FALLBACK_RESPONSE

        allowed_keys = {"knowledge", "customer", "order"}

        if not all(isinstance(key, str) for key in context):

            logger.warning("Context contains non-string keys.")

            return FALLBACK_RESPONSE

        unsupported_keys = set(context) - allowed_keys

        if unsupported_keys:

            logger.warning("Context contains unsupported keys.")

            return FALLBACK_RESPONSE

        for key, value in context.items():

            if not isinstance(value, str):

                logger.warning(

                    "Context value for '%s' must be a string.",

                    key,

                )

                return FALLBACK_RESPONSE

        try:

            message = self._validate_message(message)

            knowledge_context = self._validate_context(

                context.get("knowledge", ""),

                "knowledge_context",

            )

            customer_context = self._validate_context(

                context.get("customer", ""),

                "customer_context",

            )

            order_context = self._validate_context(

                context.get("order", ""),

                "order_context",

            )

        except ValueError as exc:

            logger.warning(

                "Invalid input to context-aware response (%s).",

                type(exc).__name__,

            )

            return FALLBACK_RESPONSE

        # Order questions use the order-specific generation path.

        # The caller must authenticate the user and authorize access

        # to the order before supplying its context.

        if order_context:

            return self.generate_order_response(

                message=message,

                order_context=order_context,

            )

        # Use both knowledge and customer context when available.

        if knowledge_context and customer_context:

            return self.generate(

                message=message,

                knowledge_context=knowledge_context,

                customer_context=customer_context,

            )

        # Ground knowledge-only answers in retrieved documents.

        if knowledge_context:

            return self.generate_from_rag(

                question=message,

                context=knowledge_context,

            )

        # General response with customer context, if available.

        return self.generate(

            message=message,

            customer_context=customer_context,

        )

# ---------------------------------------------------------

# Shared Generator

# ---------------------------------------------------------

_response_generator = ResponseGenerator()

# ---------------------------------------------------------

# Convenience Functions

# ---------------------------------------------------------

def generate_response(

    message: str,

    knowledge_context: str = "",

    customer_context: str = "",

) -> str:

    """Generate a general customer-support response."""

    return _response_generator.generate(

        message=message,

        knowledge_context=knowledge_context,

        customer_context=customer_context,

    )

def generate_rag_response(

    question: str,

    context: str,

) -> str:

    """Generate a response grounded in retrieved knowledge."""

    return _response_generator.generate_from_rag(

        question=question,

        context=context,

    )

def generate_order_response(

    message: str,

    order_context: str,

) -> str:

    """Generate an order-related response."""

    return _response_generator.generate_order_response(

        message=message,

        order_context=order_context,

    )
