"""
Custom exception hierarchy for the AI Customer Support
Integration System.

Provides consistent, domain-specific exceptions for AI processing,
RAG retrieval, database operations, external integrations, and
support workflows.
"""

from __future__ import annotations


# =========================================================
# Base Application Exceptions
# =========================================================

class ApplicationError(Exception):
    """Base exception for application-level errors."""

    default_message = "An application error occurred."

    def __init__(self, message: str | None = None) -> None:
        normalized_message = (
            message.strip()
            if isinstance(message, str) and message.strip()
            else self.default_message
        )

        self.message = normalized_message
        super().__init__(normalized_message)


# =========================================================
# Configuration Exceptions
# =========================================================

class ConfigurationError(ApplicationError):
    """Raised when application configuration is invalid."""

    default_message = "The application configuration is invalid."


# =========================================================
# AI and LLM Exceptions
# =========================================================

class AIError(ApplicationError):
    """Base exception for AI-related failures."""

    default_message = "An AI processing error occurred."


class LLMError(AIError):
    """Raised when a language model request fails."""

    default_message = "The language model request failed."


class LLMConfigurationError(LLMError):
    """Raised when the LLM provider is misconfigured."""

    default_message = "The language model configuration is invalid."


class LLMRateLimitError(LLMError):
    """Raised when an LLM provider rate limit is reached."""

    default_message = "The language model rate limit was reached."


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""

    default_message = "The language model request timed out."


class ResponseGenerationError(AIError):
    """Raised when an AI response cannot be generated."""

    default_message = "Unable to generate an AI response."


class IntentClassificationError(AIError):
    """Raised when intent classification fails."""

    default_message = "Unable to classify the customer intent."


# =========================================================
# RAG and Knowledge Base Exceptions
# =========================================================

class RAGError(ApplicationError):
    """Base exception for retrieval-augmented generation failures."""

    default_message = "The knowledge retrieval process failed."


class DocumentLoadError(RAGError):
    """Raised when a knowledge-base document cannot be loaded."""

    default_message = "Unable to load a knowledge-base document."


class DocumentProcessingError(RAGError):
    """Raised when a document cannot be processed or chunked."""

    default_message = "Unable to process a knowledge-base document."


class EmbeddingError(RAGError):
    """Raised when document or query embeddings cannot be created."""

    default_message = "Unable to generate embeddings."


class VectorStoreError(RAGError):
    """Raised when the vector store operation fails."""

    default_message = "The vector store operation failed."


class RetrievalError(RAGError):
    """Raised when relevant knowledge cannot be retrieved."""

    default_message = "Unable to retrieve relevant knowledge."


# =========================================================
# Database Exceptions
# =========================================================

class DatabaseError(ApplicationError):
    """Base exception for database-related failures."""

    default_message = "A database operation failed."


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection cannot be established."""

    default_message = "Unable to connect to the database."


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""

    default_message = "The database query failed."


# =========================================================
# Integration Exceptions
# =========================================================

class IntegrationError(ApplicationError):
    """Base exception for external integration failures."""

    default_message = "An external integration failed."


class CustomerAPIError(IntegrationError):
    """Raised when a customer API operation fails."""

    default_message = "The customer API request failed."


class OrderAPIError(IntegrationError):
    """Raised when an order API operation fails."""

    default_message = "The order API request failed."


class EmailServiceError(IntegrationError):
    """Raised when an email operation fails."""

    default_message = "The email service operation failed."


class TicketServiceError(IntegrationError):
    """Raised when a ticket system operation fails."""

    default_message = "The ticket service operation failed."


class WebhookError(IntegrationError):
    """Raised when webhook processing fails."""

    default_message = "Webhook processing failed."


class WebhookAuthenticationError(WebhookError):
    """Raised when webhook authentication fails."""

    default_message = "Webhook authentication failed."


# =========================================================
# Workflow Exceptions
# =========================================================

class WorkflowError(ApplicationError):
    """Base exception for support workflow failures."""

    default_message = "The support workflow failed."


class EscalationError(WorkflowError):
    """Raised when a support escalation fails."""

    default_message = "Unable to escalate the support request."


class ValidationError(ApplicationError):
    """Raised when application input validation fails."""

    default_message = "Input validation failed."
