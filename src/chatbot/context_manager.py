"""
Conversation context management.

Manages conversation history, session-level context, prompt-ready
history, metadata, and context size limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

ALLOWED_ROLES = {"user", "assistant", "system"}

# ---------------------------------------------------------
# Chat Message
# ---------------------------------------------------------

@dataclass
class ChatMessage:
    """Represents a single conversation message."""

    role: str
    content: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message fields."""

        if not isinstance(self.role, str):
            raise ValueError("Message role must be a string.")

        self.role = self.role.strip().lower()

        if self.role not in ALLOWED_ROLES:
            raise ValueError("Invalid message role.")

        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string.")

        self.content = self.content.strip()

        if not self.content:
            raise ValueError("Message content cannot be empty.")

        if not isinstance(self.timestamp, datetime):
            raise ValueError("Message timestamp must be a datetime.")

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

        if not isinstance(self.metadata, dict):
            raise ValueError("Message metadata must be a dictionary.")

        # Avoid sharing a caller-owned dictionary with stored history.
        self.metadata = dict(self.metadata)

# ---------------------------------------------------------
# Conversation Context
# ---------------------------------------------------------

@dataclass
class ConversationContext:
    """Stores context associated with one conversation."""

    conversation_id: str
    customer_id: str | None = None
    order_id: str | None = None
    session_id: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------
# Context Manager
# ---------------------------------------------------------

class ContextManager:
    """Manage conversation history in memory."""

    def __init__(
        self,
        max_messages: int = 20,
        max_context_characters: int = 12000,
    ) -> None:
        if (
            isinstance(max_messages, bool)
            or not isinstance(max_messages, int)
            or max_messages <= 0
        ):
            raise ValueError("max_messages must be a positive integer.")

        if (
            isinstance(max_context_characters, bool)
            or not isinstance(max_context_characters, int)
            or max_context_characters <= 0
        ):
            raise ValueError(
                "max_context_characters must be a positive integer."
            )

        self.max_messages = max_messages
        self.max_context_characters = max_context_characters
        self._conversations: dict[str, ConversationContext] = {}
        self._lock = RLock()

    # -----------------------------------------------------
    # Conversation Management
    # -----------------------------------------------------

    @staticmethod

    @staticmethod
    def _validate_identifier(
        value: str | None,
        name: str,
        optional: bool = False,
    ) -> str | None:
        """Validate and normalize an identifier."""

        if value is None:
            if optional:
                return None
            raise ValueError(f"{name} cannot be None.")

        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string.")

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{name} cannot be empty.")

        if len(normalized) > 200:
            raise ValueError(
                f"{name} cannot exceed 200 characters."
            )

        return normalized

    @classmethod
    def _validate_required_identifier(
        cls,
        value: str,
        name: str,
    ) -> str:
        """Validate an identifier that must always be present."""

        result = cls._validate_identifier(value, name)

        if result is None:
            raise ValueError(f"{name} cannot be None.")

        return result

    def create_conversation(
        self,
        conversation_id: str,
        customer_id: str | None = None,
        order_id: str | None = None,
        session_id: str | None = None,
    ) -> ConversationContext:
        """Create or replace a conversation context."""

        conversation_id = self._validate_required_identifier(
            conversation_id, "conversation_id"
        )
        customer_id = self._validate_identifier(
            customer_id, "customer_id", optional=True
        )
        order_id = self._validate_identifier(
            order_id, "order_id", optional=True
        )
        session_id = self._validate_identifier(
            session_id, "session_id", optional=True
        )

        context = ConversationContext(
            conversation_id=conversation_id,
            customer_id=customer_id,
            order_id=order_id,
            session_id=session_id,
        )

        # Preserve the original behavior: creating an existing ID
        # replaces its previous in-memory context.
        with self._lock:
            self._conversations[conversation_id] = context

        return context

    def get_conversation(
        self,
        conversation_id: str,
    ) -> ConversationContext | None:
        """Return an existing conversation, if present."""

        conversation_id = self._validate_required_identifier(
            conversation_id, "conversation_id"
        )

        with self._lock:
            return self._conversations.get(conversation_id)

    def get_or_create_conversation(
        self,
        conversation_id: str,
        customer_id: str | None = None,
        order_id: str | None = None,
        session_id: str | None = None,
    ) -> ConversationContext:
        """Return an existing conversation or create one."""

        conversation_id = self._validate_required_identifier(
            conversation_id, "conversation_id"
        )

        with self._lock:
            context = self._conversations.get(conversation_id)

            if context is not None:
                return context

            return self.create_conversation(
                conversation_id=conversation_id,
                customer_id=customer_id,
                order_id=order_id,
                session_id=session_id,
            )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and report whether it existed."""

        conversation_id = self._validate_required_identifier(
            conversation_id, "conversation_id"
        )

        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None

    def clear(self) -> None:
        """Clear all in-memory conversations."""

        with self._lock:
            self._conversations.clear()

    # -----------------------------------------------------
    # Message Management
    # -----------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Add a validated message to a conversation."""

        conversation_id = self._validate_required_identifier(
            conversation_id, "conversation_id"
        )

        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Message metadata must be a dictionary.")

        message = ChatMessage(
            role=role,
            content=content,
            metadata={} if metadata is None else dict(metadata),
        )

        with self._lock:
            context = self.get_or_create_conversation(conversation_id)
            context.messages.append(message)
            self._trim_messages(context)

        return message

    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Add a user message."""

        return self.add_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            metadata=metadata,
        )

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Add an assistant response."""

        return self.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            metadata=metadata,
        )

    def add_system_message(
        self,
        conversation_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Add a system message."""

        return self.add_message(
            conversation_id=conversation_id,
            role="system",
            content=content,
            metadata=metadata,
        )

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[ChatMessage]:
        """Return a list of conversation messages."""

        context = self.get_conversation(conversation_id)

        if context is None:
            return []

        with self._lock:
            return list(context.messages)

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Return the most recent messages."""

        if limit is None:
            limit = self.max_messages

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer or None.")

        if limit <= 0:
            return []

        return self.get_messages(conversation_id)[-limit:]

    # -----------------------------------------------------
    # Context Generation
    # -----------------------------------------------------

    def build_context(
        self,
        conversation_id: str,
        include_roles: bool = True,
    ) -> str:
        """Build prompt-ready conversation context."""

        if not isinstance(include_roles, bool):
            raise ValueError("include_roles must be a boolean.")

        messages = self.get_recent_messages(conversation_id)

        if not messages:
            return ""

        lines: list[str] = []

        for message in messages:
            content = message.content.strip()

            if include_roles:
                lines.append(
                    f"{message.role.capitalize()}: {content}"
                )
            else:
                lines.append(content)

        return self._limit_context_length("\n".join(lines))

    def build_chat_history(
        self,
        conversation_id: str,
    ) -> list[dict[str, str]]:
        """Build role/content history for chat APIs."""

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in self.get_recent_messages(conversation_id)
        ]

    # -----------------------------------------------------
    # Conversation Metadata
    # -----------------------------------------------------

    def set_customer(
        self,
        conversation_id: str,
        customer_id: str,
    ) -> None:
        """Associate a customer with a conversation."""

        customer_id = self._validate_required_identifier(
            customer_id, "customer_id"
        )
        with self._lock:
            context = self.get_or_create_conversation(conversation_id)
            context.customer_id = customer_id

    def set_order(
        self,
        conversation_id: str,
        order_id: str,
    ) -> None:
        """Associate an order with a conversation."""

        order_id = self._validate_required_identifier(order_id, "order_id")
        with self._lock:
            context = self.get_or_create_conversation(conversation_id)
            context.order_id = order_id

    def set_session(
        self,
        conversation_id: str,
        session_id: str,
    ) -> None:
        """Associate a session with a conversation."""

        session_id = self._validate_required_identifier(session_id, "session_id")
        with self._lock:
            context = self.get_or_create_conversation(conversation_id)
            context.session_id = session_id

    def set_metadata(
        self,
        conversation_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Store custom conversation metadata."""

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Metadata key cannot be empty.")

        with self._lock:
            context = self.get_or_create_conversation(conversation_id)
            context.metadata[key.strip()] = value

    def get_metadata(
        self,
        conversation_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve conversation metadata."""

        context = self.get_conversation(conversation_id)

        if context is None:
            return default

        with self._lock:
            return context.metadata.get(key, default)

    # -----------------------------------------------------
    # Context Statistics
    # -----------------------------------------------------

    def message_count(self, conversation_id: str) -> int:
        """Return the number of stored messages."""

        return len(self.get_messages(conversation_id))

    def has_conversation(self, conversation_id: str) -> bool:
        """Check whether a conversation exists."""

        conversation_id = self._validate_required_identifier(
            conversation_id, "conversation_id"
        )

        with self._lock:
            return conversation_id in self._conversations

    def list_conversation_ids(self) -> list[str]:
        """Return all active in-memory conversation IDs."""

        with self._lock:
            return list(self._conversations.keys())

    # -----------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------

    def _trim_messages(
        self,
        context: ConversationContext,
    ) -> None:
        """Keep the most recent messages within the configured limit."""

        if len(context.messages) > self.max_messages:
            context.messages = context.messages[-self.max_messages:]

    def _limit_context_length(self, context: str) -> str:
        """Keep the newest portion within the character limit."""

        if not isinstance(context, str):
            raise ValueError("Context must be a string.")

        if len(context) <= self.max_context_characters:
            return context

        return context[-self.max_context_characters:]

# ---------------------------------------------------------
# Default Context Manager
# ---------------------------------------------------------

_context_manager = ContextManager()

def get_context_manager() -> ContextManager:
    """Return the default context manager instance."""

    return _context_manager
