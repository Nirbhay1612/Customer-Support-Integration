
"""
Streamlit interface for the AI Customer Support system.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.chatbot.chatbot import CustomerSupportChatbot
from src.utils.logger import get_logger


logger = get_logger(__name__)

ERROR_MESSAGE = (
    "I'm sorry, but I couldn't process your request right now. "
    "Please try again."
)

WELCOME_MESSAGE = """
👋 **Hello! How can I help you today?**

You can ask me about:

- 📦 Orders
- 🚚 Shipping
- 🔄 Returns
- 💰 Refunds
- 💳 Payments
- ❓ General questions
"""


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Application Styling
# =========================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .chat-header {
            padding: 1rem 0;
        }

        .status-box {
            padding: 0.75rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Chatbot Initialization
# =========================================================

@st.cache_resource(show_spinner=False)
def initialize_chatbot() -> CustomerSupportChatbot:
    """
    Initialize and reuse the chatbot resource.

    The chatbot must be safe for concurrent use across sessions.
    Conversation-specific state should be isolated by conversation ID.
    """

    logger.info("Initializing customer support chatbot.")

    chatbot = CustomerSupportChatbot()

    logger.info("Customer support chatbot initialized successfully.")

    return chatbot


# =========================================================
# Session State
# =========================================================

def initialize_session() -> None:
    """Initialize session state with safe defaults."""

    defaults: dict[str, Any] = {
        "messages": [],
        "conversation_id": None,
        "customer_id": None,
        "order_id": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_conversation() -> None:
    """Clear the current conversation and its session context."""

    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.customer_id = None
    st.session_state.order_id = None

    logger.info("Conversation cleared for the current session.")


# =========================================================
# Response Normalization
# =========================================================

def normalize_response(response: Any) -> str:
    """Convert supported chatbot response formats into display text."""

    answer: Any = None
    conversation_id: Any = None

    if isinstance(response, str):
        answer = response

    elif isinstance(response, dict):
        # Use the first non-empty supported answer field.
        for key in ("response", "answer"):
            candidate = response.get(key)

            if isinstance(candidate, str) and candidate.strip():
                answer = candidate
                break

        conversation_id = response.get("conversation_id")

    else:
        # Support response objects and dataclasses.
        for attribute in ("response", "answer"):
            candidate = getattr(response, attribute, None)

            if isinstance(candidate, str) and candidate.strip():
                answer = candidate
                break

        conversation_id = getattr(
            response,
            "conversation_id",
            None,
        )

    if isinstance(conversation_id, str) and conversation_id.strip():
        st.session_state.conversation_id = conversation_id.strip()

    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    logger.warning(
        "Chatbot returned an empty or unsupported response."
    )

    return ERROR_MESSAGE


# =========================================================
# Sidebar
# =========================================================

def render_sidebar(chatbot_available: bool) -> None:
    """Render the support-center sidebar."""

    with st.sidebar:
        st.title("⚙️ Support Center")
        st.divider()
        st.subheader("System")

        if chatbot_available:
            st.success("AI Support Ready")
        else:
            st.error("AI Support Unavailable")

        st.markdown(
            """
            **Capabilities**

            - 🤖 AI Customer Support
            - 📚 RAG Knowledge Base
            - 📦 Order Lookup
            - 🎯 Intent Detection
            - 🎫 Ticket Escalation
            - 📧 Email Notifications
            """
        )

        st.divider()

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        ):
            clear_conversation()
            st.rerun()


# =========================================================
# Chat History
# =========================================================

def render_chat_history() -> None:
    """Render valid messages stored in the current session."""

    messages = st.session_state.messages

    if not isinstance(messages, list):
        logger.error("Invalid chat history in session state.")
        st.session_state.messages = []
        return

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        with st.chat_message(role):
            st.markdown(content)


# =========================================================
# Process User Message
# =========================================================

def process_message(
    chatbot: CustomerSupportChatbot,
    user_message: str,
) -> str:
    """Send a message to the chatbot and normalize its response."""

    if not isinstance(user_message, str) or not user_message.strip():
        return "Please enter a message before sending."

    try:
        response = chatbot.process_message(
            message=user_message.strip(),
            conversation_id=st.session_state.conversation_id,
        )

        return normalize_response(response)

    except Exception:
        logger.exception(
            "Error while processing customer message."
        )

        return ERROR_MESSAGE


# =========================================================
# Main Application
# =========================================================

def main() -> None:
    """Render the application and handle the chat flow."""

    initialize_session()

    st.title("🤖 AI Customer Support")

    st.caption(
        "AI-powered customer support with RAG, "
        "intent detection and business integrations."
    )

    st.divider()

    try:
        chatbot = initialize_chatbot()

    except Exception:
        logger.exception(
            "Failed to initialize chatbot."
        )
        chatbot = None

    render_sidebar(chatbot_available=chatbot is not None)

    if chatbot is None:
        st.error(
            "The AI support system could not be initialized. "
            "Check the application logs and configuration, then retry."
        )
        st.stop()

    # Display a welcome message for new conversations.
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(WELCOME_MESSAGE)

    render_chat_history()

    user_message = st.chat_input(
        "Ask your question...",
        max_chars=5000,
    )

    if not user_message or not user_message.strip():
        return

    user_message = user_message.strip()

    # Store and display the user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    # Generate and display the assistant response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_message(
                chatbot=chatbot,
                user_message=user_message,
            )

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )


# =========================================================
# Application Entry Point
# =========================================================

if __name__ == "__main__":
    main()
