"""
Reusable validation utilities for the AI Customer Support
Integration System.

Provides validation for strings, emails, identifiers, messages,
names, numeric values, choices, and optional inputs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection
from decimal import Decimal, InvalidOperation


# =========================================================
# Validation Patterns
# =========================================================

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")


# =========================================================
# Generic Validators
# =========================================================

def is_non_empty_string(value: object) -> bool:
    """Return whether a value is a non-empty string."""

    return isinstance(value, str) and bool(value.strip())


def validate_non_empty_string(
    value: object,
    field_name: str = "value",
) -> str:
    """Validate and return a trimmed, non-empty string."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")

    return normalized


def _validate_max_length(
    max_length: int,
    field_name: str = "max_length",
) -> int:
    """Validate a positive integer length limit."""

    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError(f"{field_name} must be an integer.")

    if max_length <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")

    return max_length


# =========================================================
# Email Validation
# =========================================================

def is_valid_email(email: object) -> bool:
    """Check whether an email has a reasonable basic format."""

    if not isinstance(email, str):
        return False

    email = email.strip()

    if not email or len(email) > 254:
        return False

    local_part, separator, domain = email.rpartition("@")

    if not separator or not local_part or not domain:
        return False

    if len(local_part) > 64:
        return False

    if (
        local_part.startswith(".")
        or local_part.endswith(".")
        or ".." in local_part
    ):
        return False

    if ".." in domain:
        return False

    return bool(EMAIL_PATTERN.fullmatch(email))


def validate_email(email: object) -> str:
    """Validate and return a normalized email address."""

    if not isinstance(email, str):
        raise ValueError("Email must be a string.")

    normalized = email.strip().lower()

    if not is_valid_email(normalized):
        raise ValueError("Invalid email address.")

    return normalized


def validate_optional_email(email: str | None) -> str | None:
    """Validate an optional email address."""

    if email is None:
        return None

    return validate_email(email)


# =========================================================
# Identifier Validation
# =========================================================

def is_valid_id(value: object) -> bool:
    """Check an identifier containing letters, digits, _ or -."""

    if not isinstance(value, str):
        return False

    return bool(ID_PATTERN.fullmatch(value.strip()))


def validate_id(
    value: object,
    field_name: str = "ID",
) -> str:
    """Validate and return a trimmed external identifier."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")

    normalized = value.strip()

    if not ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid {field_name}.")

    return normalized


def validate_customer_id(customer_id: object) -> str:
    """Validate a customer identifier."""

    return validate_id(customer_id, "customer ID")


def validate_order_id(order_id: object) -> str:
    """Validate an order identifier."""

    return validate_id(order_id, "order ID")


def validate_ticket_id(ticket_id: object) -> str:
    """Validate a support ticket identifier."""

    return validate_id(ticket_id, "ticket ID")


def validate_optional_id(
    value: str | None,
    field_name: str = "ID",
) -> str | None:
    """Validate an optional identifier."""

    if value is None:
        return None

    return validate_id(value, field_name)


# =========================================================
# Chat Message Validation
# =========================================================

def validate_message(
    message: object,
    max_length: int = 5000,
) -> str:
    """Validate and return a customer chat message."""

    _validate_max_length(max_length)

    if not isinstance(message, str):
        raise ValueError("Message must be a string.")

    normalized = message.strip()

    if not normalized:
        raise ValueError("Message cannot be empty.")

    if len(normalized) > max_length:
        raise ValueError(
            f"Message cannot exceed {max_length} characters."
        )

    return normalized


# =========================================================
# Name Validation
# =========================================================

def validate_name(
    name: object,
    field_name: str = "name",
    max_length: int = 150,
) -> str:
    """Validate a non-empty name within the length limit."""

    _validate_max_length(max_length)

    normalized = validate_non_empty_string(name, field_name)

    if len(normalized) > max_length:
        raise ValueError(
            f"{field_name} cannot exceed {max_length} characters."
        )

    return normalized


# =========================================================
# Numeric Validation
# =========================================================

def validate_positive_integer(
    value: object,
    field_name: str = "value",
) -> int:
    """Validate an integer greater than zero."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )

    return value


def validate_non_negative_integer(
    value: object,
    field_name: str = "value",
) -> int:
    """Validate an integer greater than or equal to zero."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")

    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return value


def validate_amount(
    amount: object,
    field_name: str = "amount",
) -> float:
    """Validate a finite, non-negative monetary amount."""

    if isinstance(amount, bool) or not isinstance(
        amount, (int, float, Decimal)
    ):
        raise ValueError(f"{field_name} must be numeric.")

    try:
        normalized = float(amount)
    except (ValueError, OverflowError, InvalidOperation) as exc:
        raise ValueError(
            f"{field_name} must be a finite number."
        ) from exc

    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number.")

    if normalized < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return normalized


# =========================================================
# Choice Validation
# =========================================================

def validate_choice(
    value: object,
    choices: Collection[str],
    field_name: str = "value",
) -> str:
    """Validate a value against allowed choices, case-insensitively."""

    if isinstance(choices, (str, bytes)) or not isinstance(
        choices, Collection
    ):
        raise ValueError("choices must be a collection of strings.")

    if not choices:
        raise ValueError("choices cannot be empty.")

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")

    normalized_choices: set[str] = set()

    for choice in choices:
        if not isinstance(choice, str):
            raise ValueError(
                "Every allowed choice must be a string."
            )

        normalized_choice = choice.strip().upper()

        if not normalized_choice:
            raise ValueError(
                "Allowed choices cannot contain empty strings."
            )

        normalized_choices.add(normalized_choice)

    normalized_value = value.strip().upper()

    if normalized_value not in normalized_choices:
        allowed = ", ".join(sorted(normalized_choices))
        raise ValueError(
            f"Invalid {field_name}. Allowed values: {allowed}"
        )

    return normalized_value
