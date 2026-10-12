"""

Repository layer for database operations.

Provides CRUD and query operations for:

- Customers

- Orders

- Tickets

- Conversations

Business logic should use this layer instead of directly

performing SQLAlchemy queries.

"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from src.database.models import Conversation, Customer, Order, Ticket


class RepositoryError(Exception):
    """Base exception for repository operations."""


class RepositoryConflictError(RepositoryError):
    """Raised when a database uniqueness or integrity conflict occurs."""


def _validate_required_text(value: str, field_name: str) -> str:
    """Validate and normalize required text fields."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty string.')
    return value.strip()


def _normalize_optional_text(value: object, field_name: str) -> str | None:
    """Normalize optional text and reject values of unexpected types."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None.")
    return value.strip() or None


def _validate_limit(limit: int, maximum: int = 500) -> int:
    """Validate a pagination limit."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError('limit must be an integer.')
    if not 1 <= limit <= maximum:
        raise ValueError(f'limit must be between 1 and {maximum}.')
    return limit


def _flush(db: Session) -> None:
    """Flush pending changes and normalize database errors."""
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise RepositoryConflictError(
            'Database integrity conflict. Check duplicate IDs, '
            'unique fields, and foreign-key references.'
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise RepositoryError('Database operation failed.') from exc


class CustomerRepository:
    """Database operations related to customers."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        name: str,
        email: str,
        phone: Optional[str] = None,
        external_id: Optional[str] = None,
    ) -> Customer:
        """Create a new customer."""
        name = _validate_required_text(name, 'name')
        email = _validate_required_text(email, 'email').lower()
        customer = Customer(
            name=name,
            email=email,
            phone=_normalize_optional_text(phone, 'phone'),
            external_id=_normalize_optional_text(external_id, 'external_id'),
        )
        self.db.add(customer)
        _flush(self.db)
        return customer

    def get_by_id(self, customer_id: int) -> Optional[Customer]:
        """Get a customer by internal database ID."""
        return self.db.get(Customer, customer_id)

    def get_by_email(self, email: str) -> Optional[Customer]:
        """Get a customer by email address."""
        email = _validate_required_text(email, 'email').lower()
        statement = select(Customer).where(Customer.email == email)
        return self.db.scalar(statement)

    def get_by_external_id(self, external_id: str) -> Optional[Customer]:
        """Get a customer using an external customer ID."""
        external_id = _validate_required_text(external_id, 'external_id')
        statement = select(Customer).where(Customer.external_id == external_id)
        return self.db.scalar(statement)

    def update(self, customer_id: int, **fields: Any) -> Optional[Customer]:
        """Update explicitly permitted customer fields."""
        customer = self.get_by_id(customer_id)
        if customer is None:
            return None
        allowed_fields = {'name', 'email', 'phone', 'external_id'}
        unknown_fields = set(fields) - allowed_fields
        if unknown_fields:
            raise ValueError(f'Unsupported customer fields: {sorted(unknown_fields)}')
        for field, value in fields.items():
            if field in {'name', 'email'}:
                value = _validate_required_text(value, field)
                if field == 'email':
                    value = value.lower()
            elif field in {'phone', 'external_id'} and value is not None:
                if not isinstance(value, str):
                    raise ValueError(f'{field} must be a string or None.')
                value = value.strip() or None
            setattr(customer, field, value)
        _flush(self.db)
        return customer

    def delete(self, customer_id: int) -> bool:
        """Delete a customer and its configured dependent records."""
        customer = self.get_by_id(customer_id)
        if customer is None:
            return False
        self.db.delete(customer)
        _flush(self.db)
        return True


class OrderRepository:
    """Database operations related to orders."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        external_id: str,
        customer_id: int,
        status: str = 'PENDING',
        total_amount: Decimal | float | int = Decimal('0.00'),
        currency: str = 'INR',
        tracking_number: Optional[str] = None,
        shipping_address: Optional[str] = None,
    ) -> Order:
        """Create a new order."""
        external_id = _validate_required_text(external_id, 'external_id')
        status = _validate_required_text(status, 'status').upper()
        currency = _validate_required_text(currency, 'currency').upper()
        if len(currency) != 3:
            raise ValueError('currency must be a three-letter currency code.')
        if isinstance(total_amount, bool):
            raise ValueError('total_amount must be a valid monetary amount.')
        try:
            amount = Decimal(str(total_amount))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError('total_amount must be a valid monetary amount.') from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError('total_amount must be finite and non-negative.')
        order = Order(
            external_id=external_id,
            customer_id=customer_id,
            status=status,
            total_amount=amount,
            currency=currency,
            tracking_number=_normalize_optional_text(tracking_number, 'tracking_number'),
            shipping_address=_normalize_optional_text(shipping_address, 'shipping_address'),
        )
        self.db.add(order)
        _flush(self.db)
        return order

    def get_by_id(self, order_id: int) -> Optional[Order]:
        """Get an order by internal database ID."""
        return self.db.get(Order, order_id)

    def get_by_external_id(self, external_id: str) -> Optional[Order]:
        """Get an order using its external ID."""
        external_id = _validate_required_text(external_id, 'external_id')
        statement = select(Order).where(Order.external_id == external_id)
        return self.db.scalar(statement)

    def get_by_customer(self, customer_id: int) -> list[Order]:
        """Get all orders belonging to a customer."""
        statement = (
            select(Order)
            .where(Order.customer_id == customer_id)
            .order_by(Order.created_at.desc(), Order.id.desc())
        )
        return list(self.db.scalars(statement).all())

    def update_status(self, order_id: int, status: str) -> Optional[Order]:
        """Update an order's status."""
        order = self.get_by_id(order_id)
        if order is None:
            return None
        order.status = _validate_required_text(status, 'status').upper()
        _flush(self.db)
        return order

    def update_tracking(self, order_id: int, tracking_number: str) -> Optional[Order]:
        """Update an order's tracking number."""
        order = self.get_by_id(order_id)
        if order is None:
            return None
        order.tracking_number = _validate_required_text(tracking_number, 'tracking_number')
        _flush(self.db)
        return order


class TicketRepository:
    """Database operations related to support tickets."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        customer_id: int,
        subject: str,
        description: str,
        priority: str = 'MEDIUM',
        category: Optional[str] = None,
        external_id: Optional[str] = None,
    ) -> Ticket:
        """Create a support ticket."""
        subject = _validate_required_text(subject, 'subject')
        description = _validate_required_text(description, 'description')
        priority = _validate_required_text(priority, 'priority').upper()
        if priority not in {'LOW', 'MEDIUM', 'HIGH', 'URGENT'}:
            raise ValueError('priority must be LOW, MEDIUM, HIGH, or URGENT.')
        ticket = Ticket(
            customer_id=customer_id,
            subject=subject,
            description=description,
            priority=priority,
            category=_normalize_optional_text(category, 'category'),
            external_id=_normalize_optional_text(external_id, 'external_id'),
            status='OPEN',
        )
        self.db.add(ticket)
        _flush(self.db)
        return ticket

    def get_by_id(self, ticket_id: int) -> Optional[Ticket]:
        """Get a ticket by internal database ID."""
        return self.db.get(Ticket, ticket_id)

    def get_by_external_id(self, external_id: str) -> Optional[Ticket]:
        """Get a ticket using its external ID."""
        external_id = _validate_required_text(external_id, 'external_id')
        statement = select(Ticket).where(Ticket.external_id == external_id)
        return self.db.scalar(statement)

    def get_by_customer(self, customer_id: int) -> list[Ticket]:
        """Get all tickets belonging to a customer."""
        statement = (
            select(Ticket)
            .where(Ticket.customer_id == customer_id)
            .order_by(Ticket.created_at.desc(), Ticket.id.desc())
        )
        return list(self.db.scalars(statement).all())

    def update_status(self, ticket_id: int, status: str) -> Optional[Ticket]:
        """Update ticket status."""
        ticket = self.get_by_id(ticket_id)
        if ticket is None:
            return None
        ticket.status = _validate_required_text(status, 'status').upper()
        _flush(self.db)
        return ticket

    def update_priority(self, ticket_id: int, priority: str) -> Optional[Ticket]:
        """Update ticket priority."""
        ticket = self.get_by_id(ticket_id)
        if ticket is None:
            return None
        priority = _validate_required_text(priority, 'priority').upper()
        if priority not in {'LOW', 'MEDIUM', 'HIGH', 'URGENT'}:
            raise ValueError('priority must be LOW, MEDIUM, HIGH, or URGENT.')
        ticket.priority = priority
        _flush(self.db)
        return ticket


class ConversationRepository:
    """Database operations related to chatbot conversations."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        customer_id: Optional[int] = None,
        session_id: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Conversation:
        """Store a chatbot conversation message."""
        conversation_id = _validate_required_text(conversation_id, 'conversation_id')
        user_message = _validate_required_text(user_message, 'user_message')
        assistant_response = _validate_required_text(assistant_response, 'assistant_response')
        conversation = Conversation(
            conversation_id=conversation_id,
            customer_id=customer_id,
            session_id=_normalize_optional_text(session_id, 'session_id'),
            user_message=user_message,
            assistant_response=assistant_response,
            intent=_normalize_optional_text(intent, 'intent'),
        )
        self.db.add(conversation)
        _flush(self.db)
        return conversation

    def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation record by conversation ID."""
        statement = select(Conversation).where(
            Conversation.conversation_id == conversation_id
        )
        return self.db.scalar(statement)

    def get_by_customer(self, customer_id: int, limit: int = 50) -> list[Conversation]:
        """Get recent conversations for a customer."""
        limit = _validate_limit(limit)
        statement = (
            select(Conversation)
            .where(Conversation.customer_id == customer_id)
            .order_by(Conversation.created_at.desc(), Conversation.id.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())

    def get_by_session(self, session_id: str, limit: int = 50) -> list[Conversation]:
        """Get conversation history for a session, oldest first."""
        session_id = _validate_required_text(session_id, 'session_id')
        limit = _validate_limit(limit)
        statement = (
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.asc(), Conversation.id.asc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())
