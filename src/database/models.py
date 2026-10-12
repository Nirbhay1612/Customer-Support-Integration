"""
SQLAlchemy database models for the AI Customer Support Integration System.

Defines the core persistent entities:
- Customer
- Order
- Ticket
- Conversation
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base


class Customer(Base):
    """Customer account information."""

    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint(
            "length(trim(name)) > 0",
            name="ck_customers_name_not_empty",
        ),
        CheckConstraint(
            "length(trim(email)) > 0",
            name="ck_customers_email_not_empty",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    external_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=True,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    phone: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    orders: Mapped[list[Order]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    tickets: Mapped[list[Ticket]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Customer(id={self.id}, "
            f"name={self.name!r}, "
            f"email={self.email!r})>"
        )


class Order(Base):
    """Customer order information."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint(
            "total_amount >= 0",
            name="ck_orders_total_amount_nonnegative",
        ),
        CheckConstraint(
            "length(trim(external_id)) > 0",
            name="ck_orders_external_id_not_empty",
        ),
        Index(
            "ix_orders_customer_status",
            "customer_id",
            "status",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    external_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        index=True,
        nullable=False,
    )

    # Decimal-backed storage avoids binary floating-point
    # rounding errors for monetary values.
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        default="INR",
        nullable=False,
    )

    tracking_number: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    shipping_address: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    customer: Mapped[Customer] = relationship(
        back_populates="orders",
    )

    def __repr__(self) -> str:
        return (
            f"<Order(id={self.id}, "
            f"external_id={self.external_id!r}, "
            f"status={self.status!r})>"
        )


class Ticket(Base):
    """Customer support ticket."""

    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint(
            "length(trim(subject)) > 0",
            name="ck_tickets_subject_not_empty",
        ),
        CheckConstraint(
            "length(trim(description)) > 0",
            name="ck_tickets_description_not_empty",
        ),
        Index(
            "ix_tickets_customer_status",
            "customer_id",
            "status",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    external_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=True,
    )

    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    subject: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="OPEN",
        index=True,
        nullable=False,
    )

    priority: Mapped[str] = mapped_column(
        String(20),
        default="MEDIUM",
        index=True,
        nullable=False,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    customer: Mapped[Customer] = relationship(
        back_populates="tickets",
    )

    def __repr__(self) -> str:
        return (
            f"<Ticket(id={self.id}, "
            f"subject={self.subject!r}, "
            f"status={self.status!r})>"
        )


class Conversation(Base):
    """Persisted chatbot conversation messages."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index(
            "ix_conversations_customer_created",
            "customer_id",
            "created_at",
        ),
        Index(
            "ix_conversations_session_created",
            "session_id",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    conversation_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    customer_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )

    session_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        index=True,
        nullable=True,
    )

    user_message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    assistant_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    intent: Mapped[Optional[str]] = mapped_column(
        String(100),
        index=True,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    customer: Mapped[Optional[Customer]] = relationship(
        back_populates="conversations",
    )

    def __repr__(self) -> str:
        return (
            f"<Conversation("
            f"id={self.id}, "
            f"conversation_id={self.conversation_id!r}, "
            f"intent={self.intent!r})>"
        )
