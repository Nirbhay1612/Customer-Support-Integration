
"""
Database seeding script.

Loads sample customers and orders from:
    data/sample_data/customers.json
    data/sample_data/orders.json

Usage:
    python scripts/seed_database.py
    python scripts/seed_database.py --stats
    python scripts/seed_database.py --reset
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import (  # noqa: E402
    Base,
    SessionLocal,
    engine,
    init_db,
)
from src.database.models import Customer, Order  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger(__name__)

SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample_data"
CUSTOMERS_FILE = SAMPLE_DATA_DIR / "customers.json"
ORDERS_FILE = SAMPLE_DATA_DIR / "orders.json"


# ---------------------------------------------------------
# JSON Helpers
# ---------------------------------------------------------

def load_json_file(file_path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON array containing objects."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Sample data file not found: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {file_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Unable to read sample data file: {file_path}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {file_path}."
        )

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Record {index} in {file_path} must be a JSON object."
            )

    return data


def _required_text(value: Any, field: str) -> str:
    """Return validated non-empty text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")

    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    """Normalize an optional string field."""
    if value is None or value == "":
        return None

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null.")

    return value.strip() or None


def _serialize_address(value: Any, field: str) -> str | None:
    """Serialize an address for storage in a Text column."""
    if value is None:
        return None

    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    raise ValueError(
        f"{field} must be a string, JSON object, array, or null."
    )


# ---------------------------------------------------------
# Database Reset
# ---------------------------------------------------------

def reset_database() -> None:
    """
    Drop and recreate registered tables.

    WARNING: This permanently deletes data in those tables.
    """
    logger.warning("Resetting database tables.")

    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError:
        logger.exception("Database reset failed.")
        raise

    logger.info("Database reset completed.")


# ---------------------------------------------------------
# Customer Seeding
# ---------------------------------------------------------

def seed_customers(
    session: Session,
    customers: list[dict[str, Any]],
) -> int:
    """
    Insert or update customers using external_id as the key.

    Returns the number of valid records processed.
    """
    processed = 0

    for index, data in enumerate(customers):
        try:
            external_id = _required_text(
                data.get("external_id"),
                "customer.external_id",
            )
            name = _required_text(
                data.get("name"),
                "customer.name",
            )
            email = _required_text(
                data.get("email"),
                "customer.email",
            ).lower()
            phone = _optional_text(
                data.get("phone"),
                "customer.phone",
            )
        except ValueError as exc:
            logger.warning(
                "Skipping customer record %d: %s",
                index,
                exc,
            )
            continue

        customer = session.scalar(
            select(Customer).where(
                Customer.external_id == external_id
            )
        )

        if customer is None:
            customer = Customer(
                external_id=external_id,
                name=name,
                email=email,
                phone=phone,
            )
            session.add(customer)
            logger.debug("Added customer: %s", external_id)
        else:
            customer.name = name
            customer.email = email
            customer.phone = phone
            logger.debug("Updated customer: %s", external_id)

        processed += 1

    # Assign IDs and validate constraints before seeding orders.
    session.flush()
    return processed


# ---------------------------------------------------------
# Order Seeding
# ---------------------------------------------------------

def seed_orders(
    session: Session,
    orders: list[dict[str, Any]],
) -> int:
    """
    Insert or update orders using external_id as the key.

    Sample order customer_id values are treated as customer
    external IDs, not internal database primary keys.
    """
    processed = 0

    for index, data in enumerate(orders):
        try:
            external_id = _required_text(
                data.get("external_id"),
                "order.external_id",
            )
            customer_external_id = _required_text(
                data.get("customer_id"),
                "order.customer_id",
            )

            raw_amount = data.get("total_amount", "0.00")
            if isinstance(raw_amount, bool):
                raise ValueError(
                    "order.total_amount must be a valid number."
                )

            try:
                total_amount = Decimal(str(raw_amount))
            except (InvalidOperation, ValueError, TypeError) as exc:
                raise ValueError(
                    "order.total_amount must be a valid number."
                ) from exc

            if not total_amount.is_finite() or total_amount < 0:
                raise ValueError(
                    "order.total_amount must be finite and non-negative."
                )

            status = _required_text(
                data.get("status", "PENDING"),
                "order.status",
            ).upper()

            currency = _required_text(
                data.get("currency", "INR"),
                "order.currency",
            ).upper()

            if len(currency) != 3:
                raise ValueError(
                    "order.currency must be a three-letter currency code."
                )

            tracking_number = _optional_text(
                data.get("tracking_number"),
                "order.tracking_number",
            )
            shipping_address = _serialize_address(
                data.get("shipping_address"),
                "order.shipping_address",
            )

        except ValueError as exc:
            logger.warning(
                "Skipping order record %d: %s",
                index,
                exc,
            )
            continue

        customer = session.scalar(
            select(Customer).where(
                Customer.external_id == customer_external_id
            )
        )

        if customer is None:
            logger.warning(
                "Skipping order %s: customer %s was not found.",
                external_id,
                customer_external_id,
            )
            continue

        order = session.scalar(
            select(Order).where(Order.external_id == external_id)
        )

        if order is None:
            order = Order(
                external_id=external_id,
                customer_id=customer.id,
                status=status,
                total_amount=total_amount,
                currency=currency,
                tracking_number=tracking_number,
                shipping_address=shipping_address,
            )
            session.add(order)
            logger.debug("Added order: %s", external_id)
        else:
            order.customer_id = customer.id
            order.status = status
            order.total_amount = total_amount
            order.currency = currency
            order.tracking_number = tracking_number
            order.shipping_address = shipping_address
            logger.debug("Updated order: %s", external_id)

        processed += 1

    session.flush()
    return processed


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def get_database_statistics(session: Session) -> dict[str, int]:
    """Return current customer and order counts."""
    customer_count = session.scalar(
        select(func.count()).select_from(Customer)
    ) or 0

    order_count = session.scalar(
        select(func.count()).select_from(Order)
    ) or 0

    return {
        "customers": int(customer_count),
        "orders": int(order_count),
    }


def print_statistics(statistics: dict[str, int]) -> None:
    """Print database statistics."""
    print("\n" + "=" * 50)
    print("DATABASE STATISTICS")
    print("=" * 50)
    print(f"Customers : {statistics['customers']}")
    print(f"Orders    : {statistics['orders']}")
    print("=" * 50)


# ---------------------------------------------------------
# Seed Database
# ---------------------------------------------------------

def seed_database(reset: bool = False) -> dict[str, int]:
    """
    Initialize and seed the database in one transaction.

    Note: reset=True is destructive and should only be used
    for a database whose existing data can be discarded.
    """
    logger.info("Starting database seeding.")

    # Validate input files before changing database contents.
    customers = load_json_file(CUSTOMERS_FILE)
    orders = load_json_file(ORDERS_FILE)

    # Ensure model tables are registered and available.
    init_db()

    if reset:
        reset_database()

    with SessionLocal() as session:
        try:
            customer_count = seed_customers(session, customers)
            order_count = seed_orders(session, orders)

            # Commit the complete seed operation together.
            session.commit()

            statistics = get_database_statistics(session)

            result = {
                "processed_customers": customer_count,
                "processed_orders": order_count,
                "customers": statistics["customers"],
                "orders": statistics["orders"],
            }

            logger.info(
                "Seeding completed: %d customers processed, "
                "%d orders processed.",
                customer_count,
                order_count,
            )
            return result

        except Exception:
            session.rollback()
            logger.exception("Database seeding failed.")
            raise


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Seed the AI Customer Support Integration System database."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate tables before seeding (destructive).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics without seeding.",
    )

    args = parser.parse_args()

    if args.reset and args.stats:
        parser.error("--reset and --stats cannot be used together.")

    return args


def main() -> int:
    """Run the database seeding command."""
    try:
        args = parse_arguments()

        if args.stats:
            init_db()
            with SessionLocal() as session:
                print_statistics(get_database_statistics(session))
            return 0

        if args.reset:
            print(
                "WARNING: This will permanently delete data "
                "from the configured database tables."
            )
            confirmation = input("Continue? [y/N]: ").strip().lower()

            if confirmation not in {"y", "yes"}:
                print("Database reset cancelled.")
                return 0

        result = seed_database(reset=args.reset)

        print("\n" + "=" * 50)
        print("DATABASE SEEDING COMPLETED")
        print("=" * 50)
        print(f"Processed customers : {result['processed_customers']}")
        print(f"Processed orders    : {result['processed_orders']}")
        print(f"Total customers     : {result['customers']}")
        print(f"Total orders        : {result['orders']}")
        print("=" * 50)

        return 0

    except KeyboardInterrupt:
        print("\nDatabase seeding interrupted.")
        return 130

    except Exception as exc:
        logger.exception("Database seeding failed.")
        print(f"\nDatabase seeding failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
