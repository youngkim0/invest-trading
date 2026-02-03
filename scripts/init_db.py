"""Initialize database tables."""

import asyncio

from loguru import logger
from sqlalchemy import text

from config import get_settings
from data.storage.models import Base
from data.storage.repository import DatabaseManager


async def init_database():
    """Create all database tables."""
    logger.info("Initializing database...")

    db_manager = DatabaseManager()

    # Create all tables
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created successfully")

    # Enable TimescaleDB extension and create hypertables
    async with db_manager.session() as session:
        try:
            # Enable TimescaleDB extension
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            await session.commit()
            logger.info("TimescaleDB extension enabled")

            # Create hypertables for time-series data
            hypertables = [
                ("ohlcv", "timestamp"),
                ("tick_data", "timestamp"),
                ("performance_snapshots", "timestamp"),
            ]

            for table, time_column in hypertables:
                try:
                    await session.execute(
                        text(f"""
                            SELECT create_hypertable(
                                '{table}',
                                '{time_column}',
                                if_not_exists => TRUE
                            );
                        """)
                    )
                    await session.commit()
                    logger.info(f"Hypertable created for {table}")
                except Exception as e:
                    logger.warning(f"Could not create hypertable for {table}: {e}")
                    await session.rollback()

        except Exception as e:
            logger.warning(f"TimescaleDB setup failed (may not be installed): {e}")
            await session.rollback()

    logger.info("Database initialization complete")


async def drop_all_tables():
    """Drop all database tables (use with caution)."""
    logger.warning("Dropping all database tables...")

    db_manager = DatabaseManager()

    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.info("All tables dropped")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        response = input("Are you sure you want to drop all tables? (yes/no): ")
        if response.lower() == "yes":
            asyncio.run(drop_all_tables())
        else:
            print("Aborted")
    else:
        asyncio.run(init_database())
