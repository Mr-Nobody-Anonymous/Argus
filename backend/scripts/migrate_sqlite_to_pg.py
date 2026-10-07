#!/usr/bin/env python
"""
Migration utility: Transfer cameras, zones, and events from SQLite to PostgreSQL.
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def migrate(sqlite_path: Path, pg_url: str):
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        logger.error("psycopg2 is required for PostgreSQL migration. Run: pip install psycopg2-binary")
        sys.exit(1)

    if not sqlite_path.is_file():
        logger.error(f"SQLite file not found at: {sqlite_path}")
        sys.exit(1)

    logger.info(f"Connecting to SQLite ({sqlite_path})...")
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row

    logger.info("Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(pg_url)
    pg_conn.autocommit = False

    tables = ["cameras", "zones", "events", "known_faces", "behavior_profiles", "audit_log"]

    try:
        with pg_conn.cursor() as pg_cur:
            for table in tables:
                # Check if table exists in SQLite
                cur = sqlite_conn.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
                )
                if not cur.fetchone():
                    logger.info(f"Table '{table}' does not exist in SQLite - skipping.")
                    continue

                rows = sqlite_conn.execute(f"SELECT * FROM {table}").fetchall()
                if not rows:
                    logger.info(f"Table '{table}' has 0 rows.")
                    continue

                cols = [desc[0] for desc in sqlite_conn.execute(f"SELECT * FROM {table} LIMIT 1").description]
                cols_str = ", ".join(cols)
                placeholders = ", ".join(["%s"] * len(cols))

                logger.info(f"Migrating {len(rows)} records for table '{table}'...")
                insert_query = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

                records = [tuple(r[col] for col in cols) for r in rows]
                psycopg2.extras.execute_batch(pg_cur, insert_query, records)
                logger.info(f"✓ Table '{table}' migrated successfully.")

        pg_conn.commit()
        logger.info("Migration complete! All data successfully transferred.")
    except Exception as exc:
        pg_conn.rollback()
        logger.error(f"Migration failed and was rolled back: {exc}")
        sys.exit(1)
    finally:
        sqlite_conn.close()
        pg_conn.close()


def main():
    p = argparse.ArgumentParser(description="Migrate Argus SQLite database to PostgreSQL.")
    p.add_argument("--sqlite-path", default="data/argus.db", type=Path, help="Path to SQLite database")
    p.add_argument("--pg-url", required=True, help="PostgreSQL connection string (e.g. postgresql://user:pass@host:5432/db)")
    args = p.parse_args()

    migrate(args.sqlite_path, args.pg_url)


if __name__ == "__main__":
    main()
