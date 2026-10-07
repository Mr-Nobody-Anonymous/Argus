"""
PostgreSQL database adapter for Argus.
Enables high-throughput multi-camera deployments and concurrent writes.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PostgresDatabase:
    """
    PostgreSQL backend adapter supporting parameterized queries and connection pooling.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.lock = threading.Lock()
        self.conn = None
        self._init_connection()
        self._create_tables()

    def _init_connection(self):
        try:
            import psycopg2
            import psycopg2.extras
            self.conn = psycopg2.connect(self.dsn)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL database successfully.")
        except ImportError:
            logger.warning(
                "psycopg2 is not installed. To use PostgreSQL with Argus, install: "
                "pip install psycopg2-binary"
            )
            raise
        except Exception as exc:
            logger.error(f"Failed to connect to PostgreSQL ({self.dsn}): {exc}")
            raise

    def _create_tables(self):
        """Create PostgreSQL tables if they do not exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS cameras (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            location_tag VARCHAR(255),
            rtsp_url TEXT NOT NULL UNIQUE,
            status VARCHAR(50) DEFAULT 'offline',
            fps REAL DEFAULT 0,
            last_frame_time TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS zones (
            id SERIAL PRIMARY KEY,
            camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(50) DEFAULT 'polygon',
            coordinates TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS events (
            id SERIAL PRIMARY KEY,
            camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
            timestamp TIMESTAMP NOT NULL,
            rule_type VARCHAR(100) NOT NULL,
            object_type VARCHAR(100),
            confidence REAL,
            bbox TEXT,
            snapshot_path TEXT,
            priority VARCHAR(50) DEFAULT 'medium',
            status VARCHAR(50) DEFAULT 'detected',
            metadata JSONB,
            track_id INTEGER,
            acknowledged_by VARCHAR(255),
            acknowledged_at TIMESTAMP,
            resolved_by VARCHAR(255),
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS known_faces (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            encoding TEXT NOT NULL,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS behavior_profiles (
            id SERIAL PRIMARY KEY,
            person_id VARCHAR(255) NOT NULL UNIQUE,
            patterns TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_pg_events_camera ON events(camera_id);
        CREATE INDEX IF NOT EXISTS idx_pg_events_timestamp ON events(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_pg_events_status ON events(status);
        """
        try:
            with self.lock:
                with self.conn.cursor() as cur:
                    cur.execute(ddl)
            logger.info("PostgreSQL schema validated/created.")
        except Exception as exc:
            logger.error(f"Error creating PostgreSQL schema: {exc}")

    def execute(self, query: str, params: tuple = ()):
        # Convert SQLite ? placeholders to PostgreSQL %s
        pg_query = query.replace("?", "%s")
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(pg_query, params)
            return cur

    def fetchone(self, query: str, params: tuple = ()):
        import psycopg2.extras
        pg_query = query.replace("?", "%s")
        with self.lock:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(pg_query, params)
                return cur.fetchone()

    def fetchall(self, query: str, params: tuple = ()):
        import psycopg2.extras
        pg_query = query.replace("?", "%s")
        with self.lock:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(pg_query, params)
                return cur.fetchall()

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("PostgreSQL connection closed.")
