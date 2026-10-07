"""
Database schema and initialization for Argus
"""

# cspell:words rtsp argus
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = "../data/argus.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.lock = threading.Lock()
        self.initialize()

    def initialize(self):
        """Initialize database with schema"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # SQLite only enforces declared foreign keys when this is enabled on the
        # connection. It makes the schema's ON DELETE CASCADE effective.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()
        logger.info(f"Database initialized at {self.db_path}")

    def _create_tables(self):
        """Create all database tables"""
        cursor = self.conn.cursor()

        # Cameras table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                location_tag TEXT,
                rtsp_url TEXT NOT NULL UNIQUE,
                status TEXT DEFAULT 'offline',
                fps REAL DEFAULT 0,
                last_frame_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Zones table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'polygon',
                coordinates TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            )
        """)

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                rule_type TEXT NOT NULL,
                object_type TEXT,
                confidence REAL,
                bbox TEXT,
                snapshot_path TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'detected',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- Lifecycle: who looked at this event and when. Without these
                -- an operator cannot prove an alert was ever reviewed, which
                -- is the whole point of an audit-capable surveillance system.
                track_id INTEGER,
                acknowledged_by TEXT,
                acknowledged_at TIMESTAMP,
                resolved_by TEXT,
                resolved_at TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            )
        """)

        # Migrate pre-existing databases: ALTER TABLE ADD COLUMN is a no-op on
        # a fresh schema but is required for a database created before the
        # lifecycle columns existed. Checked per column so a partial migration
        # can be completed rather than aborting the whole run.
        existing_event_cols = {
            row[1] for row in cursor.execute("PRAGMA table_info(events)")
        }
        for column, ddl in (
            ("track_id", "INTEGER"),
            ("acknowledged_by", "TEXT"),
            ("acknowledged_at", "TIMESTAMP"),
            ("resolved_by", "TEXT"),
            ("resolved_at", "TIMESTAMP"),
        ):
            if column not in existing_event_cols:
                cursor.execute(f"ALTER TABLE events ADD COLUMN {column} {ddl}")

        # 'new' predates the lifecycle and matches no transition rule, so any
        # event still holding it could never be acknowledged.
        cursor.execute(
            "UPDATE events SET status = 'detected' WHERE status = 'new'"
        )

        # Create indexes for fast queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_rule ON events(rule_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_composite ON events(camera_id, timestamp DESC)"
        )

        # Behavior profiles table for adaptive learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavior_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL UNIQUE,
                patterns TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Known faces table for face recognition.
        # Created here (not lazily inside FaceRecognition) so that the
        # /api/v1/faces endpoints work even when the face-recognition models
        # fail to initialise or the optional deps are missing.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding TEXT NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()
        logger.info("Database tables created successfully")

    def execute(self, query: str, params: tuple = ()):
        """Execute a query and return cursor"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return cursor

    def fetchone(self, query: str, params: tuple = ()):
        """Fetch one result"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetchall(self, query: str, params: tuple = ()):
        """Fetch all results"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Global database instance
db = None


def get_db():
    """Get database instance (SQLite or PostgreSQL based on configuration)"""
    global db
    if db is None:
        from backend.config.config import get_config
        config = get_config()
        db_url = os.environ.get("DATABASE_URL", "").strip() or config.database.url

        if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
            from backend.database.postgres import PostgresDatabase
            db = PostgresDatabase(db_url)
            return db

        # Parse path from URL (remove sqlite:/// prefix)
        db_path = Path(db_url.replace("sqlite:///", ""))
        # Anchor relative paths to the project root so the DB lands in the same
        # place no matter which directory the process was launched from.
        if not db_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = (project_root / db_path).resolve()
        db = Database(str(db_path))
    return db


def close_db():
    """Close database connection"""
    global db
    if db:
        db.close()
        db = None
