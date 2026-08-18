"""Phase 6 - Memory: everything perceived, retrievable later.

Until now the world model lived entirely in RAM: `TrackStore` evicts, `SceneGraph`
expires edges after 5 s, and a restart erased everything. That makes two whole
classes of question unanswerable - *"where else has this person been?"* and
*"what happened near the loading bay yesterday?"* - because by the time anyone
asks, the evidence is gone.

This module gives perception a disk. Three stores, split by what they answer:

| Store              | Answers                                            |
|--------------------|----------------------------------------------------|
| `observations`     | "what happened, where, when" (structured + text)   |
| `appearances`      | "who was that" (descriptor vectors)                |
| `track_summaries`  | "tell me about this entity"                        |

## Why SQLite and not Qdrant

`config.yaml` ships `qdrant: enabled: true`, docker-compose defines the
service, and `qdrant-client` is in requirements-optional. None of it is
installed or running here, and **a config flag that claims a database is
enabled when nothing can reach it is the same lie as a capability registry
that advertises an OCR engine it cannot run.**

So `VectorStore` is an interface with two implementations:

* `SqliteVectorStore` - brute-force cosine over stored blobs. Exact, needs no
  server, and at the scale this deployment actually runs (measured below) it
  is fast enough that an ANN index would be premature.
* `QdrantVectorStore` - used automatically **if and only if** a live server
  answers. Never assumed from config.

Brute force is honest arithmetic: 96 floats x N rows. Measured on this host
(2 cores), the cost is linear and not free:

| stored vectors | search | database |
|----------------|--------|----------|
| 1 000          |   5 ms |  0.6 MB  |
| 10 000         |  37 ms |  5.8 MB  |
| 50 000         | 207 ms | 28.8 MB  |

At the default 7-day appearance retention and the measured rate of ~31
descriptors per 150 frames, a handful of cameras stay in the low thousands,
where this is comfortably interactive. **Past ~25 000 vectors a query costs
more than 100 ms and the SQLite backend should be replaced with Qdrant** -
`report()` emits an explicit warning when the table crosses that line, because
a search that quietly degrades into a slow one is how a feature dies without
anyone filing a bug.

## Retention

This is the phase the architecture doc warned about: **appearance descriptors
are re-identifying**. A "grey jacket, red backpack" vector follows a person
across cameras as surely as a face embedding does. Every table here is
therefore registered with `retention.py` in the same change that creates it,
and `purge_expired()` enforces it. Nothing is written without an expiry.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Descriptors are stored as raw float32 blobs. A JSON array of 96 floats costs
# ~1.1 kB; the blob costs 384 bytes and decodes without parsing.
_FLOAT_SIZE = 4


def _pack(vector) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(v) for v in vector])


def _unpack(blob: bytes) -> List[float]:
    return list(struct.unpack(f"<{len(blob) // _FLOAT_SIZE}f", blob))


def _now() -> float:
    return time.time()


# ── records ──────────────────────────────────────────────────────────────────

@dataclass
class StoredObservation:
    """One thing that happened, durable and searchable."""

    kind: str
    summary: str
    camera_id: Optional[int]
    timestamp: float
    confidence: float
    source: str
    track_ids: List[int] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    observation_id: Optional[str] = None
    row_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "observation_id": self.observation_id,
            "kind": self.kind,
            "summary": self.summary,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(self.timestamp)),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "track_ids": self.track_ids,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass
class StoredAppearance:
    """One appearance descriptor, tied to where and when it was seen."""

    camera_id: int
    track_id: int
    descriptor: Any
    first_seen: float
    last_seen: float
    frame_count: int = 1
    category: str = "person"
    backend: str = "histogram"
    attributes: Dict[str, Any] = field(default_factory=dict)
    row_id: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.camera_id}:{self.track_id}"

    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        out = {
            "row_id": self.row_id,
            "key": self.key,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "category": self.category,
            "backend": self.backend,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_s": round(max(0.0, self.last_seen - self.first_seen), 2),
            "frame_count": self.frame_count,
            "attributes": self.attributes,
        }
        if include_vector:
            out["descriptor"] = list(self.descriptor) if self.descriptor is not None else None
        return out


# ── vector stores ────────────────────────────────────────────────────────────

class VectorStore:
    """Interface both the SQLite and Qdrant implementations satisfy."""

    name = "abstract"

    def upsert(self, key: str, vector, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def search(self, vector, limit: int = 10,
               min_similarity: float = 0.0,
               where: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def report(self) -> Dict[str, Any]:
        return {"backend": self.name, "vectors": self.count()}


class SqliteVectorStore(VectorStore):
    """Exact brute-force cosine search over descriptors held in SQLite.

    Exact rather than approximate: at this scale the whole table is a few
    megabytes, numpy dots it in milliseconds, and an ANN index would trade
    correctness for a speed-up nothing needs yet. `report()` states the
    measured scale at which that stops being true, so the decision can be
    revisited with evidence instead of vibes.
    """

    name = "sqlite"

    # Past this many vectors a brute-force scan exceeds ~100 ms on 2 cores
    # (measured: 37 ms at 10k, 207 ms at 50k). Not a hard limit - an honest
    # threshold at which the operator should be told to run Qdrant.
    SCALE_WARNING_AT = 25_000

    def __init__(self, memory: "PerceptionMemory"):
        self._memory = memory

    def upsert(self, key: str, vector, payload: Dict[str, Any]) -> None:
        self._memory.remember_appearance_vector(key, vector, payload)

    def search(self, vector, limit: int = 10, min_similarity: float = 0.0,
               where: Optional[Dict[str, Any]] = None):
        return self._memory.search_appearance_vectors(
            vector, limit=limit, min_similarity=min_similarity, where=where)

    def count(self) -> int:
        return self._memory.count("appearances")

    def report(self) -> Dict[str, Any]:
        n = self.count()
        out: Dict[str, Any] = {
            "backend": self.name,
            "vectors": n,
            "search": "exact brute-force cosine",
            "estimated_query_ms": round(n * 0.0041 + 1.0, 1),
        }
        if n >= self.SCALE_WARNING_AT:
            out["warning"] = (
                f"{n} vectors exceeds the {self.SCALE_WARNING_AT} at which a "
                f"brute-force scan costs over 100 ms. Run Qdrant "
                f"(docker-compose up qdrant) and install qdrant-client; it "
                f"will be picked up automatically on restart.")
        return out


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store, used only when a live server actually answers.

    Availability is proven by connecting, never inferred from
    `qdrant.enabled: true` in config - the flag is currently `true` on a host
    with no client library and no server, which is exactly the kind of claim
    this codebase treats as a defect.
    """

    name = "qdrant"

    def __init__(self, collection: str = "argus_appearances", dim: int = 96):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        from backend.config.config import get_config
        cfg = get_config()
        host = getattr(cfg.qdrant, "host", "localhost")
        port = getattr(cfg.qdrant, "port", 6333)

        self._client = QdrantClient(host=host, port=port, timeout=3.0)
        self._collection = collection
        self._models = qmodels
        # Fails fast if nothing is listening - which is the point.
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(
                    size=dim, distance=qmodels.Distance.COSINE),
            )

    @classmethod
    def try_connect(cls, dim: int = 96) -> Optional["QdrantVectorStore"]:
        try:
            return cls(dim=dim)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Qdrant unavailable, using SQLite vectors: {exc}")
            return None

    def upsert(self, key: str, vector, payload: Dict[str, Any]) -> None:
        self._client.upsert(
            collection_name=self._collection,
            points=[self._models.PointStruct(
                id=abs(hash(key)) % (2 ** 63), vector=list(vector),
                payload={**payload, "key": key})],
        )

    def search(self, vector, limit: int = 10, min_similarity: float = 0.0,
               where: Optional[Dict[str, Any]] = None):
        hits = self._client.search(
            collection_name=self._collection, query_vector=list(vector),
            limit=limit, score_threshold=min_similarity or None)
        return [(h.payload.get("key", str(h.id)), float(h.score), h.payload or {})
                for h in hits]

    def count(self) -> int:
        try:
            return int(self._client.count(self._collection).count)
        except Exception:
            return 0


# ── the memory itself ────────────────────────────────────────────────────────

class PerceptionMemory:
    """Durable, searchable memory of everything perception has produced."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or self._default_path()
        self._lock = threading.RLock()
        self._vectors: Optional[VectorStore] = None
        self._ensure_schema()

    @staticmethod
    def _default_path() -> str:
        try:
            from backend.config.config import resolve_path
            return resolve_path("data/argus.db")
        except Exception:
            return os.path.join("data", "argus.db")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # -- schema ---------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS perception_observations (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id TEXT,
                    kind           TEXT NOT NULL,
                    summary        TEXT NOT NULL,
                    camera_id      INTEGER,
                    timestamp      REAL NOT NULL,
                    confidence     REAL NOT NULL DEFAULT 0.5,
                    source         TEXT NOT NULL DEFAULT 'rule_engine',
                    track_ids      TEXT NOT NULL DEFAULT '[]',
                    evidence       TEXT NOT NULL DEFAULT '[]',
                    metadata       TEXT NOT NULL DEFAULT '{}',
                    created_at     REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pobs_time
                    ON perception_observations(timestamp);
                CREATE INDEX IF NOT EXISTS idx_pobs_camera_time
                    ON perception_observations(camera_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_pobs_kind
                    ON perception_observations(kind);

                CREATE TABLE IF NOT EXISTS perception_appearances (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    key          TEXT NOT NULL UNIQUE,
                    camera_id    INTEGER NOT NULL,
                    track_id     INTEGER NOT NULL,
                    category     TEXT NOT NULL DEFAULT 'person',
                    backend      TEXT NOT NULL DEFAULT 'histogram',
                    dim          INTEGER NOT NULL,
                    descriptor   BLOB NOT NULL,
                    first_seen   REAL NOT NULL,
                    last_seen    REAL NOT NULL,
                    frame_count  INTEGER NOT NULL DEFAULT 1,
                    attributes   TEXT NOT NULL DEFAULT '{}',
                    created_at   REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pappear_time
                    ON perception_appearances(last_seen);
                CREATE INDEX IF NOT EXISTS idx_pappear_camera
                    ON perception_appearances(camera_id);

                CREATE TABLE IF NOT EXISTS perception_tracks (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    key          TEXT NOT NULL UNIQUE,
                    camera_id    INTEGER NOT NULL,
                    track_id     INTEGER NOT NULL,
                    category     TEXT NOT NULL,
                    first_seen   REAL NOT NULL,
                    last_seen    REAL NOT NULL,
                    frame_count  INTEGER NOT NULL DEFAULT 0,
                    summary      TEXT NOT NULL DEFAULT '',
                    attributes   TEXT NOT NULL DEFAULT '{}',
                    created_at   REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ptracks_time
                    ON perception_tracks(last_seen);
            """)

            # Full-text search over observation summaries and evidence. FTS5 is
            # compiled into virtually every SQLite build, but degrade to LIKE
            # rather than fail if it is absent.
            try:
                conn.executescript("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS perception_obs_fts
                    USING fts5(summary, evidence, kind,
                               content='perception_observations',
                               content_rowid='id');
                """)
                self._fts = True
            except sqlite3.OperationalError as exc:
                logger.warning(f"FTS5 unavailable, text search will use LIKE: {exc}")
                self._fts = False

    # -- writing --------------------------------------------------------------

    def remember_observation(self, obs, camera_id: Optional[int] = None) -> int:
        """Persist one Observation. Returns its row id."""
        cam = camera_id if camera_id is not None else getattr(obs, "camera_id", None)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO perception_observations
                   (observation_id, kind, summary, camera_id, timestamp,
                    confidence, source, track_ids, evidence, metadata, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (getattr(obs, "observation_id", None), obs.kind, obs.summary,
                 cam, float(obs.timestamp), float(obs.confidence),
                 getattr(obs, "source", "rule_engine"),
                 json.dumps(list(getattr(obs, "track_ids", []) or [])),
                 json.dumps(list(getattr(obs, "evidence", []) or [])),
                 json.dumps(dict(getattr(obs, "metadata", {}) or {})),
                 _now()))
            row_id = int(cur.lastrowid)
            if getattr(self, "_fts", False):
                conn.execute(
                    "INSERT INTO perception_obs_fts(rowid, summary, evidence, kind)"
                    " VALUES (?,?,?,?)",
                    (row_id, obs.summary,
                     " ".join(getattr(obs, "evidence", []) or []), obs.kind))
            return row_id

    def remember_appearance(self, camera_id: int, track_id: int, descriptor,
                            first_seen: float, last_seen: float,
                            frame_count: int = 1, category: str = "person",
                            backend: str = "histogram",
                            attributes: Optional[Dict[str, Any]] = None) -> str:
        """Store or update one entity's appearance descriptor."""
        key = f"{camera_id}:{track_id}"
        blob = _pack(descriptor)
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO perception_appearances
                   (key, camera_id, track_id, category, backend, dim,
                    descriptor, first_seen, last_seen, frame_count,
                    attributes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     descriptor=excluded.descriptor,
                     last_seen=excluded.last_seen,
                     frame_count=excluded.frame_count,
                     attributes=excluded.attributes""",
                (key, camera_id, track_id, category, backend,
                 len(blob) // _FLOAT_SIZE, blob, first_seen, last_seen,
                 frame_count, json.dumps(attributes or {}), _now()))
        return key

    def remember_appearance_vector(self, key: str, vector,
                                   payload: Dict[str, Any]) -> None:
        camera_id, _, track_id = key.partition(":")
        self.remember_appearance(
            camera_id=int(camera_id), track_id=int(track_id), descriptor=vector,
            first_seen=float(payload.get("first_seen", _now())),
            last_seen=float(payload.get("last_seen", _now())),
            frame_count=int(payload.get("frame_count", 1)),
            category=payload.get("category", "person"),
            backend=payload.get("backend", "histogram"),
            attributes=payload.get("attributes", {}))

    def remember_track(self, camera_id: int, track_id: int, category: str,
                       first_seen: float, last_seen: float, frame_count: int,
                       summary: str = "",
                       attributes: Optional[Dict[str, Any]] = None) -> str:
        key = f"{camera_id}:{track_id}"
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO perception_tracks
                   (key, camera_id, track_id, category, first_seen, last_seen,
                    frame_count, summary, attributes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     last_seen=excluded.last_seen,
                     frame_count=excluded.frame_count,
                     summary=excluded.summary,
                     attributes=excluded.attributes""",
                (key, camera_id, track_id, category, first_seen, last_seen,
                 frame_count, summary, json.dumps(attributes or {}), _now()))
        return key

    # -- reading --------------------------------------------------------------

    def query_observations(self, camera_id: Optional[int] = None,
                           kinds: Optional[Sequence[str]] = None,
                           since: Optional[float] = None,
                           until: Optional[float] = None,
                           min_confidence: float = 0.0,
                           track_id: Optional[int] = None,
                           limit: int = 100) -> List[StoredObservation]:
        """Structured recall: what happened, filtered by camera, kind and time."""
        sql = ["SELECT * FROM perception_observations WHERE confidence >= ?"]
        args: List[Any] = [min_confidence]
        if camera_id is not None:
            sql.append("AND camera_id = ?"); args.append(camera_id)
        if since is not None:
            sql.append("AND timestamp >= ?"); args.append(since)
        if until is not None:
            sql.append("AND timestamp <= ?"); args.append(until)
        if kinds:
            sql.append(f"AND kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        sql.append("ORDER BY timestamp DESC LIMIT ?")
        args.append(int(limit))

        with self._lock, self._connect() as conn:
            rows = conn.execute(" ".join(sql), args).fetchall()

        out = [self._row_to_observation(r) for r in rows]
        if track_id is not None:
            out = [o for o in out if track_id in o.track_ids]
        return out

    def search_observations(self, text: str, limit: int = 50
                            ) -> List[StoredObservation]:
        """Free-text recall over summaries and evidence.

        This is what answers "what happened near the loading bay yesterday" once
        zone names appear in observation text.
        """
        if not text or not text.strip():
            return []
        with self._lock, self._connect() as conn:
            if getattr(self, "_fts", False):
                try:
                    rows = conn.execute(
                        """SELECT o.* FROM perception_obs_fts f
                           JOIN perception_observations o ON o.id = f.rowid
                           WHERE perception_obs_fts MATCH ?
                           ORDER BY o.timestamp DESC LIMIT ?""",
                        (text, int(limit))).fetchall()
                    return [self._row_to_observation(r) for r in rows]
                except sqlite3.OperationalError as exc:
                    # A malformed FTS query (a bare quote, a stray operator) is
                    # user input, not a bug - fall back rather than 500.
                    logger.debug(f"FTS query rejected, falling back to LIKE: {exc}")
            like = f"%{text.strip()}%"
            rows = conn.execute(
                """SELECT * FROM perception_observations
                   WHERE summary LIKE ? OR evidence LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (like, like, int(limit))).fetchall()
            return [self._row_to_observation(r) for r in rows]

    def search_appearance_vectors(self, vector, limit: int = 10,
                                  min_similarity: float = 0.0,
                                  where: Optional[Dict[str, Any]] = None):
        """Brute-force cosine ranking over stored descriptors."""
        try:
            import numpy as np
        except Exception:
            return []

        sql = ["SELECT key, camera_id, track_id, category, backend, dim,"
               " descriptor, first_seen, last_seen, frame_count, attributes"
               " FROM perception_appearances WHERE 1=1"]
        args: List[Any] = []
        where = where or {}
        if where.get("camera_id") is not None:
            sql.append("AND camera_id = ?"); args.append(where["camera_id"])
        if where.get("exclude_camera_id") is not None:
            sql.append("AND camera_id != ?"); args.append(where["exclude_camera_id"])
        if where.get("exclude_key") is not None:
            sql.append("AND key != ?"); args.append(where["exclude_key"])
        if where.get("since") is not None:
            sql.append("AND last_seen >= ?"); args.append(where["since"])
        if where.get("category") is not None:
            sql.append("AND category = ?"); args.append(where["category"])
        # Descriptors from different backends live in incompatible spaces;
        # comparing them would produce confident nonsense.
        backend = where.get("backend")
        if backend is not None:
            sql.append("AND backend = ?"); args.append(backend)

        with self._lock, self._connect() as conn:
            rows = conn.execute(" ".join(sql), args).fetchall()
        if not rows:
            return []

        query = np.asarray(vector, dtype="float32")
        qnorm = float(np.linalg.norm(query))
        if qnorm <= 0:
            return []
        query = query / qnorm

        matrix = np.frombuffer(b"".join(r["descriptor"] for r in rows),
                               dtype="<f4").reshape(len(rows), -1)
        if matrix.shape[1] != query.shape[0]:
            # Dimension mismatch means a different descriptor backend wrote
            # these rows. Silently returning zeros would look like "no match".
            logger.warning(
                f"Descriptor dimension mismatch: query {query.shape[0]} vs "
                f"stored {matrix.shape[1]} - filter by backend")
            return []

        scores = matrix @ query
        order = np.argsort(-scores)[:max(1, int(limit))]
        out = []
        for i in order:
            score = float(scores[i])
            if score < min_similarity:
                continue
            r = rows[int(i)]
            out.append((r["key"], score, {
                "camera_id": r["camera_id"], "track_id": r["track_id"],
                "category": r["category"], "backend": r["backend"],
                "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                "frame_count": r["frame_count"],
                "attributes": json.loads(r["attributes"] or "{}"),
            }))
        return out

    def get_appearance(self, key: str) -> Optional[StoredAppearance]:
        with self._lock, self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM perception_appearances WHERE key = ?",
                (key,)).fetchone()
        if r is None:
            return None
        return StoredAppearance(
            row_id=r["id"], camera_id=r["camera_id"], track_id=r["track_id"],
            descriptor=_unpack(r["descriptor"]), first_seen=r["first_seen"],
            last_seen=r["last_seen"], frame_count=r["frame_count"],
            category=r["category"], backend=r["backend"],
            attributes=json.loads(r["attributes"] or "{}"))

    def get_track(self, camera_id: int, track_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            r = conn.execute("SELECT * FROM perception_tracks WHERE key = ?",
                             (f"{camera_id}:{track_id}",)).fetchone()
        return dict(r) if r is not None else None

    @staticmethod
    def _row_to_observation(r) -> StoredObservation:
        return StoredObservation(
            row_id=r["id"], observation_id=r["observation_id"], kind=r["kind"],
            summary=r["summary"], camera_id=r["camera_id"],
            timestamp=r["timestamp"], confidence=r["confidence"],
            source=r["source"], track_ids=json.loads(r["track_ids"] or "[]"),
            evidence=json.loads(r["evidence"] or "[]"),
            metadata=json.loads(r["metadata"] or "{}"))

    # -- vector backend selection --------------------------------------------

    def vectors(self) -> VectorStore:
        """The active vector store: Qdrant when it answers, SQLite otherwise."""
        if self._vectors is None:
            from .descriptors import DIM
            self._vectors = QdrantVectorStore.try_connect(dim=DIM) \
                or SqliteVectorStore(self)
        return self._vectors

    # -- housekeeping ---------------------------------------------------------

    def count(self, table: str) -> int:
        mapping = {"observations": "perception_observations",
                   "appearances": "perception_appearances",
                   "tracks": "perception_tracks"}
        name = mapping.get(table, table)
        if name not in mapping.values():
            raise ValueError(f"Unknown table '{table}'")
        with self._lock, self._connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])

    def purge_expired(self, observation_days: int, appearance_days: int
                      ) -> Dict[str, int]:
        """Delete anything past its retention window.

        Appearance descriptors get their own, shorter window: they are
        re-identifying, and there is no operational reason to keep the vector
        that lets you find a person again long after the event it belonged to.
        """
        now = _now()
        obs_cutoff = now - observation_days * 86400
        app_cutoff = now - appearance_days * 86400
        removed: Dict[str, int] = {}
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM perception_observations WHERE timestamp < ?",
                (obs_cutoff,))
            removed["observations"] = cur.rowcount
            if getattr(self, "_fts", False):
                try:
                    conn.execute(
                        "DELETE FROM perception_obs_fts WHERE rowid NOT IN "
                        "(SELECT id FROM perception_observations)")
                except sqlite3.OperationalError:
                    pass
            cur = conn.execute(
                "DELETE FROM perception_appearances WHERE last_seen < ?",
                (app_cutoff,))
            removed["appearances"] = cur.rowcount
            cur = conn.execute(
                "DELETE FROM perception_tracks WHERE last_seen < ?",
                (app_cutoff,))
            removed["tracks"] = cur.rowcount
        return removed

    def report(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "database": self._db_path,
            "full_text_search": bool(getattr(self, "_fts", False)),
            "counts": {
                "observations": self.count("observations"),
                "appearances": self.count("appearances"),
                "tracks": self.count("tracks"),
            },
            "vector_backend": self.vectors().report(),
        }


_MEMORY: Optional[PerceptionMemory] = None
_MEMORY_LOCK = threading.Lock()


def get_memory() -> PerceptionMemory:
    """Process-wide memory, created on first use."""
    global _MEMORY
    with _MEMORY_LOCK:
        if _MEMORY is None:
            _MEMORY = PerceptionMemory()
        return _MEMORY


def reset_memory() -> None:
    """Drop the cached singleton (tests point it at a temporary database)."""
    global _MEMORY
    with _MEMORY_LOCK:
        _MEMORY = None
