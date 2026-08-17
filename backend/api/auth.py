"""
Authentication and authorization for the Argus API.

Design decisions
----------------
1. **Single user store.** Users live in the Django ``auth_user`` table inside
   the shared ``data/argus.db``. The Django admin (``backend/scripts/run_admin.py``)
   is therefore the user-management UI for free, and there is exactly one
   password hash format to reason about. No parallel user table is created.

2. **No Django import at request time.** Django's PBKDF2-SHA256 hashes are
   verified with stdlib ``hashlib``, so the API process does not need Django
   loaded (or configured) to authenticate. Django remains an optional admin
   dependency, not a runtime dependency of the API.

3. **Roles are derived, not invented.** Django groups named ``admin`` /
   ``operator`` / ``viewer`` map to Argus roles. A superuser is always an admin.
   A staff user with no group defaults to operator. Everyone else is a viewer.
   This means permissions are managed in the admin panel, not in code.

Roles
-----
    admin     manage users, cameras, zones, identities, system config
    operator  view cameras, acknowledge events, manage zones, start/stop streams
    viewer    view cameras, view events
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config.config import resolve_path

logger = logging.getLogger(__name__)

# ── Roles ────────────────────────────────────────────────────────────────────

ROLE_ADMIN = "admin"
ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"

# Ordered least → most privileged. A role satisfies a requirement if its rank
# is >= the required rank, so require_role("operator") also admits admins.
_ROLE_RANK: Dict[str, int] = {
    ROLE_VIEWER: 0,
    ROLE_OPERATOR: 1,
    ROLE_ADMIN: 2,
}

ALL_ROLES = tuple(_ROLE_RANK)


def role_rank(role: str) -> int:
    return _ROLE_RANK.get(role, -1)


# ── JWT configuration ────────────────────────────────────────────────────────

_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_TTL = timedelta(minutes=int(os.environ.get("ARGUS_ACCESS_TOKEN_MINUTES", "30")))
_REFRESH_TOKEN_TTL = timedelta(days=int(os.environ.get("ARGUS_REFRESH_TOKEN_DAYS", "7")))

# Set when the server generates an ephemeral development key, so /health and
# startup can warn loudly instead of silently accepting throwaway tokens.
EPHEMERAL_SECRET_IN_USE = False


def _load_jwt_secret() -> str:
    """
    Resolve the JWT signing key.

    Production MUST supply ARGUS_JWT_SECRET. If it is absent we generate a
    random ephemeral key so development still works, but every token becomes
    invalid on restart and the condition is surfaced loudly — a fixed default
    secret would be worse than no auth at all, because it looks secure.
    """
    global EPHEMERAL_SECRET_IN_USE
    env_secret = os.environ.get("ARGUS_JWT_SECRET", "").strip()
    if env_secret:
        if len(env_secret) < 32:
            logger.warning(
                "ARGUS_JWT_SECRET is shorter than 32 characters; use a longer random value."
            )
        return env_secret

    EPHEMERAL_SECRET_IN_USE = True
    logger.warning(
        "ARGUS_JWT_SECRET is not set - generating an EPHEMERAL signing key. "
        "All tokens will be invalidated on restart. Set ARGUS_JWT_SECRET before deploying."
    )
    return secrets.token_urlsafe(48)


_JWT_SECRET = _load_jwt_secret()


# ── Django password verification (stdlib only) ───────────────────────────────

def verify_django_password(raw_password: str, encoded: str) -> bool:
    """
    Verify a password against a Django password hash.

    Supports Django's default ``pbkdf2_sha256`` and the ``unsalted_md5`` /
    ``md5`` legacy formats. Comparison is constant-time. Unknown algorithms
    return False rather than raising, so a malformed row cannot authenticate.
    """
    if not encoded or not raw_password:
        return False

    parts = encoded.split("$")
    algorithm = parts[0]

    try:
        if algorithm == "pbkdf2_sha256":
            _, iterations, salt, digest = parts
            expected = hashlib.pbkdf2_hmac(
                "sha256", raw_password.encode(), salt.encode(), int(iterations)
            )
            return hmac.compare_digest(base64.b64encode(expected).decode(), digest)

        if algorithm == "pbkdf2_sha1":
            _, iterations, salt, digest = parts
            expected = hashlib.pbkdf2_hmac(
                "sha1", raw_password.encode(), salt.encode(), int(iterations)
            )
            return hmac.compare_digest(base64.b64encode(expected).decode(), digest)

        if algorithm == "md5" and len(parts) == 3:
            _, salt, digest = parts
            expected = hashlib.md5((salt + raw_password).encode()).hexdigest()
            return hmac.compare_digest(expected, digest)
    except (ValueError, TypeError) as exc:
        logger.warning(f"Malformed password hash encountered: {exc}")
        return False

    logger.warning(f"Unsupported password hash algorithm: {algorithm!r}")
    return False


# ── User lookup ──────────────────────────────────────────────────────────────

@dataclass
class AuthUser:
    id: int
    username: str
    role: str
    is_superuser: bool
    is_staff: bool
    email: str = ""

    def has_role(self, required: str) -> bool:
        return role_rank(self.role) >= role_rank(required)


def _db_path() -> str:
    return str(resolve_path("data/argus.db"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    # SQLite defaults foreign_keys to OFF per connection. Without this, deleting
    # a user leaves its auth_user_groups row behind as a dangling role grant.
    try:
        conn.execute("PRAGMA foreign_keys = ON")
    except sqlite3.Error:  # pragma: no cover - older/locked SQLite builds
        logger.debug("Could not enable foreign_keys pragma")
    return conn


def _derive_role(conn: sqlite3.Connection, user_row: sqlite3.Row) -> str:
    """Map Django superuser/staff/group membership onto an Argus role."""
    if user_row["is_superuser"]:
        return ROLE_ADMIN

    try:
        rows = conn.execute(
            """
            SELECT g.name FROM auth_group g
            JOIN auth_user_groups ug ON ug.group_id = g.id
            -- Re-join auth_user so a grant whose user no longer exists can
            -- never be honoured, even if the row was orphaned while foreign
            -- keys were disabled (they are OFF by default in SQLite).
            JOIN auth_user u ON u.id = ug.user_id
            WHERE ug.user_id = ? AND u.is_active = 1
            """,
            (user_row["id"],),
        ).fetchall()
        group_names = {r["name"].strip().lower() for r in rows}
    except sqlite3.Error:
        group_names = set()

    # Most privileged group wins.
    for role in (ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER):
        if role in group_names:
            return role

    # Staff with no explicit group can operate but not administer.
    return ROLE_OPERATOR if user_row["is_staff"] else ROLE_VIEWER


def authenticate_user(username: str, password: str) -> Optional[AuthUser]:
    """Return an AuthUser when credentials are valid and the account is active."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT id, username, password, is_superuser, is_staff, is_active, email "
                "FROM auth_user WHERE username = ?",
                (username,),
            ).fetchone()

            if row is None:
                # Equalise timing against the hash-verification path so the
                # response time does not reveal whether the user exists.
                hashlib.pbkdf2_hmac("sha256", b"timing", b"equaliser", 1_000_000)
                return None

            if not row["is_active"]:
                logger.info(f"Login rejected for inactive account: {username!r}")
                return None

            if not verify_django_password(password, row["password"]):
                return None

            return AuthUser(
                id=row["id"],
                username=row["username"],
                role=_derive_role(conn, row),
                is_superuser=bool(row["is_superuser"]),
                is_staff=bool(row["is_staff"]),
                email=row["email"] or "",
            )
    except sqlite3.Error as exc:
        logger.error(f"Authentication database error: {exc}")
        return None


def get_user_by_id(user_id: int) -> Optional[AuthUser]:
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT id, username, is_superuser, is_staff, is_active, email "
                "FROM auth_user WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None or not row["is_active"]:
                return None
            return AuthUser(
                id=row["id"],
                username=row["username"],
                role=_derive_role(conn, row),
                is_superuser=bool(row["is_superuser"]),
                is_staff=bool(row["is_staff"]),
                email=row["email"] or "",
            )
    except sqlite3.Error as exc:
        logger.error(f"User lookup database error: {exc}")
        return None


# ── Token issue / verify ─────────────────────────────────────────────────────

def create_token(user: AuthUser, token_type: str = "access") -> str:
    ttl = _ACCESS_TOKEN_TTL if token_type == "access" else _REFRESH_TOKEN_TTL
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "type": token_type,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_token(token: str, expected_type: str = "access") -> Optional[dict]:
    """Decode and validate a token. Returns None for any invalid token."""
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.debug("Token rejected: expired")
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug(f"Token rejected: {exc}")
        return None

    if payload.get("type") != expected_type:
        logger.debug("Token rejected: wrong token type")
        return None
    return payload


def token_to_user(token: str, expected_type: str = "access") -> Optional[AuthUser]:
    """
    Resolve a token to a *live* user record.

    The role is re-read from the database rather than trusted from the token,
    so revoking a user or changing their group takes effect immediately instead
    of at token expiry.
    """
    payload = decode_token(token, expected_type)
    if payload is None:
        return None
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        return None
    return get_user_by_id(user_id)


# ── Login throttling ─────────────────────────────────────────────────────────

_MAX_ATTEMPTS = int(os.environ.get("ARGUS_LOGIN_MAX_ATTEMPTS", "5"))
_LOCKOUT_SECONDS = int(os.environ.get("ARGUS_LOGIN_LOCKOUT_SECONDS", "300"))
_failed_attempts: Dict[str, List[float]] = {}


def _prune(key: str, now: float) -> List[float]:
    attempts = [t for t in _failed_attempts.get(key, []) if now - t < _LOCKOUT_SECONDS]
    if attempts:
        _failed_attempts[key] = attempts
    else:
        _failed_attempts.pop(key, None)
    return attempts


def is_locked_out(key: str) -> Tuple[bool, int]:
    """Return (locked, seconds_remaining) for a throttle key."""
    now = time.time()
    attempts = _prune(key, now)
    if len(attempts) >= _MAX_ATTEMPTS:
        return True, int(_LOCKOUT_SECONDS - (now - attempts[0]))
    return False, 0


def record_failed_attempt(key: str) -> None:
    now = time.time()
    _prune(key, now)
    _failed_attempts.setdefault(key, []).append(now)


def clear_failed_attempts(key: str) -> None:
    _failed_attempts.pop(key, None)


# ── FastAPI dependencies ─────────────────────────────────────────────────────

# auto_error=False so a missing header yields our own 401 with a WWW-Authenticate
# challenge rather than FastAPI's bare 403.
_bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def auth_enabled() -> bool:
    """
    Authentication is on by default and can only be disabled explicitly.

    ARGUS_DISABLE_AUTH=1 exists for local single-user development. It is
    reported by /health and logged at startup so it can never be on by accident.
    """
    return os.environ.get("ARGUS_DISABLE_AUTH", "").strip().lower() not in ("1", "true", "yes")


_DEV_USER = AuthUser(
    id=0, username="dev-auth-disabled", role=ROLE_ADMIN,
    is_superuser=True, is_staff=True, email="",
)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> AuthUser:
    if not auth_enabled():
        return _DEV_USER
    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED
    user = token_to_user(credentials.credentials, expected_type="access")
    if user is None:
        raise _UNAUTHENTICATED
    return user


def require_role(required: str):
    """
    Build a dependency enforcing a minimum role.

    Usage::

        @app.post("/api/v1/cameras", dependencies=[Depends(require_role(ROLE_ADMIN))])
    """

    async def _dependency(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if not user.has_role(required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{required}' role; authenticated as '{user.role}'",
            )
        return user

    return _dependency


require_admin = require_role(ROLE_ADMIN)
require_operator = require_role(ROLE_OPERATOR)
require_viewer = require_role(ROLE_VIEWER)


# ── WebSocket authentication ─────────────────────────────────────────────────

async def authenticate_websocket(websocket, required: str = ROLE_VIEWER) -> Optional[AuthUser]:
    """
    Authenticate a WebSocket *before* accepting the connection.

    Browsers cannot set headers on a WebSocket handshake, so the token is read
    from the ``token`` query parameter (falling back to the Authorization header
    and the ``Sec-WebSocket-Protocol`` header for non-browser clients).

    Returns the user, or None after closing the socket with a policy-violation
    code. Callers must abort when None is returned.
    """
    if not auth_enabled():
        return _DEV_USER

    token = websocket.query_params.get("token")

    if not token:
        header = websocket.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()

    if not token:
        proto = websocket.headers.get("sec-websocket-protocol", "")
        if proto.strip():
            token = proto.split(",")[0].strip()

    user = token_to_user(token, expected_type="access") if token else None

    if user is None:
        # 1008 = policy violation. Close before accept() so no frame is ever sent.
        await websocket.close(code=1008, reason="Unauthorized")
        return None

    if not user.has_role(required):
        await websocket.close(code=1008, reason="Forbidden")
        return None

    return user


def client_key(request: Request, username: str = "") -> str:
    """Throttle key combining client IP and username."""
    client_host = request.client.host if request.client else "unknown"
    return f"{client_host}:{username.lower()}"
