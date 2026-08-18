"""
Authentication and authorization tests.

Before this work, all 40 HTTP routes and the video WebSocket were completely
open: anyone who could reach port 8000 could register a face, delete a camera,
or watch a live stream. These tests enforce that this cannot regress.

Run:
    pytest tests/test_api_security.py -v
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "argus.db"

TEST_USERS = {
    "pytest_viewer": ("viewer", "pytest-pass-123"),
    "pytest_operator": ("operator", "pytest-pass-123"),
    "pytest_admin": ("admin", "pytest-pass-123"),
}


def _django_hash(password: str, iterations: int = 100_000) -> str:
    salt = secrets.token_urlsafe(12)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${base64.b64encode(dk).decode()}"


@pytest.fixture(scope="module", autouse=True)
def seeded_users():
    """Create viewer/operator/admin users, then remove them afterwards."""
    if not DB_PATH.exists():
        pytest.skip("argus.db not initialised")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='auth_user'")
        if not cur.fetchone():
            pytest.skip("Django auth tables absent; run backend/scripts/run_admin.py once")

        now = datetime.now().isoformat()
        for group in ("admin", "operator", "viewer"):
            cur.execute("INSERT OR IGNORE INTO auth_group (name) VALUES (?)", (group,))

        for username, (group, password) in TEST_USERS.items():
            # Delete the group membership BEFORE the user. SQLite runs with
            # foreign_keys=OFF by default, so deleting only auth_user leaves the
            # auth_user_groups row behind pointing at a dead id - a stale role
            # grant that would apply to whoever inherits that id.
            cur.execute(
                "DELETE FROM auth_user_groups WHERE user_id IN "
                "(SELECT id FROM auth_user WHERE username = ?)", (username,)
            )
            cur.execute("DELETE FROM auth_user WHERE username = ?", (username,))
            cur.execute(
                """INSERT INTO auth_user
                   (password, last_login, is_superuser, username, last_name,
                    email, is_staff, is_active, date_joined, first_name)
                   VALUES (?, NULL, 0, ?, '', ?, 0, 1, ?, '')""",
                (_django_hash(password), username, f"{username}@test.local", now),
            )
            uid = cur.lastrowid
            gid = cur.execute("SELECT id FROM auth_group WHERE name = ?", (group,)).fetchone()[0]
            cur.execute(
                "INSERT INTO auth_user_groups (user_id, group_id) VALUES (?, ?)", (uid, gid)
            )
        conn.commit()
    finally:
        conn.close()

    yield

    conn = sqlite3.connect(str(DB_PATH))
    try:
        for username in TEST_USERS:
            conn.execute(
                "DELETE FROM auth_user_groups WHERE user_id IN "
                "(SELECT id FROM auth_user WHERE username = ?)", (username,)
            )
            conn.execute("DELETE FROM auth_user WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


# ── Password hashing ─────────────────────────────────────────────────────────

class TestPasswordVerification:
    def test_accepts_correct_django_hash(self):
        from backend.api.auth import verify_django_password
        encoded = _django_hash("correct-horse")
        assert verify_django_password("correct-horse", encoded)

    def test_rejects_wrong_password(self):
        from backend.api.auth import verify_django_password
        assert not verify_django_password("wrong", _django_hash("correct-horse"))

    def test_rejects_malformed_hash(self):
        from backend.api.auth import verify_django_password
        for bad in ("", "garbage", "unknown_algo$1$2$3", "pbkdf2_sha256$notanint$s$d"):
            assert not verify_django_password("x", bad), f"Accepted malformed hash: {bad!r}"

    def test_rejects_empty_password(self):
        from backend.api.auth import verify_django_password
        assert not verify_django_password("", _django_hash("something"))


# ── Roles ────────────────────────────────────────────────────────────────────

class TestRoleResolution:
    def test_group_membership_maps_to_role(self):
        from backend.api.auth import authenticate_user
        for username, (expected_role, password) in TEST_USERS.items():
            user = authenticate_user(username, password)
            assert user is not None, f"Could not authenticate {username}"
            assert user.role == expected_role

    def test_role_hierarchy(self):
        from backend.api.auth import AuthUser, ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER

        def mk(role):
            return AuthUser(id=1, username="u", role=role, is_superuser=False, is_staff=False)

        assert mk(ROLE_ADMIN).has_role(ROLE_VIEWER)
        assert mk(ROLE_ADMIN).has_role(ROLE_ADMIN)
        assert mk(ROLE_OPERATOR).has_role(ROLE_VIEWER)
        assert not mk(ROLE_OPERATOR).has_role(ROLE_ADMIN)
        assert not mk(ROLE_VIEWER).has_role(ROLE_OPERATOR)
        assert not mk(ROLE_VIEWER).has_role(ROLE_ADMIN)

    def test_inactive_account_cannot_log_in(self):
        from backend.api.auth import authenticate_user

        conn = sqlite3.connect(str(DB_PATH))
        try:
            conn.execute("UPDATE auth_user SET is_active = 0 WHERE username = ?", ("pytest_viewer",))
            conn.commit()
            assert authenticate_user("pytest_viewer", "pytest-pass-123") is None
        finally:
            conn.execute("UPDATE auth_user SET is_active = 1 WHERE username = ?", ("pytest_viewer",))
            conn.commit()
            conn.close()

    def test_unknown_user_rejected(self):
        from backend.api.auth import authenticate_user
        assert authenticate_user("no_such_user_here", "whatever") is None


# ── Tokens ───────────────────────────────────────────────────────────────────

class TestTokens:
    def test_roundtrip(self):
        from backend.api.auth import authenticate_user, create_token, token_to_user
        user = authenticate_user("pytest_admin", "pytest-pass-123")
        resolved = token_to_user(create_token(user, "access"), "access")
        assert resolved is not None and resolved.username == "pytest_admin"

    def test_refresh_token_rejected_as_access(self):
        """Token-type confusion must not grant API access."""
        from backend.api.auth import authenticate_user, create_token, token_to_user
        user = authenticate_user("pytest_admin", "pytest-pass-123")
        assert token_to_user(create_token(user, "refresh"), "access") is None

    def test_tampered_token_rejected(self):
        from backend.api.auth import authenticate_user, create_token, token_to_user
        user = authenticate_user("pytest_admin", "pytest-pass-123")
        token = create_token(user)
        assert token_to_user(token[:-4] + "AAAA") is None

    def test_token_signed_with_other_key_rejected(self):
        """A token minted with a different secret must not validate."""
        import jwt
        from backend.api.auth import token_to_user
        forged = jwt.encode(
            {"sub": "1", "username": "admin", "role": "admin", "type": "access"},
            "an-attacker-chosen-secret",
            algorithm="HS256",
        )
        assert token_to_user(forged) is None

    def test_expired_token_rejected(self):
        import jwt
        from datetime import timedelta, timezone
        from backend.api.auth import token_to_user, _JWT_SECRET

        past = datetime.now(timezone.utc) - timedelta(hours=2)
        expired = jwt.encode(
            {
                "sub": "1", "username": "admin", "role": "admin", "type": "access",
                "iat": past, "exp": past + timedelta(minutes=1),
            },
            _JWT_SECRET, algorithm="HS256",
        )
        assert token_to_user(expired) is None

    def test_role_is_read_live_not_from_token(self):
        """
        Revoking privileges must take effect immediately, not at token expiry,
        so the role is re-read from the database on every request.
        """
        import jwt
        from backend.api.auth import token_to_user, _JWT_SECRET, authenticate_user

        user = authenticate_user("pytest_viewer", "pytest-pass-123")
        assert user.role == "viewer"

        # Self-promote inside the token payload.
        forged = jwt.encode(
            {"sub": str(user.id), "username": user.username, "role": "admin", "type": "access"},
            _JWT_SECRET, algorithm="HS256",
        )
        resolved = token_to_user(forged)
        assert resolved is not None
        assert resolved.role == "viewer", (
            "Role was trusted from the token payload - privilege escalation."
        )


# ── Lockout ──────────────────────────────────────────────────────────────────

class TestLoginThrottling:
    def test_lockout_after_repeated_failures(self):
        from backend.api.auth import (
            is_locked_out, record_failed_attempt, clear_failed_attempts, _MAX_ATTEMPTS,
        )
        key = "test-throttle-key"
        clear_failed_attempts(key)
        try:
            assert not is_locked_out(key)[0]
            for _ in range(_MAX_ATTEMPTS):
                record_failed_attempt(key)
            locked, retry_after = is_locked_out(key)
            assert locked and retry_after > 0
        finally:
            clear_failed_attempts(key)

    def test_successful_login_clears_counter(self):
        from backend.api.auth import (
            is_locked_out, record_failed_attempt, clear_failed_attempts,
        )
        key = "test-clear-key"
        record_failed_attempt(key)
        record_failed_attempt(key)
        clear_failed_attempts(key)
        assert not is_locked_out(key)[0]


# ── Route coverage ───────────────────────────────────────────────────────────

class TestRouteProtection:
    """Static guarantee that no sensitive route is left unauthenticated."""

    PUBLIC_ALLOWLIST = {
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/health",
        # The unauthenticated banner moved from "/" to "/api" when the built
        # dashboard took over "/". It exposes only a name, version and a
        # static feature list - no data, no configuration.
        "/api",
        "/",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }

    def _routes(self):
        from backend.api.main import app
        collected = []
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", set()) or set()
            if path:
                collected.append((path, methods, route))
            inner = getattr(route, "original_router", None)
            if inner is not None:
                for sub in inner.routes:
                    sub_path = getattr(sub, "path", None)
                    if sub_path:
                        collected.append((
                            f"/api{sub_path}", getattr(sub, "methods", set()) or set(), sub
                        ))
        return collected

    def _has_auth(self, route) -> bool:
        deps = getattr(getattr(route, "dependant", None), "dependencies", [])
        for dep in deps:
            call = getattr(dep, "call", None)
            name = getattr(call, "__name__", "") or getattr(call, "__qualname__", "")
            if "_dependency" in name or "current_user" in name or "require" in name:
                return True
            if any(
                "_dependency" in (getattr(sub.call, "__qualname__", "") or "")
                for sub in getattr(dep, "dependencies", [])
            ):
                return True
        return False

    def test_no_unprotected_api_routes(self):
        offenders = []
        for path, methods, route in self._routes():
            if path in self.PUBLIC_ALLOWLIST or not path.startswith("/api"):
                continue
            if "websocket" in str(type(route)).lower():
                continue  # asserted separately
            if not self._has_auth(route):
                offenders.append(f"{','.join(sorted(methods)) or 'WS'} {path}")

        assert not offenders, (
            "These API routes have no authentication dependency:\n  "
            + "\n  ".join(sorted(offenders))
        )

    def test_mutating_routes_require_elevated_role(self):
        """No POST/PUT/DELETE should be reachable by a plain viewer."""
        from backend.api.main import app

        weak = []
        for route in app.routes:
            path = getattr(route, "path", "")
            methods = getattr(route, "methods", set()) or set()
            if not path.startswith("/api/v1"):
                continue
            if path in self.PUBLIC_ALLOWLIST:
                continue
            if not (methods & {"POST", "PUT", "PATCH", "DELETE"}):
                continue
            if not self._has_auth(route):
                weak.append(f"{','.join(sorted(methods))} {path}")

        assert not weak, "Unauthenticated mutating routes:\n  " + "\n  ".join(sorted(weak))


# ── Secret hygiene ───────────────────────────────────────────────────────────

class TestSecretHandling:
    def test_no_plaintext_secrets_in_config_yaml(self):
        import re
        config_file = PROJECT_ROOT / "config" / "config.yaml"
        if not config_file.exists():
            pytest.skip("config.yaml missing")

        offenders = []
        for num, line in enumerate(config_file.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = re.match(r"^\s*(\w*(?:password|secret|token|api_key)\w*)\s*:\s*(\S.*)$",
                         stripped, re.IGNORECASE)
            if m:
                value = m.group(2).strip().strip("\"'")
                if value and not value.startswith("${") and value.lower() not in ("null", "~", ""):
                    offenders.append(f"line {num}: {m.group(1)}")

        assert not offenders, (
            "Plaintext secrets in config.yaml (use ${ENV_VAR}):\n  " + "\n  ".join(offenders)
        )

    def test_env_example_documents_jwt_secret(self):
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example is missing"
        assert "ARGUS_JWT_SECRET" in env_example.read_text()

    def test_env_file_is_git_ignored(self):
        gitignore = PROJECT_ROOT / ".gitignore"
        assert gitignore.exists()
        assert ".env" in gitignore.read_text(), ".env must never be committed"


class TestRoleGrantIntegrity:
    """Stale role grants are a privilege-escalation hazard.

    SQLite runs with foreign_keys=OFF unless a connection opts in, so deleting a
    row from auth_user leaves its auth_user_groups row behind, still naming a
    group. If that user id is ever reused, the new account silently inherits the
    dead account's role. The test fixture itself was leaking three such rows per
    run and had accumulated 33, eleven of them granting `admin`.
    """

    def test_no_orphaned_group_memberships(self):
        conn = sqlite3.connect(str(DB_PATH))
        try:
            orphans = conn.execute(
                """SELECT ug.user_id, COALESCE(g.name, '<missing group>')
                     FROM auth_user_groups ug
                     LEFT JOIN auth_group g ON g.id = ug.group_id
                    WHERE ug.user_id NOT IN (SELECT id FROM auth_user)
                       OR ug.group_id NOT IN (SELECT id FROM auth_group)"""
            ).fetchall()
        finally:
            conn.close()

        assert not orphans, (
            f"{len(orphans)} role grant(s) reference a user or group that no "
            f"longer exists: {orphans[:10]}. Whoever inherits one of these ids "
            f"would silently gain that role."
        )

    def test_no_duplicate_group_memberships(self):
        conn = sqlite3.connect(str(DB_PATH))
        try:
            dupes = conn.execute(
                """SELECT user_id, group_id, COUNT(*) c FROM auth_user_groups
                    GROUP BY user_id, group_id HAVING c > 1"""
            ).fetchall()
        finally:
            conn.close()
        assert not dupes, f"Duplicate role grants: {dupes}"
