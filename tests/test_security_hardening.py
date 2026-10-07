"""
Security hardening tests:
- Biometric AES-256-GCM encryption & tamper detection
- Cryptographic hash-chained audit log & verification
- Rate limiter sliding window enforcement
- HTTP defensive security headers
- Authentication rejection on malformed and missing tokens
"""

import json
import sqlite3
import pytest
from starlette.testclient import TestClient

from backend.security.crypto import encrypt_embedding, decrypt_embedding, is_encrypted
from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.services.management.audit_log import AuditLog


class TestBiometricEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        original = [0.1234, -0.5678, 0.9012, -0.3456]
        ciphertext = encrypt_embedding(original)
        assert is_encrypted(ciphertext)
        assert ciphertext.startswith("aes-gcm:")
        assert json.dumps(original) not in ciphertext

        recovered = decrypt_embedding(ciphertext)
        assert recovered == pytest.approx(original)

    def test_legacy_format_backward_compatible(self):
        legacy = json.dumps([1.0, 2.0, 3.0])
        assert not is_encrypted(legacy)
        assert decrypt_embedding(legacy) == [1.0, 2.0, 3.0]

    def test_corrupted_ciphertext_rejected(self):
        bad_ciphertext = "aes-gcm:dGhpc19pc19pbnZhbGlkX2Jhc2U2NA=="
        with pytest.raises(Exception):
            decrypt_embedding(bad_ciphertext)


class TestAuditLogHashChain:
    def test_hash_chain_and_tamper_detection(self, tmp_path):
        db_file = tmp_path / "test_audit.db"
        audit = AuditLog(db_path=str(db_file))

        # Add sequential entries
        id1 = audit.record(username="admin", action="create_camera", resource="camera", resource_id="1")
        id2 = audit.record(username="admin", action="update_zone", resource="zone", resource_id="1")
        id3 = audit.record(username="operator", action="acknowledge_event", resource="event", resource_id="42")

        assert id1 is not None and id2 is not None and id3 is not None

        # Verify initial valid chain
        res = audit.verify_integrity()
        assert res["verified"] is True
        assert res["entries_checked"] == 3

        # Simulate database tampering (malicious modification of action)
        conn = sqlite3.connect(str(db_file))
        conn.execute("UPDATE audit_log SET action = 'delete_camera' WHERE id = 1")
        conn.commit()
        conn.close()

        # Verify integrity detection flags tampering
        tamper_res = audit.verify_integrity()
        assert tamper_res["verified"] is False
        assert tamper_res["broken_at_id"] == 1


class TestRateLimiter:
    def test_sliding_window_throttling(self):
        limiter = SlidingWindowRateLimiter()
        key = "test_client_ip"

        # Allow up to 3 requests
        for _ in range(3):
            allowed, _ = limiter.is_allowed(key, max_requests=3, window_seconds=60)
            assert allowed is True

        # 4th request must be rejected
        allowed, retry_after = limiter.is_allowed(key, max_requests=3, window_seconds=60)
        assert allowed is False
        assert retry_after > 0


class TestHttpSecurityHeaders:
    def test_defensive_headers_present(self):
        from backend.api.main import app
        client = TestClient(app)
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_unauthenticated_request_rejected(self):
        from backend.api.main import app
        client = TestClient(app)
        # Calling protected endpoint without token must yield 401
        response = client.get("/api/v1/cameras")
        assert response.status_code == 401

    def test_malformed_token_rejected(self):
        from backend.api.main import app
        client = TestClient(app)
        headers = {"Authorization": "Bearer totally-invalid-jwt-token"}
        response = client.get("/api/v1/cameras", headers=headers)
        assert response.status_code == 401


class TestPasswordChangeEnforcement:
    def test_must_change_password_blocks_access_until_updated(self, monkeypatch):
        from backend.api.auth import AuthUser, create_token
        from backend.api.main import app
        client = TestClient(app)

        temp_user = AuthUser(
            id=9999,
            username="tempadmin",
            role="admin",
            is_superuser=True,
            is_staff=True,
            must_change_password=True,
        )

        # Monkeypatch token_to_user to simulate live user lookup
        monkeypatch.setattr("backend.api.auth.token_to_user", lambda token, expected_type="access": temp_user)
        token = create_token(temp_user, "access")
        headers = {"Authorization": f"Bearer {token}"}

        # Any protected mutating or role-restricted API call must be blocked with 403
        resp = client.get("/api/v1/cameras", headers=headers)
        assert resp.status_code == 403
        assert "Password change required" in resp.json()["detail"]


class TestBiometricKeyRotation:
    def test_key_rotation_roundtrip(self, tmp_path):
        import secrets
        from backend.security.crypto import (
            encrypt_embedding_with_key,
            decrypt_embedding_with_key,
        )
        from backend.scripts.rotate_biometric_key import rotate_biometric_keys

        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        test_db = tmp_path / "test_faces.db"

        conn = sqlite3.connect(str(test_db))
        conn.execute("""
            CREATE TABLE known_faces (
                id INTEGER PRIMARY KEY,
                name TEXT,
                encoding TEXT
            )
        """)
        original_vector = [0.1, 0.2, 0.3, 0.4]
        ciphertext1 = encrypt_embedding_with_key(original_vector, key1)
        conn.execute("INSERT INTO known_faces (name, encoding) VALUES ('Alice', ?)", (ciphertext1,))
        conn.commit()
        conn.close()

        # Run rotation from key1 to key2
        rotated = rotate_biometric_keys(test_db, key1, key2)
        assert rotated == 1

        # Verify record in database is now decryptable with key2
        conn = sqlite3.connect(str(test_db))
        row = conn.execute("SELECT encoding FROM known_faces WHERE name = 'Alice'").fetchone()
        conn.close()

        recovered = decrypt_embedding_with_key(row[0], key2)
        assert recovered == pytest.approx(original_vector)
