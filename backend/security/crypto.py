"""
Cryptographic services for Argus.

Provides authenticated application-level encryption for sensitive biometric
fields (e.g. face embeddings) stored in the database.
Uses AES-256-GCM with distinct nonces per operation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, List, Optional, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from backend.config.config import resolve_path

logger = logging.getLogger(__name__)

# Domain-separation label for face embeddings; it does not bind ciphertext to a database row.
BIOMETRIC_ASSOCIATED_DATA = b"argus_biometric_face_v1"

_CACHED_KEY: Optional[bytes] = None


def get_biometric_key() -> bytes:
    """
    Resolve the 256-bit symmetric encryption key for biometric embeddings.

    Priority:
    1. Environment variable `ARGUS_BIOMETRIC_KEY` (hex, base64, or passphrase).
    2. Local persistent key file `data/.biometric.key`.
    3. If neither exists, generates a secure 32-byte key and persists it to
       `data/.biometric.key` with restricted file permissions.
    """
    global _CACHED_KEY
    if _CACHED_KEY is not None:
        return _CACHED_KEY

    env_val = os.environ.get("ARGUS_BIOMETRIC_KEY", "").strip()
    if env_val:
        # Check if hex-encoded (64 chars)
        if len(env_val) == 64:
            try:
                _CACHED_KEY = bytes.fromhex(env_val)
                return _CACHED_KEY
            except ValueError:
                pass

        # Check if base64-encoded (44 chars for 32 bytes)
        try:
            decoded = base64.b64decode(env_val)
            if len(decoded) == 32:
                _CACHED_KEY = decoded
                return _CACHED_KEY
        except Exception:
            pass

        # Otherwise derive a 256-bit key via SHA-256 from the provided passphrase
        _CACHED_KEY = hashlib.sha256(env_val.encode("utf-8")).digest()
        return _CACHED_KEY

    # Never replace an existing key that is malformed or unreadable: doing so
    # would make previously encrypted biometric records permanently inaccessible.
    key_path = resolve_path("data/.biometric.key")
    if key_path.exists():
        try:
            raw = key_path.read_bytes()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read biometric key at {key_path}; refusing to generate a replacement."
            ) from exc
        if len(raw) != 32:
            raise RuntimeError(
                f"Biometric key at {key_path} must contain exactly 32 bytes; refusing to replace it."
            )
        _CACHED_KEY = raw
        return _CACHED_KEY

    # Generate and persist a key atomically. If another process wins the
    # exclusive create, read the key it created rather than overwriting it.
    new_key = secrets.token_bytes(32)
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as key_file:
            key_file.write(new_key)
            key_file.flush()
            os.fsync(key_file.fileno())
    except FileExistsError:
        try:
            raw = key_path.read_bytes()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read biometric key at {key_path}; refusing to generate a replacement."
            ) from exc
        if len(raw) != 32:
            raise RuntimeError(
                f"Biometric key at {key_path} must contain exactly 32 bytes; refusing to replace it."
            )
        _CACHED_KEY = raw
        return _CACHED_KEY
    except Exception as exc:
        raise RuntimeError(
            f"Cannot persist biometric key at {key_path}; refusing to use a temporary key."
        ) from exc

    try:
        # Unix permission hardening (owner read/write only)
        os.chmod(key_path, 0o600)
    except Exception:
        pass
    logger.info(f"Generated new biometric encryption key at {key_path}")
    _CACHED_KEY = new_key
    return _CACHED_KEY


def encrypt_embedding_with_key(embedding: Union[List[float], List[int], Any], key: bytes) -> str:
    """Encrypt a biometric embedding list with an explicit 256-bit AES key."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)

    serialized = json.dumps(embedding).encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, serialized, BIOMETRIC_ASSOCIATED_DATA)

    payload = base64.b64encode(nonce + ciphertext).decode("ascii")
    return f"aes-gcm:{payload}"


def decrypt_embedding_with_key(stored_value: str, key: bytes) -> List[float]:
    """Decrypt an embedding string back into a list of floats using an explicit 256-bit AES key."""
    if not stored_value:
        return []

    stripped = stored_value.strip()
    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except Exception as exc:
            logger.error(f"Failed to parse legacy embedding JSON: {exc}")
            return []

    if not stripped.startswith("aes-gcm:"):
        raise ValueError(f"Unknown or unsupported embedding format: {stored_value[:20]}...")

    raw = base64.b64decode(stripped[len("aes-gcm:"):])
    if len(raw) < 28:  # 12-byte nonce + 16-byte tag minimum
        raise ValueError("Corrupted biometric ciphertext: length too short")

    nonce = raw[:12]
    ciphertext = raw[12:]

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, BIOMETRIC_ASSOCIATED_DATA)
    return json.loads(plaintext.decode("utf-8"))


def encrypt_embedding(embedding: Union[List[float], List[int], Any]) -> str:
    """
    Encrypt a biometric embedding list into an authenticated ciphertext string.

    Format: `aes-gcm:<base64(nonce_12b + ciphertext + tag_16b)>`
    """
    return encrypt_embedding_with_key(embedding, get_biometric_key())


def decrypt_embedding(stored_value: str) -> List[float]:
    """
    Decrypt an embedding string back into a list of floats.

    Supports transparent backward-compatibility:
    - If the value starts with `[` (unencrypted legacy format), it parses JSON.
    - If the value starts with `aes-gcm:`, it decrypts with AES-256-GCM.
    """
    return decrypt_embedding_with_key(stored_value, get_biometric_key())


def is_encrypted(stored_value: str) -> bool:
    """Check if the given stored embedding is encrypted."""
    return bool(stored_value and stored_value.strip().startswith("aes-gcm:"))
