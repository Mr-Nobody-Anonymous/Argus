#!/usr/bin/env python3
"""
Biometric Key Rotation Utility for Argus.

Rotates the AES-256-GCM symmetric encryption key used for biometric face embeddings.
Decrypts existing face embeddings using the current key, re-encrypts them under a
new cryptographic key, updates the database atomically in a single transaction,
and saves the new key to `data/.biometric.key` (or prints the environment variable).

Usage:
    python backend/scripts/rotate_biometric_key.py [--dry-run] [--new-key <base64_key>]
"""

import argparse
import base64
import os
import secrets
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.config import resolve_path
from backend.security.crypto import (
    decrypt_embedding_with_key,
    encrypt_embedding_with_key,
    get_biometric_key,
)


def parse_key(key_input: str) -> bytes:
    """Parse hex, base64, or raw string key into 32 bytes."""
    val = key_input.strip()
    if len(val) == 64:
        try:
            return bytes.fromhex(val)
        except ValueError:
            pass
    try:
        decoded = base64.b64decode(val)
        if len(decoded) == 32:
            return decoded
    except Exception:
        pass
    import hashlib
    return hashlib.sha256(val.encode("utf-8")).digest()


def rotate_biometric_keys(
    db_path: Path,
    old_key: bytes,
    new_key: bytes,
    dry_run: bool = False,
) -> int:
    """Rotate all biometric face embeddings from old_key to new_key."""
    if not db_path.exists():
        print(f"[!] Database file not found: {db_path}")
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Verify table existence
    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='known_faces'"
    ).fetchone()
    if not table_check:
        print("[*] No 'known_faces' table found in database. Nothing to rotate.")
        conn.close()
        return 0

    rows = conn.execute("SELECT id, name, encoding FROM known_faces").fetchall()
    total = len(rows)
    print(f"[*] Found {total} enrolled biometric face records.")

    rotated_count = 0
    errors = 0
    updates = []

    for row in rows:
        row_id = row["id"]
        name = row["name"]
        stored_encoding = row["encoding"]

        try:
            embedding = decrypt_embedding_with_key(stored_encoding, old_key)
            re_encrypted = encrypt_embedding_with_key(embedding, new_key)
            updates.append((re_encrypted, row_id))
            rotated_count += 1
        except Exception as exc:
            print(f"[!] Error decrypting record id={row_id} (name='{name}'): {exc}")
            errors += 1

    if errors > 0:
        print(f"[!] Aborting rotation: {errors} records failed to decrypt with the old key.")
        conn.close()
        return 0

    if dry_run:
        print(f"[DRY-RUN] Successfully verified re-encryption for {rotated_count}/{total} records.")
        conn.close()
        return rotated_count

    # Execute atomic batch update
    try:
        with conn:
            conn.executemany(
                "UPDATE known_faces SET encoding = ? WHERE id = ?",
                updates,
            )
        print(f"[✓] Successfully re-encrypted {rotated_count} records in database.")
    except Exception as exc:
        print(f"[!] Database update failed: {exc}")
        conn.close()
        return 0
    finally:
        conn.close()

    # Update local key file if applicable
    key_file = resolve_path("data/.biometric.key")
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(new_key)
        try:
            os.chmod(key_file, 0o600)
        except Exception:
            pass
        print(f"[✓] Wrote updated encryption key to {key_file}")
    except Exception as exc:
        print(f"[!] Warning: Could not write new key to {key_file}: {exc}")

    b64_new_key = base64.b64encode(new_key).decode("ascii")
    print(f"[✓] New key (Base64): {b64_new_key}")
    print("[*] Update ARGUS_BIOMETRIC_KEY in your environment or .env file:")
    print(f"    export ARGUS_BIOMETRIC_KEY=\"{b64_new_key}\"")

    return rotated_count


def main():
    parser = argparse.ArgumentParser(description="Rotate Argus biometric encryption keys.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without committing changes.")
    parser.add_argument("--old-key", type=str, default="", help="Current key (default: read from env/file)")
    parser.add_argument("--new-key", type=str, default="", help="New key (default: generate random 32-byte key)")
    parser.add_argument("--db", type=str, default="", help="Path to database (default: data/argus.db)")

    args = parser.parse_args()

    db_path = Path(args.db) if args.db else resolve_path("data/argus.db")

    if args.old_key:
        old_key = parse_key(args.old_key)
    else:
        old_key = get_biometric_key()

    if args.new_key:
        new_key = parse_key(args.new_key)
    else:
        new_key = secrets.token_bytes(32)

    rotate_biometric_keys(db_path, old_key, new_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
