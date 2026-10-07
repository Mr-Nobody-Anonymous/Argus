# Biometric Data Security & Encryption Architecture

Face recognition is disabled by default. If explicitly enabled, Argus processes face data and stores encrypted feature embeddings with identity labels. New registrations do not persist source face crops. Older installations may still have plaintext crop files in `data/known_faces`; this release leaves those files untouched so operators can review and remove them deliberately.

This document describes the cryptographic safeguards, key management, operational procedures, and failure models governing biometric data in Argus.

---

## 1. Threat Model & Safeguards

| Threat Vector | Risk Scenario | Argus Defensive Mechanism |
|---|---|---|
| **Database-only Exfiltration** | Attacker obtains a database dump but not the application key | Encrypted embeddings remain opaque after migration. A local key file stored beside the database does not protect against theft of the entire data volume; production should use a separately managed key. |
| **Tampering / Bit-flipping** | Attacker modifies ciphertext in database | AES-GCM provides authenticated encryption (AEAD). Any modified ciphertext fails tag verification and is rejected at load time. |
| **Cross-record substitution** | Attacker copies a valid encrypted embedding to another face row | The current shared AAD identifies face-embedding data but does not bind ciphertext to a row or identity; row swaps are not prevented. Protect database write access and treat this as a known limitation. |
| **API Leakage** | Compromised viewer or operator account dumps user embeddings | Raw 10,000-dimensional biometric embeddings are **never** returned via public REST or WebSocket APIs; only matched names and similarity scores are emitted. |
| **Key Leakage in Git** | Accidental commit of cryptographic keys | Keep keys out of Git and use a secret manager in production. The repository checks selected configuration files but does not guarantee detection of every secret. |

---

## 2. Cryptographic Architecture

When explicitly enabled, face embeddings are serialized and encrypted at the application layer before persistence. New enrollment images are not written to disk:

```
Face Image (OpenCV)
       │
       ▼
Feature Vector (e.g. 100x100 matrix -> float array)
       │
       ▼
JSON Serialization
       │
       ▼
AES-256-GCM Authenticated Encryption
  - Key: 256-bit symmetric key from ARGUS_BIOMETRIC_KEY
  - Nonce: 12 bytes randomly generated per operation (os.urandom)
  - AAD: b"argus_biometric_face_v1" (domain label only; it does not bind the row or identity)
  - Tag: 16-byte GCM authentication tag
       │
       ▼
Stored Format: aes-gcm:<base64(12B Nonce + Ciphertext + 16B Tag)>
```

### Format Specification
- Storage column: `known_faces.encoding` (TEXT)
- Prefix: `aes-gcm:`
- Body: Base64-encoded `[12-byte IV/Nonce] + [Ciphertext] + [16-byte Auth Tag]`

### Backward Compatibility
Argus retains backwards compatibility for legacy databases:
- If a stored value starts with `[`, it is treated as a legacy unencrypted JSON array and upgraded to AES-256-GCM when face recognition loads it.
- Legacy face-crop image files are not automatically deleted or migrated. Review `data/known_faces` locally and remove old plaintext crops deliberately before enabling face recognition.

---

## 3. Key Generation & Management

### Key Derivation & Priority
When Argus initializes, `get_biometric_key()` resolves the 256-bit key in this order:

1. **Environment Variable**: `ARGUS_BIOMETRIC_KEY`
   - Accepts a 64-character hexadecimal string (`openssl rand -hex 32`), or
   - A 44-character Base64 string (`openssl rand -base64 32`), or
   - A high-entropy passphrase (derived via SHA-256).
2. **Local Key File**: `data/.biometric.key` (read with `0600` permissions).
3. **Automatic Provisioning**: If neither exists on first use, Argus creates a cryptographically secure 32-byte key (`secrets.token_bytes(32)`) at `data/.biometric.key` with owner-only permissions where supported. If an existing key is unreadable or invalid, startup fails closed instead of replacing it.

### Manual Key Generation
To generate a production-ready key:
```bash
# Using OpenSSL (recommended)
openssl rand -base64 32

# Or using Python
python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```
Set the result in your environment:
```bash
export ARGUS_BIOMETRIC_KEY="YOUR_BASE64_KEY_HERE"
```

---

## 4. Key Rotation Procedure

Argus includes a key rotation utility: [`backend/scripts/rotate_biometric_key.py`](../backend/scripts/rotate_biometric_key.py).

The database rows are updated in one SQLite transaction, but the database and key store cannot be updated atomically together. Back up both the database and the current key, keep the new key securely available, and verify recovery before deleting any backup.

### Step 1: Pre-flight Dry Run
Verify that the current key can decrypt all enrolled faces without modifying the database:
```bash
python backend/scripts/rotate_biometric_key.py --dry-run
```

### Step 2: Execute Atomic Re-encryption
Re-encrypt all biometric records under a new cryptographic key:
```bash
python backend/scripts/rotate_biometric_key.py
```
Output:
```text
[*] Found 14 enrolled biometric face records.
[✓] Successfully re-encrypted 14 records in database.
[✓] Wrote updated encryption key to /app/data/.biometric.key
[✓] New key (Base64): K2d9Fv1...==
[*] Update ARGUS_BIOMETRIC_KEY in your environment or .env file:
    export ARGUS_BIOMETRIC_KEY="K2d9Fv1...=="
```

The database transaction is atomic: if any record fails decryption or re-encryption, the entire transaction rolls back, leaving your database untouched.

---

## 5. Backup & Recovery

1. **Key Backup**: Always store a copy of `ARGUS_BIOMETRIC_KEY` in a secure enterprise secret manager (e.g. AWS Secrets Manager, HashiCorp Vault, Azure Key Vault, or 1Password).
2. **Database Backup**: Protect database dumps and the encryption key separately. The local key-file fallback is stored under the application data directory, so copying the entire data volume also copies that key. Use `ARGUS_BIOMETRIC_KEY` from a separate secret manager when protection against full-volume theft is required.
3. **Recovery**:
   - Restore database file or import SQL dump.
   - Inject the backed-up `ARGUS_BIOMETRIC_KEY` into the container environment.
   - Start Argus (`python argus.py start` or `docker compose -f docker-compose.prod.yml up -d`).

---

## 6. Failure Modes & Lost Key Scenario

### What Happens if the Key is Lost?
Because Argus uses standard AES-256-GCM authenticated encryption with no backdoors or escrow keys:

> [!CAUTION]
> If `ARGUS_BIOMETRIC_KEY` and `data/.biometric.key` are lost, **all stored face embeddings are mathematically unrecoverable**.

This property guarantees **cryptographic erasure** (crypto-shredding): deleting the key permanently erases all biometric subjects from compromised storage without requiring raw disk overwrites.

To recover from a lost key:
1. Clear the `known_faces` table:
   ```sql
   DELETE FROM known_faces;
   ```
2. Re-enroll subjects via the admin dashboard or face registration API.

---

## 7. Decryption Access Control

- **Who can decrypt embeddings?** Only the active Argus backend process possessing the in-memory `ARGUS_BIOMETRIC_KEY`.
- **Can database administrators decrypt embeddings?** No. Direct database access (via SQLite shell or `psql`) exposes only `aes-gcm:<base64>` ciphertext.
- **Can authenticated API users inspect raw embeddings?** No. The REST API exposes only metadata (subject name, registration timestamp, camera ID). Embeddings are loaded directly into memory for similarity matching by OpenCV/DeepFace and are never serialized into HTTP responses.
