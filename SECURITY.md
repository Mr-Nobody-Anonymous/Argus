# Security Policy

The Argus maintainer accepts security reports about the edge video analytics, identity-processing, and biometric-data components. This policy describes the available reporting path and current support expectations.

---

## Supported Versions

Argus is an early-stage project and does not yet publish a formal version-support window. Security fixes are intended for the current default branch; do not assume older commits or untagged builds receive updates.

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues or discussions.**

Use GitHub **Report a vulnerability** on the repository Security tab if private vulnerability reporting is enabled. If that option is unavailable, contact the maintainer through the GitHub profile. Do not post exploitable details in a public issue.

### What to Include in Your Report

To help us investigate and reproduce the issue quickly, please provide:

- A clear description of the vulnerability and its potential impact.
- Affected component(s) (e.g. Authentication/JWT, Biometric Storage, RTSP Ingestion, API endpoint, Docker container).
- Step-by-step instructions to reproduce the issue (proof-of-concept scripts or curl commands).
- Environment details (Operating system, Python version, Docker vs. native, dependency versions).
- Proposed fix or mitigation (if available).

---

## Response & Disclosure Timeline

This is a small, independently maintained project and no response or remediation SLA is promised. The maintainer will review reports as capacity allows and may request additional details through a private channel.

---

## Vulnerability Severity Classification

We classify vulnerabilities using CVSS v3.1 standards:

| Severity | Criteria / Examples | Suggested Triage Priority |
|---|---|---|
| **Critical** | Remote code execution, unauthenticated arbitrary file read/write, biometric key extraction, auth bypass granting admin privileges. | Highest |
| **High** | Privilege escalation (viewer to admin), cleartext biometric data leakage, rate-limiting circumvention leading to credential brute-forcing. | High |
| **Medium** | Denial of Service (DoS) against RTSP pipeline, insecure default CORS configuration, path traversal with restricted read access. | Normal |
| **Low** | Verbose error stack traces, minor information disclosure without sensitive data exposure. | As capacity allows |

---

## Scope & Security Posture

### In Scope
- FastAPI REST API endpoints and middleware (`backend/api/`)
- Authentication, JWT issuance/validation, and RBAC authorization (`backend/api/auth.py`)
- Biometric enrollment, encrypted embedding storage, and legacy embedding migration (`backend/services/vision/face_recognition.py`)
- Database access layers and migration scripts (`backend/database/`)
- Docker container build and runtime permissions (`Dockerfile`, `docker-compose.prod.yml`)
- RTSP / WebSocket frame transmission security and access control

### Out of Scope
- Attacks requiring physical access to the host machine or cameras
- Compromise of external RTSP camera hardware firmware
- Denial of service caused by saturating physical network bandwidth
- Social engineering attacks targeting operators

---

## Security Best Practices for Operators

1. **Use TLS for remote access**: Keep the backend bound to loopback or a private service network; configure and validate a TLS reverse proxy before public exposure.
2. **Set a strong JWT secret**: Provide an `ARGUS_JWT_SECRET` of at least 32 cryptographically random characters.
3. **Keep the biometric key separate from the database**: Set `ARGUS_BIOMETRIC_KEY` from a secret manager for production. The local `data/.biometric.key` fallback is stored beside application data and does not protect against theft of the entire data volume.
4. **Review legacy data before enabling face recognition**: Existing plaintext face-crop files are not automatically deleted; back them up if needed and remove them deliberately.
5. **Isolate camera subnets**: Run RTSP cameras on a restricted surveillance network.
