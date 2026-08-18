# Storage audit — every SQLite use, classified

Produced by inspecting each call site and each table's readers/writers, then
measuring behaviour on a running system. The governing constraints are: no
persistent database storage, no image-based data storage, store as little
persistent data as possible, and never weaken authentication or audit
guarantees to satisfy a rule.

## Summary

| # | Use | Volume | Survives restart? | Verdict |
|---|-----|--------|-------------------|---------|
| 1 | `cameras.status` / `.fps` / `.last_frame_time` | up to **3.15 billion writes/yr** at 100 cameras | must **not** | **Replaced with in-memory state** |
| 2 | `auth_user`, `auth_group`, `auth_user_groups` | 3 users | must | Keep — see "Authentication" |
| 3 | `audit_log` | 1 row per mutating request | must | Keep — see "Audit" |
| 4 | `events`, `anomalies`, `license_plates` | 1 row per detection event | must | Keep — the product *is* an event recorder |
| 5 | `cameras` (identity: `id`, `name`, `rtsp_url`, `location_tag`) | 1 row per camera | must | Keep — operator-managed configuration |
| 6 | `zones`, `behavior_profiles` | operator-defined | must | Keep — operator-managed configuration |
| 7 | `known_faces` | biometric | must | Keep — enrolment data |
| 8 | Django `django_*` tables | framework | must | Keep — required by the admin UI |

## 1. Camera runtime state — **fixed**

`update_status()` wrote `status`, `fps` and `last_frame_time` to disk roughly
once per second per camera. Measured cost:

```
  1 camera:      86,400 status writes/day  (    31,536,000/yr)
 20 cameras:  1,728,000 status writes/day  (   630,720,000/yr)
100 cameras:  8,640,000 status writes/day  ( 3,153,600,000/yr)
```

Every one of those is a durable commit of data that is meaningless once the
process stops. It was also **incorrect**, demonstrated on a live server:

```
while running: ('online', 14.73899584190893)
--- SIGKILL -9 (crash / OOM-kill / power loss) ---
after crash:   ('online', 14.73899584190893)
```

A clean shutdown resets the row to `offline`, but `SIGKILL` — a crash, an
OOM-kill, or power loss — bypasses that hook, so the API served
`status: "online", fps: 14.76` for a camera that did not exist any more. An
operator's wall display would show a dead camera as live.

Liveness is a property of the *running process*, so it now lives in memory
(`backend/services/management/camera_runtime.py`). A camera not reported by the
current process is `offline` by definition, which is correct after a crash for
free. The three columns remain in the schema, always serialised from memory.

This satisfies data minimisation, removes the write amplification, and fixes
the correctness bug — one change, three problems.

## 2. Authentication — **kept, deliberately**

The task states authentication must reuse the Django `auth_user` table, and the
constraint list forbids weakening authentication to satisfy the storage rule.

Moving credentials to a JSON file would mean hand-rolling password storage and
losing Django's PBKDF2 hasher, its admin UI, and its permission model — a
material security regression to satisfy a rule aimed at *casual* persistence.
Users are also not application data: they are a small, slow-changing operator
registry that must survive restarts, or nobody can log in.

Kept, with the integrity fixes already applied (`PRAGMA foreign_keys = ON`,
orphan purge, regression tests).

## 3. Audit log — **kept, deliberately**

An audit trail that does not survive a restart is not an audit trail. Its whole
purpose is to answer "who changed what" after an incident, including after the
crash that caused the incident. Retention is bounded (`audit_days = 365`).

## 4–8. Event and configuration records — **kept**

Argus is a video analytics recorder: events, anomalies and plate reads are the
product's output, not incidental state. Camera identity, zones and enrolled
faces are operator-managed configuration. All of it must outlive the process,
all of it is bounded by the retention policy, and none of it is stored as
images-as-data (snapshots are genuine visual evidence, now bounded by
`snapshots_max_mb`).

## What was actually removed

Only the state that failed the test *"does this need to survive the process
stopping?"* — camera liveness telemetry, the single highest-volume writer in
the system.
