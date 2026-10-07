# Database Architecture & PostgreSQL Migration Guide

Argus supports a dual-tier database strategy:

1. **SQLite (Default for Edge / Single-Node)**:
   - Zero-configuration deployment: `python argus.py start` works out of the box with zero external dependencies.
   - Ideal for local testbenches, single-camera edge appliances, and isolated offline monitoring.
   - Stored in `data/argus.db` with WAL mode enabled.

2. **PostgreSQL (Production / Multi-Camera Ingestion)**:
   - Required for production clusters with concurrent camera feeds, high-frequency event ingestion, and multiple operator dashboards.
   - Eliminates SQLite `database is locked` table-level write locks.
   - Provides native connection pooling and concurrent ACID transactions.

---

## PostgreSQL Setup & Configuration

### 1. Run PostgreSQL with Docker Compose

Add PostgreSQL to your deployment stack or start a local container:

```bash
docker run -d \
  --name argus-postgres \
  -e POSTGRES_DB=argus \
  -e POSTGRES_USER=argus \
  -e POSTGRES_PASSWORD=secure_production_password \
  -p 5432:5432 \
  -v argus_pgdata:/var/lib/postgresql/data \
  postgres:16-alpine
```

### 2. Configure Argus Environment

Set `DATABASE_URL` in your `.env` or container environment:

```env
DATABASE_URL=postgresql://argus:secure_production_password@localhost:5432/argus
```

Install PostgreSQL client dependencies if running in a native virtual environment:

```bash
pip install psycopg2-binary
```

Argus automatically detects the `postgresql://` URI scheme on boot and initializes all tables and indices.

---

## Data Migration from SQLite to PostgreSQL

To migrate existing cameras, zones, and events from an existing `data/argus.db` to PostgreSQL, run the migration script:

```bash
python backend/scripts/migrate_sqlite_to_pg.py \
  --sqlite-path data/argus.db \
  --pg-url postgresql://argus:secure_production_password@localhost:5432/argus
```
