#!/usr/bin/env python
"""
Run the Django admin server alongside the FastAPI backend.

Usage:
    python backend/scripts/run_admin.py            # serves on :8001
    python backend/scripts/run_admin.py 0.0.0.0:8002
    python backend/scripts/run_admin.py --setup-only   # create tables + admin, then exit

Then open http://localhost:8001/admin/
Admin credentials are created automatically on first run (or set via ARGUS_ADMIN_PASSWORD).
"""
import os
import secrets
import sys
from pathlib import Path

# Add the project root to sys.path so `backend.*` imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.django_admin.settings')

import django

django.setup()

from django.core.management import execute_from_command_line, call_command


def ensure_schema():
    """Create the Argus tables (if missing) plus Django's auth/session tables.

    The admin models are `managed = False` and map onto the tables created by
    backend/database/db.py, so that schema has to exist first.
    """
    from backend.database.db import get_db, close_db

    get_db()      # creates cameras / zones / events / behavior_profiles / ...
    close_db()

    # Creates Django's own tables (auth_user, django_session, ...) in the same
    # SQLite file. Unmanaged models are skipped.
    call_command('migrate', '--run-syncdb', verbosity=0)


def ensure_admin_user(username: str = "admin", password: str = None):
    from django.contrib.auth.models import User

    if not User.objects.filter(is_superuser=True).exists():
        admin_pass = password or os.environ.get("ARGUS_ADMIN_PASSWORD", "").strip()
        is_generated = False
        if not admin_pass:
            admin_pass = secrets.token_urlsafe(16)
            is_generated = True

        u = User.objects.create_superuser(username, f"{username}@argus.local", admin_pass)
        if is_generated:
            from backend.api.auth import set_user_must_change_password
            set_user_must_change_password(u.id, True)
            print("\n" + "=" * 64)
            print(" [SECURITY] Generated initial superuser credentials:")
            print(f"   Username: {username}")
            print(f"   Password: {admin_pass}")
            print("   Store this securely. Password change is enforced on first login!")
            print("=" * 64 + "\n")
        else:
            print(f"[SECURITY] Superuser '{username}' initialized from environment.")


def main():
    ensure_schema()
    ensure_admin_user()

    # --setup-only exists for CI and first-time setup: the auth tables live in
    # Django's migrations, and without them the security suite skips almost
    # every test. Creating them must not require starting a blocking server.
    if '--setup-only' in sys.argv[1:]:
        print("Schema and admin user ready (--setup-only): not starting the server.")
        return

    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    addrport = args[0] if args else '8001'
    print(f"\nStarting Django admin on {addrport} -> http://localhost:8001/admin/\n")

    # --noreload keeps the auto-created superuser logic from running twice.
    execute_from_command_line(['manage.py', 'runserver', addrport, '--noreload'])


if __name__ == '__main__':
    main()
