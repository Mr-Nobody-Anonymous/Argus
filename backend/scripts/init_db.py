"""Initialize the database and create tables"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.database.db import get_db, close_db

print("=" * 50)
print("Initializing Argus Database...")
print("=" * 50)

try:
    db = get_db()
    print(f"\n✓ Database initialized at: {db.db_path}")
    
    # Check tables
    tables = db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    print(f"✓ Tables created: {[t['name'] for t in tables]}")
    
    print("\nDatabase initialization complete!")
except Exception as e:
    print(f"\n✗ Error initializing database: {e}")
    sys.exit(1)
finally:
    close_db()
