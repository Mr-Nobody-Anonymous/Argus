#!/usr/bin/env python
"""
Script to run Django Admin server alongside FastAPI
Usage: python run_admin.py
Then access admin at http://localhost:8001/admin/
"""
import os
import sys
from pathlib import Path

# Add Argus to path
sys.path.insert(0, str(Path(__file__).parent))

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.django_admin.settings')

import django
django.setup()

from django.core.management import execute_from_command_line

if __name__ == '__main__':
    # Create admin user if needed
    from django.contrib.auth.models import User
    if not User.objects.filter(username='admin').exists():
        print("Creating admin user...")
        User.objects.create_superuser('admin', 'admin@argus.local', 'admin123')
        print("Admin user created: admin / admin123")
    
    # Run Django development server on port 8001
    sys.argv = ['manage.py', 'runserver', '8001']
    execute_from_command_line(sys.argv)