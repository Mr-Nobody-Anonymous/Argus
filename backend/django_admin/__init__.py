"""
Django Admin app for Argus
Provides Django admin interface alongside FastAPI
"""
import os
import sys
from pathlib import Path

# Ensure data directory exists
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configure Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.django_admin.settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY="django-insecure-argus-admin-key-change-in-production",
        DEBUG=True,
        ALLOWED_HOSTS=["*"],
        INSTALLED_APPS=[
            'django.contrib.admin',
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.messages',
            'django.contrib.staticfiles',
            'backend.django_admin',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': str(DATA_DIR / 'argus_django.db'),
            }
        },
        DEFAULT_AUTO_FIELD='django.db.models.BigAutoField',
        STATIC_URL='/static/',
        STATICFILES_DIRS=[str(BASE_DIR / 'frontend' / 'dist')],
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.debug',
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
            },
        }],
        ROOT_URLCONF='backend.django_admin.urls',
        MIDDLEWARE=[
            'django.middleware.security.SecurityMiddleware',
            'django.contrib.sessions.middleware.SessionMiddleware',
            'django.middleware.common.CommonMiddleware',
            'django.middleware.csrf.CsrfViewMiddleware',
            'django.contrib.auth.middleware.AuthenticationMiddleware',
            'django.contrib.messages.middleware.MessageMiddleware',
        ],
    )
    django.setup()


# Import models for registration (they are defined in admin.py)
from .admin import Camera, Zone, Event, BehaviorProfile

# Create tables on module import
def setup_database():
    """Create database tables for Django admin"""
    from django.core.management import call_command
    try:
        call_command('migrate', '--run-syncdb', verbosity=0)
    except Exception:
        pass  # Tables might already exist


setup_database()