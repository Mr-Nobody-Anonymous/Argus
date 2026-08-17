"""
Django settings for Argus admin interface
"""
from pathlib import Path

# settings.py lives at <repo>/backend/django_admin/settings.py, so the repo root
# is three levels up. Going four levels up pointed BASE_DIR outside the project
# and put data/argus_django.db in the parent directory.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = "django-insecure-argus-admin-key-change-in-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'backend.django_admin',
]

# Point Django at the SAME SQLite file the FastAPI backend uses, so the admin
# lists real cameras/zones/events rather than an empty parallel database.
# Django's own auth/session tables are created alongside them via migrate.
def _argus_db_path():
    try:
        from backend.config.config import get_config, resolve_path
        return resolve_path(get_config().database.url.replace("sqlite:///", ""))
    except Exception:
        return BASE_DIR / 'data' / 'argus.db'


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(_argus_db_path()),
    }
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'frontend' / 'dist']

TEMPLATES = [
    {
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
    },
]

ROOT_URLCONF = 'backend.django_admin.urls'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]