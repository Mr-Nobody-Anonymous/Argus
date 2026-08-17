"""
Django Admin app for Argus.

Provides a Django admin interface over the same SQLite database the FastAPI
backend uses.

NOTE: This module deliberately contains no ``settings.configure()`` call and no
``django.setup()``. It is an installed *app* package (listed in
``INSTALLED_APPS``), so Django imports it while the settings module is already
being loaded. Configuring settings here previously duplicated - and silently
overrode - ``settings.py`` (including pointing the admin at a different
database file), and calling ``django.setup()`` at import time risks
``AppRegistryNotReady``.

Configuration lives in ``backend/django_admin/settings.py``. Launch the admin
with::

    python backend/scripts/run_admin.py

or point ``DJANGO_SETTINGS_MODULE`` at ``backend.django_admin.settings``.
"""
