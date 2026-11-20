"""A stock Django admin, as many internal tools are. Nothing here is tuned for Mimic."""

import os
from pathlib import Path

DATA = Path(os.environ.get("REALAPP_DATA", "/tmp/realapp"))
SECRET_KEY = "test-only-not-secret"
DEBUG = False
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "realapps.expenses",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]
ROOT_URLCONF = "realapps.expenses_site.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": DATA / "db.sqlite3"}}
MEDIA_ROOT = DATA / "media"
STATIC_URL = "/static/"
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
