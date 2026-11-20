from pathlib import Path

import django.contrib.admin
from django.contrib import admin
from django.urls import path
from django.views.static import serve

ADMIN_STATIC = Path(django.contrib.admin.__file__).parent / "static"

urlpatterns = [
    path("admin/", admin.site.urls),
    # DEBUG is off (like production), so serve the admin's CSS/JS here.
    path("static/<path:path>", serve, {"document_root": ADMIN_STATIC}),
]
