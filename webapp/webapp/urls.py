"""
Definition of urls for webapp.
"""

from django.urls import path
from app import views
from . import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.home, name="home"),
    path("index/", views.home, name="index"),
    path("doTheMagic/", views.do_the_magic, name="doTheMagic"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
