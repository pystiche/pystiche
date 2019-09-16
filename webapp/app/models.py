"""
Definition of models.
"""
import os

from django.db import models
from django.db.models import CharField


class Content(models.Model):
    contentfile = models.ImageField(upload_to="media")
    stylefile = models.ImageField(upload_to="media")


# Create your models here.
