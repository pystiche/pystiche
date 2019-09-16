"""
Definition of forms.
"""

from django import forms
from .models import *


class UploadForm(forms.ModelForm):
    class Meta:
        model = Content
        fields = ["contentfile", "stylefile"]
