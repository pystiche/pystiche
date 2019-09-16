"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
from django.core.files.storage import FileSystemStorage
import os
import torch
from pystiche.encoding import vgg19_encoder
from pystiche.image import read_image
from pystiche.image.transforms.functional import export_to_pil
from pystiche.nst import (
    MultiOperatorEncoder,
    DirectEncodingComparisonOperator,
    GramEncodingComparisonOperator,
    ImageOptimizer,
)
from io import BytesIO


def perform_style_transfer(content_file, style_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = read_image(content_file[1:], size=500).to(device)
    style_image = read_image(style_file[1:], size=500).to(device)

    encoder = MultiOperatorEncoder(vgg19_encoder())

    name = "Content loss"
    layers = ("relu_4_2",)
    content_operator = DirectEncodingComparisonOperator(encoder, layers, name=name)

    name = "Style loss"
    layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
    score_weight = 1e4
    style_operator = GramEncodingComparisonOperator(
        encoder, layers, name=name, score_weight=score_weight
    )
    nst = ImageOptimizer(content_operator, style_operator).to(device)

    input_image = content_image.clone()
    content_operator.set_target(content_image)
    style_operator.set_target(style_image)

    num_steps = 500
    output_image = nst(input_image, num_steps)

    pil_image = export_to_pil(output_image)
    output_file = BytesIO()
    pil_image.save(output_file, format="jpeg")
    return output_file


def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(request, "app/index.html", {"year": datetime.now().year})


def upload(fs, file, name):
    content_file_path = name + os.path.splitext(file.name)[1]
    fs.delete(content_file_path)
    content_file = fs.save(content_file_path, file)
    return fs.url(content_file)


def ready_for_style_transfer(request):
    return "contentfile" in request.FILES and "stylefile" in request.FILES


def do_the_magic(request):
    fs = FileSystemStorage()

    if not ready_for_style_transfer(request):
        # FIXME: this is currently without any effect
        response = JsonResponse({"error": "there was an error"})
        response.status_code = 400
        return response

    content_file_url = upload(fs, request.FILES["contentfile"], "content")
    style_file_url = upload(fs, request.FILES["stylefile"], "style")

    output_file = perform_style_transfer(content_file_url, style_file_url)
    output_file_url = fs.url(fs.save("output_image.jpg", output_file))
    return JsonResponse({"imgsrc": output_file_url})
