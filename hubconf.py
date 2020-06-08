dependencies = ["torch", "torchvision"]

import torch
from torchvision.models import vgg16 as _vgg16, vgg19 as _vgg19


def vgg16_caffe(progress=True, **kwargs):
    model = _vgg16(pretrained=False, **kwargs)
    url = "https://download.pystiche.org/models/vgg16-781be684.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, progress=progress)
    model.load_state_dict(state_dict)
    return model


def vgg19_caffe(progress=True, **kwargs):
    model = _vgg19(pretrained=False, **kwargs)
    url = "https://download.pystiche.org/models/vgg19-74e45263.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, progress=progress)
    model.load_state_dict(state_dict)
    return model
