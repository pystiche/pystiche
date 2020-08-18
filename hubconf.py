dependencies = ["pystiche"]

from pystiche.enc.models.vgg import MODELS

globals().update(MODELS)

del MODELS
