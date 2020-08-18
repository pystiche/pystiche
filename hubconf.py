dependencies = ["pystiche"]

from pystiche.enc.models.vgg import MODELS

for arch in ("vgg16", "vgg19"):
    globals()[arch] = MODELS[arch]

del arch
del MODELS
