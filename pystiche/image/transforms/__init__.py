from pystiche.misc import suppress_warnings

with suppress_warnings(UserWarning):
    from .color import *
    from .core import *
    from .crop import *
    from .io import *
    from .misc import *
    from .motif import *
    from .processing import *
    from .resize import *
