import os
import sys

import pytest

from .fixtures import *

# Easy debugging without installing auto-round.
sys.path.insert(0, "..")

try:
    import torch

    # When loaded via the "meta" device, `gptqmodel==6.0.3` raises an error because the
    # internal loading process within the `transformers` library defaults to "meta" mode.
    # Importing under a CPU device context avoids that failure during module loading.
    with torch.device("cpu"):
        import gptqmodel  # pylint: disable=E0401
except ImportError:
    pass
