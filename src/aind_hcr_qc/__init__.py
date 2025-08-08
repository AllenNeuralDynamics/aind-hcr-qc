"""Init package

With curated public API for aind_hcr_qc.

For example:
import aind_hcr_qc as qc
qc.run_qc(...)

"""

__version__ = "0.3.5"

# Public API
from .constants import CHANNEL_COLORS

__all = [
    "CHANNEL_COLORS",
]