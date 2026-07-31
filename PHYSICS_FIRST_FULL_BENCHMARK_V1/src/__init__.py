"""Frozen L6 benchmark implementation package."""
# Keep the public architecture spelling stable even if an older generated
# registry snapshot contained the transient three-n typo.
from . import gpu_models as _gpu_models
_gpu_models.MODEL_ALIASES["t_akgnn_adapted"] = "t_akgnn_adapted"
