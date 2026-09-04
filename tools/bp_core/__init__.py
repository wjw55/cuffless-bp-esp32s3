"""Offline PPG-to-BP research pipeline.

The package deliberately has no firmware or serial-port dependencies.
"""

from .config import load_config
from .datasets import Recording, audit_datasets, discover_recordings, load_recording
from .features import build_occasion_features
from .models import build_personalized_examples, evaluate_personalized_models

__all__ = [
    "Recording",
    "audit_datasets",
    "build_occasion_features",
    "build_personalized_examples",
    "discover_recordings",
    "evaluate_personalized_models",
    "load_config",
    "load_recording",
]
