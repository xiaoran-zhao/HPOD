"""Core optimization components extracted from the OPD unlearning project."""

from .config import CoreConfig
from .training import StrictOnPolicyEngine

__all__ = ["CoreConfig", "StrictOnPolicyEngine"]
