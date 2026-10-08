"""Shared experiment protocol and capability declarations."""

from .capabilities import AVAILABLE_CAPABILITIES, Capability, missing_capabilities
from .protocol import DEFAULT_PROTOCOL, ExperimentProtocol
from .registry import (
    EXPERIMENTS,
    ExperimentDefinition,
    ExperimentStatus,
    get_experiment,
    validate_experiment,
    validate_registry,
)

__all__ = (
    "ExperimentProtocol",
    "DEFAULT_PROTOCOL",
    "Capability",
    "AVAILABLE_CAPABILITIES",
    "missing_capabilities",
    "ExperimentStatus",
    "ExperimentDefinition",
    "EXPERIMENTS",
    "get_experiment",
    "validate_registry",
    "validate_experiment",
)
