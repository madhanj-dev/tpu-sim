"""
TPU Simulator Engine package.
"""

from .models import ModelConfig, PREDEFINED_MODELS, get_model_config
from .hardware import HardwareConfig, PREDEFINED_HARDWARE, get_hardware_config
from .roofline import (
    SimulationConfig,
    BatchDataPoint,
    SimulationResult,
    RooflineEngine,
    RooflineStrategy,
    StandardRooflineStrategy,
    DetailedRooflineStrategy,
    DynamicFormulaRooflineStrategy,
    register_strategy,
)

__all__ = [
    "ModelConfig",
    "PREDEFINED_MODELS",
    "get_model_config",
    "HardwareConfig",
    "PREDEFINED_HARDWARE",
    "get_hardware_config",
    "SimulationConfig",
    "BatchDataPoint",
    "SimulationResult",
    "RooflineEngine",
    "RooflineStrategy",
    "StandardRooflineStrategy",
    "DetailedRooflineStrategy",
    "DynamicFormulaRooflineStrategy",
    "register_strategy",
]
