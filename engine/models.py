"""
Model definitions and configurations for LLM Roofline Simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    name: str
    display_name: str
    total_params_b: float       # Total parameters in Billions
    active_params_b: float      # Active parameters per token in Billions
    num_layers: int
    hidden_dim: int
    num_kv_heads: int
    head_dim: int
    is_moe: bool = False
    description: str = ""

    @property
    def total_params(self) -> float:
        return self.total_params_b * 1e9

    @property
    def active_params(self) -> float:
        return self.active_params_b * 1e9


# Predefined models
PREDEFINED_MODELS: Dict[str, ModelConfig] = {
    "qwen3-397b": ModelConfig(
        name="qwen3-397b",
        display_name="Qwen3-397B (MoE)",
        total_params_b=397.0,
        active_params_b=39.0,
        num_layers=96,
        hidden_dim=8192,
        num_kv_heads=8,
        head_dim=128,
        is_moe=True,
        description="Qwen3 397B Mixture-of-Experts model with ~39B active parameters per token."
    ),
    "gemma4": ModelConfig(
        name="gemma4",
        display_name="Gemma 4 (31B)",
        total_params_b=31.0,
        active_params_b=31.0,
        num_layers=64,
        hidden_dim=4096,
        num_kv_heads=8,
        head_dim=128,
        is_moe=False,
        description="Gemma 4 dense model with 31B active parameters."
    ),
    "gemma2-27b": ModelConfig(
        name="gemma2-27b",
        display_name="Gemma 2 (27B)",
        total_params_b=27.0,
        active_params_b=27.0,
        num_layers=46,
        hidden_dim=4608,
        num_kv_heads=16,
        head_dim=128,
        is_moe=False,
        description="Gemma 2 dense model with 27B parameters."
    ),
    "llama3.1-70b": ModelConfig(
        name="llama3.1-70b",
        display_name="Llama 3.1 (70B)",
        total_params_b=70.0,
        active_params_b=70.0,
        num_layers=80,
        hidden_dim=8192,
        num_kv_heads=8,
        head_dim=128,
        is_moe=False,
        description="Llama 3.1 70B dense model with Grouped Query Attention."
    ),
    "llama3.1-405b": ModelConfig(
        name="llama3.1-405b",
        display_name="Llama 3.1 (405B)",
        total_params_b=405.0,
        active_params_b=405.0,
        num_layers=126,
        hidden_dim=16384,
        num_kv_heads=16,
        head_dim=128,
        is_moe=False,
        description="Llama 3.1 405B flagship dense model."
    ),
    "deepseek-v3": ModelConfig(
        name="deepseek-v3",
        display_name="DeepSeek V3 (671B MoE)",
        total_params_b=671.0,
        active_params_b=37.0,
        num_layers=61,
        hidden_dim=7168,
        num_kv_heads=128,
        head_dim=128,
        is_moe=True,
        description="DeepSeek V3 671B total params with 37B active params."
    )
}


def get_model_config(name: str) -> Optional[ModelConfig]:
    return PREDEFINED_MODELS.get(name.lower())
