"""
TPU Hardware definitions and specs for Roofline calculations.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HardwareConfig:
    name: str
    display_name: str
    tflops: float              # Peak TFLOPS (BF16 / INT8) per chip
    memory_bw_tbs: float        # Memory Bandwidth in TB/s per chip
    hbm_gb: float               # HBM capacity per chip in GB
    interconnect_bw_gbs: float  # ICI Interconnect Bandwidth in GB/s per chip
    description: str = ""

    @property
    def flops_per_sec(self) -> float:
        """Flops per second per chip"""
        return self.tflops * 1e12

    @property
    def memory_bw_bytes_per_sec(self) -> float:
        """Bytes per second per chip"""
        return self.memory_bw_tbs * 1e12

    @property
    def hbm_bytes(self) -> float:
        """HBM capacity in bytes per chip"""
        return self.hbm_gb * 1e9


PREDEFINED_HARDWARE: Dict[str, HardwareConfig] = {
    "tpu-trillium": HardwareConfig(
        name="tpu-trillium",
        display_name="TPU Trillium / v6e (4604 TFLOPS, 7.3 TB/s)",
        tflops=4604.0,
        memory_bw_tbs=7.3,
        hbm_gb=32.0,
        interconnect_bw_gbs=1600.0,
        description="TPU Trillium offering 4604 TFLOPS peak compute and 7.3 TB/s HBM bandwidth per chip."
    ),
    "tpu-v5p": HardwareConfig(
        name="tpu-v5p",
        display_name="TPU v5p (459 TFLOPS, 4.8 TB/s)",
        tflops=459.0,
        memory_bw_tbs=4.8,
        hbm_gb=95.0,
        interconnect_bw_gbs=4800.0,
        description="TPU v5p pod chip with 459 TFLOPS and 4.8 TB/s memory bandwidth."
    ),
    "tpu-v5e": HardwareConfig(
        name="tpu-v5e",
        display_name="TPU v5e (197 TFLOPS, 819 GB/s)",
        tflops=197.0,
        memory_bw_tbs=0.819,
        hbm_gb=16.0,
        interconnect_bw_gbs=800.0,
        description="TPU v5e cost-optimized chip."
    ),
    "tpu-v4": HardwareConfig(
        name="tpu-v4",
        display_name="TPU v4 (275 TFLOPS, 1.2 TB/s)",
        tflops=275.0,
        memory_bw_tbs=1.2,
        hbm_gb=32.0,
        interconnect_bw_gbs=1200.0,
        description="TPU v4 architecture chip."
    )
}


def get_hardware_config(name: str) -> Optional[HardwareConfig]:
    return PREDEFINED_HARDWARE.get(name.lower())
