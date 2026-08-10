"""
Roofline Simulation Engine for Disaggregated LLM Inference on TPU Clusters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import math

from .models import ModelConfig
from .hardware import HardwareConfig


@dataclass
class SimulationConfig:
    model: ModelConfig
    hardware: HardwareConfig
    num_tpus: int = 16              # Total TPUs in cluster
    prefill_tpus: int = 8           # TPUs allocated to Prefill pool
    decode_tpus: int = 8            # TPUs allocated to Decode pool
    input_len: int = 2048           # Input prompt sequence length
    output_len: int = 512           # Output decode sequence length
    bytes_per_param: float = 2.0    # 2.0 for BF16/FP16, 1.0 for INT8/FP8, 0.5 for INT4
    include_kv_cache: bool = True   # Include KV cache memory bandwidth in decode
    strategy_name: str = "standard" # "standard", "detailed", or "custom"
    custom_prefill_formula: Optional[str] = None
    custom_decode_formula: Optional[str] = None

    def __post_init__(self):
        # Validate prefill/decode split
        if self.prefill_tpus + self.decode_tpus > self.num_tpus:
            # Scale proportionally if sum exceeds total
            self.prefill_tpus = max(1, self.num_tpus // 2)
            self.decode_tpus = max(1, self.num_tpus - self.prefill_tpus)


@dataclass
class BatchDataPoint:
    batch_size: int                # Concurrency (batch size)
    ttft_ms: float                 # Time to First Token (ms)
    tpot_ms: float                 # Time Per Output Token (ms)
    interactivity_tps: float       # Throughput / s / user (tokens/sec/user)
    output_throughput_tps: float   # Total system output tokens/sec
    throughput_per_chip_tps: float # Throughput / s / chip (tokens/sec/chip)
    total_throughput_tps: float    # Total system prompt+output tokens/sec
    qps: float                     # Requests/sec completed
    e2e_latency_ms: float          # Total request latency (ms)
    prefill_is_compute_bound: bool
    decode_is_compute_bound: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    config: Dict[str, Any]
    knee_point: Optional[BatchDataPoint]
    max_interactivity_tps: float
    max_throughput_tps: float
    max_throughput_per_chip_tps: float
    datapoints: List[BatchDataPoint]


class RooflineStrategy:
    """Base class for roofline calculation strategies."""
    def compute_prefill_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        """Returns (prefill_time_seconds, is_compute_bound)"""
        raise NotImplementedError

    def compute_decode_step_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        """Returns (decode_step_time_seconds, is_compute_bound)"""
        raise NotImplementedError


class StandardRooflineStrategy(RooflineStrategy):
    """
    Standard roofline strategy following simple approximations:
    - Prefill: Compute-bound = (B * active_params * 2 * token_length) / Prefill_FLOPS
    - Decode: Memory-bound weight load time = (active_params * bytes_per_param) / Decode_BW
      vs Decode compute time = (B * 2 * active_params) / Decode_FLOPS
    """
    def compute_prefill_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        active_params = config.model.active_params
        total_prefill_flops = config.hardware.flops_per_sec * config.prefill_tpus
        total_prefill_bw = config.hardware.memory_bw_bytes_per_sec * config.prefill_tpus

        # Compute FLOPS needed
        prefill_flops = batch_size * 2.0 * active_params * config.input_len
        time_compute = prefill_flops / total_prefill_flops

        # Optional memory bound check
        model_bytes = active_params * config.bytes_per_param
        time_mem = model_bytes / total_prefill_bw

        time_prefill = max(time_compute, time_mem)
        is_compute_bound = time_compute >= time_mem
        return time_prefill, is_compute_bound

    def compute_decode_step_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        active_params = config.model.active_params
        total_decode_flops = config.hardware.flops_per_sec * config.decode_tpus
        total_decode_bw = config.hardware.memory_bw_bytes_per_sec * config.decode_tpus

        # Compute FLOPS needed for 1 decode token step across batch
        decode_flops = batch_size * 2.0 * active_params
        time_compute = decode_flops / total_decode_flops

        # Weight load memory transfer time (memory bound limit)
        weight_bytes = active_params * config.bytes_per_param
        time_weight_mem = weight_bytes / total_decode_bw

        # KV cache memory bandwidth if enabled
        time_kv_mem = 0.0
        if config.include_kv_cache:
            # Average KV cache size across context during decode
            avg_context_len = config.input_len + (config.output_len / 2.0)
            kv_bytes_per_token = (
                2.0 * config.model.num_layers * config.model.num_kv_heads * config.model.head_dim
                * config.bytes_per_param * avg_context_len * batch_size
            )
            time_kv_mem = kv_bytes_per_token / total_decode_bw

        time_mem = time_weight_mem + time_kv_mem

        time_decode_step = max(time_mem, time_compute)
        is_compute_bound = time_compute >= time_mem
        return time_decode_step, is_compute_bound


class DetailedRooflineStrategy(RooflineStrategy):
    """
    Detailed strategy including Tensor Parallelism communication overhead
    and quadratic attention FLOPs in prefill.
    """
    def compute_prefill_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        active_params = config.model.active_params
        total_prefill_flops = config.hardware.flops_per_sec * config.prefill_tpus
        total_prefill_bw = config.hardware.memory_bw_bytes_per_sec * config.prefill_tpus

        # Linear weights compute FLOPS
        weight_flops = batch_size * 2.0 * active_params * config.input_len
        # Quadratic attention FLOPS: 2 * batch * num_layers * seq_len^2 * hidden_dim
        attn_flops = batch_size * 4.0 * config.model.num_layers * (config.input_len ** 2) * config.model.hidden_dim
        total_flops = weight_flops + attn_flops

        time_compute = total_flops / total_prefill_flops

        # Prefill memory read
        weight_bytes = active_params * config.bytes_per_param
        time_mem = weight_bytes / total_prefill_bw

        # Interconnect sync latency penalty for TP split across chips
        tp_penalty = 0.0001 * math.log2(max(1, config.prefill_tpus))

        time_prefill = max(time_compute, time_mem) + tp_penalty
        is_compute_bound = time_compute >= time_mem
        return time_prefill, is_compute_bound

    def compute_decode_step_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        active_params = config.model.active_params
        total_decode_flops = config.hardware.flops_per_sec * config.decode_tpus
        total_decode_bw = config.hardware.memory_bw_bytes_per_sec * config.decode_tpus

        # FLOPS
        decode_flops = batch_size * 2.0 * active_params
        time_compute = decode_flops / total_decode_flops

        # Memory weight reading
        weight_bytes = active_params * config.bytes_per_param
        time_weight_mem = weight_bytes / total_decode_bw

        # KV cache transfer
        avg_context_len = config.input_len + (config.output_len / 2.0)
        kv_bytes = (
            2.0 * config.model.num_layers * config.model.num_kv_heads * config.model.head_dim
            * config.bytes_per_param * avg_context_len * batch_size
        )
        time_kv_mem = kv_bytes / total_decode_bw

        time_mem = time_weight_mem + time_kv_mem

        # Interconnect AllReduce latency penalty
        tp_penalty = 0.00005 * math.log2(max(1, config.decode_tpus))

        time_decode_step = max(time_mem, time_compute) + tp_penalty
        is_compute_bound = time_compute >= time_mem
        return time_decode_step, is_compute_bound


class DynamicFormulaRooflineStrategy(RooflineStrategy):
    """
    Evaluates custom formulas provided by user/API.
    Exposes safe math environment with variables:
    batch_size, active_params, total_params, input_len, output_len,
    tflops, memory_bw, bytes_per_param, num_tpus, prefill_tpus, decode_tpus.
    """
    def __init__(self, prefill_formula: Optional[str], decode_formula: Optional[str]):
        self.prefill_formula = prefill_formula
        self.decode_formula = decode_formula
        self.standard = StandardRooflineStrategy()

    def _eval_safe(self, formula: str, context: Dict[str, float]) -> float:
        allowed_globals = {"__builtins__": None, "math": math, "abs": abs, "min": min, "max": max}
        return float(eval(formula, allowed_globals, context))

    def compute_prefill_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        if not self.prefill_formula:
            return self.standard.compute_prefill_latency_sec(config, batch_size)

        ctx = {
            "batch_size": float(batch_size),
            "active_params": config.model.active_params,
            "total_params": config.model.total_params,
            "input_len": float(config.input_len),
            "output_len": float(config.output_len),
            "tflops": config.hardware.tflops * 1e12,
            "memory_bw": config.hardware.memory_bw_bytes_per_sec,
            "bytes_per_param": config.bytes_per_param,
            "prefill_tpus": float(config.prefill_tpus),
            "decode_tpus": float(config.decode_tpus),
            "num_tpus": float(config.num_tpus),
        }
        try:
            val = self._eval_safe(self.prefill_formula, ctx)
            return max(0.000001, val), True
        except Exception as e:
            return self.standard.compute_prefill_latency_sec(config, batch_size)

    def compute_decode_step_latency_sec(self, config: SimulationConfig, batch_size: int) -> tuple[float, bool]:
        if not self.decode_formula:
            return self.standard.compute_decode_step_latency_sec(config, batch_size)

        ctx = {
            "batch_size": float(batch_size),
            "active_params": config.model.active_params,
            "total_params": config.model.total_params,
            "input_len": float(config.input_len),
            "output_len": float(config.output_len),
            "tflops": config.hardware.tflops * 1e12,
            "memory_bw": config.hardware.memory_bw_bytes_per_sec,
            "bytes_per_param": config.bytes_per_param,
            "prefill_tpus": float(config.prefill_tpus),
            "decode_tpus": float(config.decode_tpus),
            "num_tpus": float(config.num_tpus),
        }
        try:
            val = self._eval_safe(self.decode_formula, ctx)
            return max(0.000001, val), True
        except Exception as e:
            return self.standard.compute_decode_step_latency_sec(config, batch_size)


# Strategy registry
STRATEGIES: Dict[str, Callable[..., RooflineStrategy]] = {
    "standard": StandardRooflineStrategy,
    "detailed": DetailedRooflineStrategy,
}


def register_strategy(name: str, strategy_factory: Callable[..., RooflineStrategy]):
    STRATEGIES[name.lower()] = strategy_factory


class RooflineEngine:
    """
    Roofline Simulation Engine that generates throughput vs. interactivity curves.
    """
    def __init__(self):
        pass

    def run_simulation(
        self,
        config: SimulationConfig,
        batch_sizes: Optional[List[int]] = None
    ) -> SimulationResult:
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

        # Select strategy
        if config.strategy_name == "custom" or config.custom_prefill_formula or config.custom_decode_formula:
            strategy = DynamicFormulaRooflineStrategy(config.custom_prefill_formula, config.custom_decode_formula)
        else:
            factory = STRATEGIES.get(config.strategy_name.lower(), StandardRooflineStrategy)
            strategy = factory()

        datapoints: List[BatchDataPoint] = []
        knee_point: Optional[BatchDataPoint] = None
        prev_compute_bound = False

        for b in batch_sizes:
            time_prefill_sec, prefill_cb = strategy.compute_prefill_latency_sec(config, b)
            time_decode_step_sec, decode_cb = strategy.compute_decode_step_latency_sec(config, b)

            ttft_ms = time_prefill_sec * 1000.0
            tpot_ms = time_decode_step_sec * 1000.0

            # Interactivity = Throughput / s / user (tokens/sec per user stream)
            interactivity_tps = 1.0 / time_decode_step_sec if time_decode_step_sec > 0 else 0.0

            # Total System Output Token Throughput (tokens/sec total)
            output_throughput_tps = b * interactivity_tps

            # Throughput / s / chip (tokens/sec/chip) across cluster
            throughput_per_chip_tps = output_throughput_tps / config.num_tpus if config.num_tpus > 0 else 0.0

            # Total End-to-End Latency
            e2e_latency_sec = time_prefill_sec + (config.output_len * time_decode_step_sec)
            e2e_latency_ms = e2e_latency_sec * 1000.0

            # Queries per second (QPS)
            qps = b / e2e_latency_sec if e2e_latency_sec > 0 else 0.0

            # Total Tokens Throughput (Prompt + Output)
            total_throughput_tps = qps * (config.input_len + config.output_len)

            dp = BatchDataPoint(
                batch_size=b,
                ttft_ms=round(ttft_ms, 3),
                tpot_ms=round(tpot_ms, 3),
                interactivity_tps=round(interactivity_tps, 2),
                output_throughput_tps=round(output_throughput_tps, 2),
                throughput_per_chip_tps=round(throughput_per_chip_tps, 2),
                total_throughput_tps=round(total_throughput_tps, 2),
                qps=round(qps, 3),
                e2e_latency_ms=round(e2e_latency_ms, 2),
                prefill_is_compute_bound=prefill_cb,
                decode_is_compute_bound=decode_cb,
                details={
                    "prefill_tpus": config.prefill_tpus,
                    "decode_tpus": config.decode_tpus,
                }
            )
            datapoints.append(dp)

            # Detect knee point where decode transitions to compute bound
            if decode_cb and not prev_compute_bound and knee_point is None:
                knee_point = dp
            prev_compute_bound = decode_cb

        max_interactivity = max((d.interactivity_tps for d in datapoints), default=0.0)
        max_throughput = max((d.output_throughput_tps for d in datapoints), default=0.0)
        max_throughput_per_chip = max((d.throughput_per_chip_tps for d in datapoints), default=0.0)

        config_dict = {
            "model_name": config.model.name,
            "model_display": config.model.display_name,
            "active_params_b": config.model.active_params_b,
            "total_params_b": config.model.total_params_b,
            "hardware_name": config.hardware.name,
            "hardware_display": config.hardware.display_name,
            "tflops_per_chip": config.hardware.tflops,
            "memory_bw_tbs_per_chip": config.hardware.memory_bw_tbs,
            "num_tpus": config.num_tpus,
            "prefill_tpus": config.prefill_tpus,
            "decode_tpus": config.decode_tpus,
            "input_len": config.input_len,
            "output_len": config.output_len,
            "bytes_per_param": config.bytes_per_param,
            "include_kv_cache": config.include_kv_cache,
            "strategy_name": config.strategy_name,
        }

        return SimulationResult(
            config=config_dict,
            knee_point=knee_point,
            max_interactivity_tps=max_interactivity,
            max_throughput_tps=max_throughput,
            max_throughput_per_chip_tps=max_throughput_per_chip,
            datapoints=datapoints,
        )
