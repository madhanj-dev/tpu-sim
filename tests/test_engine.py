"""
Unit tests for TPU simulator engine.
"""

import unittest
from engine.models import get_model_config, PREDEFINED_MODELS, ModelConfig
from engine.hardware import get_hardware_config, PREDEFINED_HARDWARE
from engine.roofline import (
    SimulationConfig,
    RooflineEngine,
    StandardRooflineStrategy,
    DetailedRooflineStrategy,
    DynamicFormulaRooflineStrategy,
)


class TestEngine(unittest.TestCase):
    def test_predefined_models(self):
        qwen = get_model_config("qwen3-397b")
        self.assertIsNotNone(qwen)
        self.assertEqual(qwen.total_params_b, 397.0)
        self.assertEqual(qwen.active_params_b, 39.0)

        gemma = get_model_config("gemma4")
        self.assertIsNotNone(gemma)
        self.assertEqual(gemma.active_params_b, 31.0)

    def test_hardware_specs(self):
        trillium = get_hardware_config("tpu-trillium")
        self.assertIsNotNone(trillium)
        self.assertEqual(trillium.tflops, 4604.0)
        self.assertEqual(trillium.memory_bw_tbs, 7.3)

    def test_simulation_roofline(self):
        model = get_model_config("qwen3-397b")
        hw = get_hardware_config("tpu-trillium")

        cfg = SimulationConfig(
            model=model,
            hardware=hw,
            num_tpus=16,
            prefill_tpus=8,
            decode_tpus=8,
            input_len=2048,
            output_len=512,
            bytes_per_param=2.0,
            strategy_name="standard"
        )

        engine = RooflineEngine()
        res = engine.run_simulation(cfg, batch_sizes=[1, 16, 64, 256])

        self.assertEqual(len(res.datapoints), 4)
        # Check batch 1 interactivity > 0
        dp1 = res.datapoints[0]
        self.assertGreater(dp1.interactivity_tps, 0.0)
        self.assertGreater(dp1.output_throughput_tps, 0.0)

        # Batch 256 should have higher total system throughput than batch 1
        dp256 = res.datapoints[3]
        self.assertGreater(dp256.output_throughput_tps, dp1.output_throughput_tps)

    def test_custom_formula_strategy(self):
        model = get_model_config("gemma4")
        hw = get_hardware_config("tpu-trillium")

        # Custom formula for prefill and decode
        prefill_f = "(batch_size * active_params * 2 * input_len) / (prefill_tpus * tflops)"
        decode_f = "(active_params * bytes_per_param) / (decode_tpus * memory_bw) + 0.001"

        cfg = SimulationConfig(
            model=model,
            hardware=hw,
            num_tpus=4,
            prefill_tpus=2,
            decode_tpus=2,
            input_len=1024,
            output_len=256,
            custom_prefill_formula=prefill_f,
            custom_decode_formula=decode_f,
        )

        engine = RooflineEngine()
        res = engine.run_simulation(cfg, batch_sizes=[1, 4])
        self.assertEqual(len(res.datapoints), 2)
        # TTFT for batch 1
        expected_ttft_sec = (1 * model.active_params * 2 * 1024) / (2 * hw.flops_per_sec)
        self.assertAlmostEqual(res.datapoints[0].ttft_ms, expected_ttft_sec * 1000.0, places=1)


if __name__ == "__main__":
    unittest.main()
