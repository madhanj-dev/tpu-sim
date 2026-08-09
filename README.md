# tpu-sim: TPU Roofline & Throughput Simulator

`tpu-sim` is a simulator for analyzing Large Language Model (LLM) inference performance on TPU clusters using a disaggregated prefill/decode architecture and roofline performance models.

It features an interactive Web GUI, Python calculation engine, unit test suite, and REST APIs for dynamic algorithm customization.

---

## Key Features

1. **Model Selection**: Predefined specs for models including:
   - `qwen3-397b`: Qwen3 397B MoE (39B active parameters per token).
   - `gemma4`: Gemma 4 (31B active parameters).
   - `gemma2-27b`: Gemma 2 (27B dense).
   - `llama3.1-70b` & `llama3.1-405b`: Llama 3.1 dense models.
   - `deepseek-v3`: DeepSeek V3 (671B MoE, 37B active parameters).
   - Dynamic custom model definition via UI or API.

2. **TPU Hardware Specs**:
   - `tpu-trillium` (TPU v6e): Peak 4,604 TFLOPS, 7.3 TB/s Memory Bandwidth per chip.
   - `tpu-v5p`: 459 TFLOPS, 4.8 TB/s Memory Bandwidth per chip.
   - `tpu-v5e`: 197 TFLOPS, 819 GB/s Memory Bandwidth per chip.
   - `tpu-v4`: 275 TFLOPS, 1.2 TB/s Memory Bandwidth per chip.

3. **Disaggregated Prefill / Decode Model**:
   - Split total TPUs into Prefill pool ($N_{prefill}$) and Decode pool ($N_{decode}$).
   - **Prefill (Compute Bound)**:
     $$\text{Prefill FLOPS} = B_{prefill} \times 2 \times P_{active} \times L_{in}$$
     $$\text{TTFT} = \frac{\text{Prefill FLOPS}}{N_{prefill} \times \text{TFLOPS}_{tpu}}$$
   - **Decode (Memory Bandwidth Bound)**:
     $$\text{Weight Load Time} = \frac{P_{active} \times \text{bytes\_per\_param}}{N_{decode} \times \text{BW}_{tpu}}$$
     $$\text{Decode Compute Time} = \frac{B_{decode} \times 2 \times P_{active}}{N_{decode} \times \text{TFLOPS}_{tpu}}$$
     $$\text{TPOT} = \max(\text{Weight Load Time} + \text{KV Cache Transfer Time}, \text{Decode Compute Time})$$
   - **Interactivity**: $1 / \text{TPOT}$ (tokens/sec per user stream).
   - **Throughput**: $B_{decode} \times \text{Interactivity}$ (output tokens/sec total).

4. **Extensible Strategy & Custom Formula API**:
   - Easily register custom Python strategy classes or send custom math expressions via API / Web UI.

---

## Quick Start

### 1. Run Unit Tests
```bash
python3 -m unittest discover -s tests
```

### 2. Start the Simulator Server
```bash
python3 server.py --port 8080
```

Open your browser and navigate to:
```
http://127.0.0.1:8080
```

---

## REST API Reference

### Get Models
`GET /api/models`

### Get Hardware Specs
`GET /api/hardware`

### Run Simulation
`POST /api/simulate`

Example Payload:
```json
{
  "model": "qwen3-397b",
  "hardware": "tpu-trillium",
  "num_tpus": 16,
  "prefill_tpus": 8,
  "decode_tpus": 8,
  "input_len": 2048,
  "output_len": 512,
  "bytes_per_param": 2.0,
  "strategy_name": "standard"
}
```

### Run Custom Formula Simulation
```json
{
  "model": "gemma4",
  "hardware": "tpu-trillium",
  "num_tpus": 8,
  "prefill_tpus": 4,
  "decode_tpus": 4,
  "input_len": 1024,
  "output_len": 256,
  "strategy_name": "custom",
  "custom_prefill_formula": "(batch_size * active_params * 2 * input_len) / (prefill_tpus * tflops)",
  "custom_decode_formula": "max((active_params * bytes_per_param) / (decode_tpus * memory_bw), (batch_size * 2 * active_params) / (decode_tpus * tflops))"
}
```
