"""
HTTP REST Server & Static File Server for TPU Simulator.
Listens on localhost (127.0.0.1).
"""

import json
import os
import sys
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from engine.models import PREDEFINED_MODELS, ModelConfig, get_model_config
from engine.hardware import PREDEFINED_HARDWARE, HardwareConfig, get_hardware_config
from engine.roofline import SimulationConfig, RooflineEngine, STRATEGIES


STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

# Dynamic custom models registered via API
CUSTOM_MODELS: dict[str, ModelConfig] = {}
CUSTOM_HARDWARE: dict[str, HardwareConfig] = {}


class TPUSimRequestHandler(BaseHTTPRequestHandler):
    def _send_headers(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.end_headers()

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/models":
            self._handle_get_models()
        elif path == "/api/hardware":
            self._handle_get_hardware()
        elif path == "/api/strategies":
            self._handle_get_strategies()
        else:
            # Serve static files
            self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/simulate":
            self._handle_post_simulate()
        elif path == "/api/models":
            self._handle_post_model()
        elif path == "/api/hardware":
            self._handle_post_hardware()
        else:
            self._send_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def _handle_get_models(self):
        all_models = {}
        for name, cfg in PREDEFINED_MODELS.items():
            all_models[name] = {
                "name": cfg.name,
                "display_name": cfg.display_name,
                "total_params_b": cfg.total_params_b,
                "active_params_b": cfg.active_params_b,
                "num_layers": cfg.num_layers,
                "hidden_dim": cfg.hidden_dim,
                "num_kv_heads": cfg.num_kv_heads,
                "head_dim": cfg.head_dim,
                "is_moe": cfg.is_moe,
                "description": cfg.description,
            }
        for name, cfg in CUSTOM_MODELS.items():
            all_models[name] = {
                "name": cfg.name,
                "display_name": cfg.display_name,
                "total_params_b": cfg.total_params_b,
                "active_params_b": cfg.active_params_b,
                "num_layers": cfg.num_layers,
                "hidden_dim": cfg.hidden_dim,
                "num_kv_heads": cfg.num_kv_heads,
                "head_dim": cfg.head_dim,
                "is_moe": cfg.is_moe,
                "description": cfg.description,
            }
        self._send_headers(200)
        self.wfile.write(json.dumps(all_models).encode("utf-8"))

    def _handle_get_hardware(self):
        all_hw = {}
        for name, cfg in PREDEFINED_HARDWARE.items():
            all_hw[name] = {
                "name": cfg.name,
                "display_name": cfg.display_name,
                "tflops": cfg.tflops,
                "memory_bw_tbs": cfg.memory_bw_tbs,
                "hbm_gb": cfg.hbm_gb,
                "interconnect_bw_gbs": cfg.interconnect_bw_gbs,
                "description": cfg.description,
            }
        for name, cfg in CUSTOM_HARDWARE.items():
            all_hw[name] = {
                "name": cfg.name,
                "display_name": cfg.display_name,
                "tflops": cfg.tflops,
                "memory_bw_tbs": cfg.memory_bw_tbs,
                "hbm_gb": cfg.hbm_gb,
                "interconnect_bw_gbs": cfg.interconnect_bw_gbs,
                "description": cfg.description,
            }
        self._send_headers(200)
        self.wfile.write(json.dumps(all_hw).encode("utf-8"))

    def _handle_get_strategies(self):
        strategies = list(STRATEGIES.keys()) + ["custom"]
        self._send_headers(200)
        self.wfile.write(json.dumps({"strategies": strategies}).encode("utf-8"))

    def _handle_post_model(self):
        try:
            body = self._read_json_body()
            name = body.get("name", "").strip().lower()
            if not name:
                raise ValueError("Model name is required")

            cfg = ModelConfig(
                name=name,
                display_name=body.get("display_name", name),
                total_params_b=float(body.get("total_params_b", 10.0)),
                active_params_b=float(body.get("active_params_b", 10.0)),
                num_layers=int(body.get("num_layers", 32)),
                hidden_dim=int(body.get("hidden_dim", 4096)),
                num_kv_heads=int(body.get("num_kv_heads", 8)),
                head_dim=int(body.get("head_dim", 128)),
                is_moe=bool(body.get("is_moe", False)),
                description=body.get("description", "Custom model"),
            )
            CUSTOM_MODELS[name] = cfg
            self._send_headers(200)
            self.wfile.write(json.dumps({"status": "success", "name": name}).encode("utf-8"))
        except Exception as e:
            self._send_headers(400)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _handle_post_hardware(self):
        try:
            body = self._read_json_body()
            name = body.get("name", "").strip().lower()
            if not name:
                raise ValueError("Hardware name is required")

            cfg = HardwareConfig(
                name=name,
                display_name=body.get("display_name", name),
                tflops=float(body.get("tflops", 1000.0)),
                memory_bw_tbs=float(body.get("memory_bw_tbs", 1.0)),
                hbm_gb=float(body.get("hbm_gb", 32.0)),
                interconnect_bw_gbs=float(body.get("interconnect_bw_gbs", 1000.0)),
                description=body.get("description", "Custom hardware"),
            )
            CUSTOM_HARDWARE[name] = cfg
            self._send_headers(200)
            self.wfile.write(json.dumps({"status": "success", "name": name}).encode("utf-8"))
        except Exception as e:
            self._send_headers(400)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _handle_post_simulate(self):
        try:
            body = self._read_json_body()

            # Resolve model
            model_name = body.get("model", "qwen3-397b").lower()
            model = CUSTOM_MODELS.get(model_name) or PREDEFINED_MODELS.get(model_name)
            if not model:
                # If inline custom model parameters passed
                if "model_custom" in body:
                    mc = body["model_custom"]
                    model = ModelConfig(
                        name="custom-model",
                        display_name=mc.get("display_name", "Custom Model"),
                        total_params_b=float(mc.get("total_params_b", 30.0)),
                        active_params_b=float(mc.get("active_params_b", 30.0)),
                        num_layers=int(mc.get("num_layers", 32)),
                        hidden_dim=int(mc.get("hidden_dim", 4096)),
                        num_kv_heads=int(mc.get("num_kv_heads", 8)),
                        head_dim=int(mc.get("head_dim", 128)),
                        is_moe=bool(mc.get("is_moe", False)),
                    )
                else:
                    raise ValueError(f"Unknown model: {model_name}")

            # Resolve hardware
            hw_name = body.get("hardware", "tpu-trillium").lower()
            hardware = CUSTOM_HARDWARE.get(hw_name) or PREDEFINED_HARDWARE.get(hw_name)
            if not hardware:
                if "hardware_custom" in body:
                    hc = body["hardware_custom"]
                    hardware = HardwareConfig(
                        name="custom-hw",
                        display_name=hc.get("display_name", "Custom TPU"),
                        tflops=float(hc.get("tflops", 4604.0)),
                        memory_bw_tbs=float(hc.get("memory_bw_tbs", 7.3)),
                        hbm_gb=float(hc.get("hbm_gb", 32.0)),
                        interconnect_bw_gbs=float(hc.get("interconnect_bw_gbs", 1600.0)),
                    )
                else:
                    raise ValueError(f"Unknown hardware: {hw_name}")

            num_tpus = int(body.get("num_tpus", 16))
            prefill_ratio = float(body.get("prefill_ratio", 0.5)) # fraction of TPUs for prefill
            prefill_tpus = int(body.get("prefill_tpus", max(1, int(num_tpus * prefill_ratio))))
            decode_tpus = int(body.get("decode_tpus", max(1, num_tpus - prefill_tpus)))

            input_len = int(body.get("input_len", 2048))
            output_len = int(body.get("output_len", 512))
            bytes_per_param = float(body.get("bytes_per_param", 2.0))
            include_kv_cache = bool(body.get("include_kv_cache", True))
            strategy_name = body.get("strategy_name", "standard")

            custom_prefill_formula = body.get("custom_prefill_formula")
            custom_decode_formula = body.get("custom_decode_formula")

            sim_config = SimulationConfig(
                model=model,
                hardware=hardware,
                num_tpus=num_tpus,
                prefill_tpus=prefill_tpus,
                decode_tpus=decode_tpus,
                input_len=input_len,
                output_len=output_len,
                bytes_per_param=bytes_per_param,
                include_kv_cache=include_kv_cache,
                strategy_name=strategy_name,
                custom_prefill_formula=custom_prefill_formula,
                custom_decode_formula=custom_decode_formula,
            )

            batch_sizes = body.get("batch_sizes")
            if batch_sizes:
                batch_sizes = [int(b) for b in batch_sizes]

            engine = RooflineEngine()
            result = engine.run_simulation(sim_config, batch_sizes=batch_sizes)

            # Format response JSON
            res_dict = {
                "config": result.config,
                "knee_point": result.knee_point.__dict__ if result.knee_point else None,
                "max_interactivity_tps": result.max_interactivity_tps,
                "max_throughput_tps": result.max_throughput_tps,
                "datapoints": [dp.__dict__ for dp in result.datapoints],
            }

            self._send_headers(200)
            self.wfile.write(json.dumps(res_dict).encode("utf-8"))

        except Exception as e:
            self._send_headers(400)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _serve_static(self, rel_path: str):
        if rel_path in ("/", "", "/index.html"):
            rel_path = "/index.html"

        # Sanitize path to prevent directory traversal
        filename = rel_path.lstrip("/")
        file_path = os.path.abspath(os.path.join(STATIC_DIR, filename))

        if not file_path.startswith(STATIC_DIR + os.sep) and file_path != STATIC_DIR:
            self._send_headers(403)
            self.wfile.write(b"Forbidden")
            return

        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self._send_headers(404)
            self.wfile.write(b"404 Not Found")
            return

        # Determine MIME type
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }
        ctype = content_types.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._send_headers(200, content_type=ctype)
            self.wfile.write(content)
        except Exception as e:
            self._send_headers(500)
            self.wfile.write(f"Server Error: {e}".encode("utf-8"))


def run_server(port=8080):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, TPUSimRequestHandler)
    print(f"TPU Simulator Server running on http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPU Simulator Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()
    run_server(port=args.port)
