"""Regression checks for the strict realtime generation API contract.

Uses a local HTTP server, without importing the application's model instances.
Run: python3 -m unittest discover -s backend/tests -p test_motion_backend_contract.py
"""

import importlib.util
import json
import sys
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "app/agent/general_conversation/tools/motion_backends.py"
)
PACKAGE_NAME = "motion_backend_contract_tools"
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(MODULE_PATH.parent)]
sys.modules[PACKAGE_NAME] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE_NAME + ".motion_backends", MODULE_PATH
)
BACKENDS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKENDS
SPEC.loader.exec_module(BACKENDS)
BVH = "HIERARCHY\nROOT Hips\nMOTION\nFrames: 2\nFrame Time: 0.033333\n0 0 0 0 0 0\n0 0 0 0 30 0\n"


class RealtimeContractTest(unittest.TestCase):
    def setUp(self):
        self.loaded = False
        self.drop_next_generation = False
        self.loads = 0
        self.generated_texts = []
        self.generated_payloads = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, data):
                payload = (
                    json.dumps(data).encode()
                    if isinstance(data, dict)
                    else data.encode()
                )
                self.send_response(code)
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/api/realtime/status":
                    self.reply(
                        200, {"current_model": "t2m_gpt" if owner.loaded else None}
                    )
                elif self.path == "/bvh/wave.bvh":
                    self.reply(200, BVH)
                else:
                    self.reply(404, {"detail": "Unknown route"})

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/realtime/load":
                    if data.get("model") != "t2m_gpt":
                        self.reply(400, {"detail": "Unsupported model"})
                        return
                    owner.loaded = True
                    owner.loads += 1
                    self.reply(200, {"status": "success"})
                elif self.path == "/api/realtime/generate":
                    # Mirrors the independent server's extra="forbid" request schema.
                    if "text" not in data or set(data) - {
                        "text",
                        "cfg_scale",
                        "top_k",
                        "temperature",
                    }:
                        self.reply(422, {"detail": "Extra inputs are not permitted"})
                        return
                    if owner.drop_next_generation:
                        owner.drop_next_generation = False
                        owner.loaded = False
                    if not owner.loaded:
                        self.reply(409, {"detail": "T2M-GPT is not loaded"})
                        return
                    owner.generated_texts.append(data["text"])
                    owner.generated_payloads.append(data)
                    self.reply(200, {"bvh_path": "wave.bvh", "latency_seconds": 0.2})
                else:
                    self.reply(404, {"detail": "Unknown route"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        with patch.dict(
            "os.environ",
            {
                "T2M_SERVER_URL": f"http://127.0.0.1:{self.server.server_port}",
                "T2M_CFG_SCALE": "",
                "T2M_TOP_K": "",
                "T2M_TEMPERATURE": "",
            },
        ):
            self.backend = BACKENDS.CustomT2MBackend()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_load_generate_and_download_against_strict_schema(self):
        result = self.backend.generate(BACKENDS.MotionRequest(prompt="手を振る"))
        self.assertEqual(result.bvh, BVH)
        self.assertEqual(self.generated_texts, ["手を振る"])
        self.assertEqual(self.generated_payloads, [{"text": "手を振る"}])
        self.assertEqual(self.loads, 1)

    def test_configured_sampling_options_are_forwarded_without_model(self):
        with patch.dict(
            "os.environ",
            {
                "T2M_SERVER_URL": self.backend.server_url,
                "T2M_CFG_SCALE": "7",
                "T2M_TOP_K": "50",
                "T2M_TEMPERATURE": "0.8",
            },
        ):
            backend = BACKENDS.CustomT2MBackend()
        result = backend.generate(
            BACKENDS.MotionRequest(prompt="左手を胸の前で小さく振る")
        )
        self.assertEqual(result.bvh, BVH)
        self.assertEqual(
            self.generated_payloads,
            [
                {
                    "text": "左手を胸の前で小さく振る",
                    "cfg_scale": 7.0,
                    "top_k": 50,
                    "temperature": 0.8,
                }
            ],
        )

    def test_reload_once_when_loaded_model_is_lost(self):
        self.backend.ensure_model_loaded()
        self.drop_next_generation = True
        result = self.backend.generate(BACKENDS.MotionRequest(prompt="お辞儀する"))
        self.assertEqual(result.bvh, BVH)
        self.assertEqual(self.loads, 2)
        self.assertEqual(self.generated_texts, ["お辞儀する"])


class SamplingConfigurationTest(unittest.TestCase):
    def test_invalid_sampling_configuration_fails_before_any_http_request(self):
        invalid = {
            "T2M_CFG_SCALE": ["nan", "inf", "-inf", "-0.1", "10.1", "invalid"],
            "T2M_TEMPERATURE": ["nan", "inf", "0", "0.049", "5.01", "invalid"],
            "T2M_TOP_K": ["-1", "1.5", "nan", "inf", "invalid"],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), patch.dict(
                    "os.environ",
                    {
                        "T2M_CFG_SCALE": "",
                        "T2M_TOP_K": "",
                        "T2M_TEMPERATURE": "",
                        name: value,
                    },
                ):
                    with self.assertRaisesRegex(ValueError, name):
                        BACKENDS.CustomT2MBackend()

    def test_server_contract_boundaries_and_zero_top_k_are_accepted(self):
        for cfg, temperature in [("0", "0.05"), ("10", "5")]:
            with self.subTest(cfg=cfg, temperature=temperature), patch.dict(
                "os.environ",
                {
                    "T2M_CFG_SCALE": cfg,
                    "T2M_TOP_K": "0",
                    "T2M_TEMPERATURE": temperature,
                },
            ):
                backend = BACKENDS.CustomT2MBackend()
                self.assertEqual(
                    backend.generation_options,
                    {
                        "cfg_scale": float(cfg),
                        "temperature": float(temperature),
                        "top_k": 0,
                    },
                )

    def test_unset_overrides_do_not_add_generation_options(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(BACKENDS.CustomT2MBackend().generation_options, {})


if __name__ == "__main__":
    unittest.main()
