import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from causalforge.meta_api import MuseSparkClient


def valid_response() -> bytes:
    return json.dumps({
        "id": "local-resilience-test",
        "model": "muse-spark-1.3",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"tool_calls": [{
                "function": {
                    "name": "run_experiment",
                    "arguments": json.dumps({
                        "temperature": 1, "pressure": 0, "catalyst": -1,
                        "field": 0, "rationale": "resilience test",
                    }),
                }
            }]},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }).encode("utf-8")


class ResilienceHandler(BaseHTTPRequestHandler):
    requests = 0
    mode = "rate_limit"

    def do_POST(self):
        type(self).requests += 1
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if type(self).mode == "rate_limit" and type(self).requests == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"slow down"}}')
            return
        if type(self).mode == "malformed" and type(self).requests == 1:
            body = json.dumps({
                "id": "malformed",
                "model": "muse-spark-1.3",
                "choices": [{"finish_reason": "stop", "message": {"content": "I should explain first."}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            }).encode("utf-8")
        else:
            body = valid_response()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class MetaResilienceTests(unittest.TestCase):
    def setUp(self):
        ResilienceHandler.requests = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ResilienceHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = MuseSparkClient(
            api_key="local-test-key",
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            timeout=1,
            transport_retries=2,
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_rate_limit_retries_then_uses_real_returned_tool_call(self):
        ResilienceHandler.mode = "rate_limit"
        decision = self.client.choose_experiment({"test": "rate limit"})
        self.assertEqual(ResilienceHandler.requests, 2)
        self.assertEqual(decision.intervention.temperature, 1)
        self.assertEqual(decision.api_calls, 1)

    def test_malformed_response_gets_bounded_correction(self):
        ResilienceHandler.mode = "malformed"
        decision = self.client.choose_experiment({"test": "malformed call"})
        self.assertEqual(ResilienceHandler.requests, 2)
        self.assertEqual(decision.api_calls, 2)
        self.assertEqual(decision.intervention.catalyst, -1)
