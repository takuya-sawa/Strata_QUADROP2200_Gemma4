import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from gemma4_server import Engine, Stats, make_handler


class FakeProcess:
    def __init__(self, *args, **kwargs):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(
            "INFO expert_slots=12 experts=384 expert_cache_mib=256 expert_layers_gpu=3 expert_layers_total=30\n"
            "READY 1024 stop\n"
            "PP 5 10 100 50\nT 42\nT 99\n"
            "DONE 2 10 100 250 stop 0 0 4 3 6 0 0 0.0 6 2\n"
        )

    def poll(self):
        return None


class FakeTokenizer:
    def encode(self, text, parse_special=True):
        return list(range(10))

    def decode(self, tokens):
        return "hello" if tokens else ""


class FakeTemplate:
    def render(self, messages, tools):
        return "prompt"


class MonitorTest(unittest.TestCase):
    def test_metrics_and_stream(self):
        with patch("gemma4_server.subprocess.Popen", FakeProcess):
            engine = Engine("strata", "model", 1024, [])
        stats = Stats(1024)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
            engine, FakeTokenizer(), FakeTemplate(), (99,), "test", stats, None, experts=128))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        request = urllib.request.Request(base + "/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                      if line.startswith("data: {")]
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 4)
        self.assertEqual(events[-1]["timings"]["predicted_per_second"], 8)
        with urllib.request.urlopen(base + "/metrics?requests=all") as response:
            metrics = json.load(response)
        self.assertEqual(metrics["api"], {"host": "127.0.0.1", "port": server.server_port, "authentication": "none"})
        self.assertEqual(metrics["engine"]["expert_slots"], 12)
        self.assertEqual(metrics["engine"]["experts"], 384)
        self.assertEqual(metrics["engine"]["expert_cache_mib"], 256)
        self.assertEqual(metrics["engine"]["expert_layers_gpu"], 3)
        self.assertEqual(metrics["engine"]["expert_layers_total"], 30)
        self.assertEqual(metrics["live"]["state"], "idle")
        self.assertEqual(metrics["requests"][0]["reused"], 4)
        self.assertEqual(metrics["requests"][0]["hit_rate"], 0.5)
        self.assertEqual(metrics["requests"][0]["pcie_share"], 0.25)
        self.assertEqual(metrics["totals"]["output_tokens"], 2)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/monitor")
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
