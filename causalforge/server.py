from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlparse

from .budget import SpendLedger
from .config import load_dotenv
from .meta_api import MetaAPIError, MuseSparkClient
from .orchestrator import BudgetExceeded, DiscoveryRun
from .presentation import render_state
from .replay import ReplaySession


WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
REPLAY_ROOT = Path(__file__).resolve().parent.parent / "replays"


class ApplicationState:
    def __init__(self, runs_dir: Path = Path("runs")):
        self.runs_dir = runs_dir
        self.lock = threading.RLock()
        self.run: DiscoveryRun | None = None
        self.last_event: dict | None = None
        self.revealed: dict | None = None
        self.busy = False
        self.replay: ReplaySession | None = None

    def reset(self, seed: int, experiments: int = 10, max_cost_usd: float = 1.0) -> dict:
        if not 1 <= experiments <= 30:
            raise ValueError("experiments must be between 1 and 30")
        with self.lock:
            if self.busy:
                raise RuntimeError("an experiment is currently running")
            ledger = SpendLedger(self.runs_dir / "meta-spend.json", 120.0)
            self.replay = None
            self.run = DiscoveryRun(
                seed=seed,
                client=MuseSparkClient(),
                max_experiments=experiments,
                max_cost_usd=max_cost_usd,
                log_path=self.runs_dir / f"web-seed-{seed}.jsonl",
                spend_ledger=ledger,
            )
            self.last_event = None
            self.revealed = None
            return self.snapshot()

    def step(self) -> dict:
        with self.lock:
            if self.replay:
                self.replay.step()
                return self.replay.snapshot()
            if not self.run:
                raise RuntimeError("reset a world before running an experiment")
            if self.busy:
                raise RuntimeError("an experiment is already running")
            self.busy = True
        try:
            event = self.run.step()
            with self.lock:
                self.last_event = event
                self.busy = False
                return self.snapshot()
        finally:
            with self.lock:
                self.busy = False

    def reveal(self) -> dict:
        with self.lock:
            if self.replay:
                self.replay.reveal()
                return self.replay.snapshot()
            if not self.run:
                raise RuntimeError("no world exists")
            self.revealed = self.run.simulator.reveal()
            return self.snapshot()

    def snapshot(self) -> dict:
        with self.lock:
            if self.replay:
                return self.replay.snapshot()
            if not self.run:
                return {"status": "empty", "busy": self.busy, "render": render_state(None)}
            observations = self.run.simulator.observations
            latest = observations[-1] if observations else None
            return {
                "status": "complete" if len(observations) >= self.run.max_experiments else "ready",
                "busy": self.busy,
                "mode": "live",
                "seed": self.run.simulator.public.seed,
                "experiment_count": len(observations),
                "max_experiments": self.run.max_experiments,
                "public_world": self.run.simulator.public.as_dict(),
                "observations": [item.public_dict() for item in observations],
                "learner": self.run.learner.summary(),
                "last_event": self.last_event,
                "total_cost_usd": round(self.run.total_cost_usd, 8),
                "revealed_rule": self.revealed,
                "render": render_state(latest),
            }

    def available_replays(self) -> list[dict]:
        items = []
        if REPLAY_ROOT.is_dir():
            for path in sorted(REPLAY_ROOT.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                items.append({"id": path.stem, "title": payload.get("title", path.stem), "seed": payload.get("seed")})
        return items

    def load_replay(self, replay_id: str) -> dict:
        if not replay_id.replace("-", "").isalnum():
            raise ValueError("invalid replay id")
        path = REPLAY_ROOT / f"{replay_id}.json"
        if not path.is_file():
            raise ValueError("replay not found")
        with self.lock:
            if self.busy:
                raise RuntimeError("an experiment is currently running")
            self.run = None
            self.replay = ReplaySession(json.loads(path.read_text(encoding="utf-8")))
            return self.replay.snapshot()


class CausalForgeHandler(BaseHTTPRequestHandler):
    app: ApplicationState

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._json(self.app.snapshot())
            return
        if path == "/api/health":
            self._json({"ok": True})
            return
        if path == "/api/replays":
            self._json({"replays": self.app.available_replays()})
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        candidate = (WEB_ROOT / relative).resolve()
        try:
            candidate.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/run/reset":
                body = self._body()
                result = self.app.reset(int(body.get("seed", 42)), int(body.get("experiments", 10)), float(body.get("max_cost_usd", 1.0)))
            elif path == "/api/run/step":
                result = self.app.step()
            elif path == "/api/run/reveal":
                result = self.app.reveal()
            elif path == "/api/replay/load":
                result = self.app.load_replay(str(self._body().get("id", "")))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._json(result)
        except (ValueError, RuntimeError, MetaAPIError, BudgetExceeded, StopIteration) as error:
            self._json({"error": str(error), "state": self.app.snapshot()}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Serve the CausalForge Phase 2 visualization")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    handler = type("ConfiguredHandler", (CausalForgeHandler,), {"app": ApplicationState()})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"CausalForge available at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
