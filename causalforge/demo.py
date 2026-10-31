from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request
import webbrowser

from .config import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


def preflight(host: str = "127.0.0.1", port: int = 8000, check_meta: bool = True) -> dict:
    load_dotenv(ROOT / ".env")
    checks = {"python": {"ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]}}
    required = [ROOT / "web" / name for name in ("index.html", "app.js", "styles.css")]
    checks["web_assets"] = {"ok": all(path.is_file() for path in required), "detail": f"{sum(path.is_file() for path in required)}/{len(required)} present"}
    replays = list((ROOT / "replays").glob("*.json")) if (ROOT / "replays").is_dir() else []
    checks["offline_replay"] = {"ok": bool(replays), "detail": f"{len(replays)} replay(s)"}
    has_key = bool(os.environ.get("MODEL_API_KEY"))
    checks["api_key"] = {"ok": has_key, "detail": "configured" if has_key else "missing; replay-only mode available"}
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        occupied = sock.connect_ex((host, port)) == 0
        port_ok, detail = not occupied, "in use by an active listener" if occupied else "available"
    except OSError as error:
        port_ok, detail = True, f"available ({error})"
    finally:
        sock.close()
    checks["port"] = {"ok": port_ok, "detail": detail}
    if check_meta:
        try:
            with urllib.request.urlopen("https://api.meta.ai/v1/status", timeout=5) as response:
                status = response.status
            checks["meta_status"] = {"ok": status == 200, "detail": f"HTTP {status}"}
        except Exception as error:
            checks["meta_status"] = {"ok": False, "detail": f"unavailable; replay ready ({error})"}
    live_ready = all(checks[name]["ok"] for name in ("python", "web_assets", "offline_replay", "api_key", "port"))
    replay_ready = all(checks[name]["ok"] for name in ("python", "web_assets", "offline_replay", "port"))
    return {"live_ready": live_ready, "replay_ready": replay_ready, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight and launch the CausalForge demo")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--skip-meta-status", action="store_true"); parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    report = preflight(args.host, args.port, not args.skip_meta_status)
    print(json.dumps(report, indent=2))
    if not report["replay_ready"]:
        raise SystemExit("Preflight failed: neither live nor replay demo can start")
    if args.preflight_only:
        raise SystemExit(0 if report["live_ready"] else 2)
    if args.open:
        webbrowser.open(f"http://{args.host}:{args.port}")
    from .server import main as serve
    sys.argv = [sys.argv[0], "--host", args.host, "--port", str(args.port)]
    serve()


if __name__ == "__main__":
    main()
