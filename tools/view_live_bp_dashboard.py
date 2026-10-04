"""Localhost, display-only dashboard for the experimental BP viewer."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
import webbrowser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
except ImportError as exc:  # pragma: no cover - exercised by the launch environment
    raise SystemExit(
        "The dashboard requires Flask. Install it with: "
        "python -m pip install -r requirements-dashboard.txt"
    ) from exc

from line_transport import BLE_DEVICE_PREFIX
from live_bp_presentation import (
    PresentationSnapshot,
    RESEARCH_DISCLAIMER,
    build_connection_args,
    run_presentation_worker,
)
from view_live_bp import load_viewer_context


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_HISTORY_POINTS = 500
PARTICIPANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(__file__).resolve().with_name("dashboard_assets")


def discover_serial_devices() -> list[dict[str, str]]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [
        {"name": port.device, "address": port.device, "description": port.description or ""}
        for port in sorted(list_ports.comports(), key=lambda value: value.device)
    ]


def discover_ble_devices(timeout: float = 4.0) -> list[dict[str, str]]:
    try:
        from bleak import BleakScanner
    except ImportError as exc:
        raise RuntimeError(
            "BLE discovery requires Bleak. Install requirements-ble.txt first."
        ) from exc

    async def scan():
        return await BleakScanner.discover(timeout=timeout)

    devices = asyncio.run(scan())
    found: dict[str, dict[str, str]] = {}
    for device in devices:
        name = getattr(device, "name", None) or ""
        address = str(getattr(device, "address", ""))
        if name.startswith(BLE_DEVICE_PREFIX):
            found[address or name] = {"name": name, "address": address, "description": "BLE"}
    return sorted(found.values(), key=lambda item: (item["name"], item["address"]))


def discover_models(participant_id: str, root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    if participant_id and not PARTICIPANT_ID_PATTERN.fullmatch(participant_id):
        raise ValueError("Participant ID may contain only letters, numbers, underscores and hyphens.")
    results = []
    pattern = "data/processed/bp/*/single_subject/*/model_manifest.json"
    for manifest_path in root.glob(pattern):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_participant = str(manifest.get("participant_id", ""))
        if participant_id and model_participant.casefold() != participant_id.casefold():
            continue
        calibration = manifest.get("calibration", {})
        results.append(
            {
                "path": str(manifest_path.parent.resolve()),
                "participant_id": model_participant,
                "viewer_eligible": bool(manifest.get("viewer_eligible", False)),
                "population_validated": bool(manifest.get("population_validated", False)),
                "calibration_id": calibration.get("label_group_id"),
                "calibration_sbp": calibration.get("sbp"),
                "calibration_dbp": calibration.get("dbp"),
                "run_id": manifest_path.parents[2].name,
            }
        )
    return sorted(results, key=lambda item: (item["run_id"], item["path"]), reverse=True)


class DashboardSessionController:
    """Own exactly one transport worker and publish thread-safe dashboard state."""

    def __init__(self, *, worker=run_presentation_worker, clock=time.monotonic):
        self._worker_function = worker
        self._clock = clock
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._revision = 0
        self._history: list[dict[str, Any]] = []
        self._history_keys: set[tuple[Any, ...]] = set()
        self._snapshot: PresentationSnapshot | None = None
        self._session: dict[str, Any] = {}
        self._connection_state = "idle"
        self._error = ""

    def _notify(self) -> None:
        self._revision += 1
        self._changed.notify_all()

    def _append_history(self, snapshot: PresentationSnapshot) -> None:
        if snapshot.mode != "current" or snapshot.sbp is None or snapshot.dbp is None:
            return
        key = (
            snapshot.sensor_timestamp_ms,
            round(float(snapshot.sbp), 6),
            round(float(snapshot.dbp), 6),
            snapshot.source_status,
        )
        if key in self._history_keys:
            return
        self._history_keys.add(key)
        self._history.append(
            {
                "sensor_timestamp_ms": snapshot.sensor_timestamp_ms,
                "received_at": datetime.now(timezone.utc).isoformat(),
                "sbp": snapshot.sbp,
                "dbp": snapshot.dbp,
                "source_status": snapshot.source_status,
            }
        )
        if len(self._history) > MAX_HISTORY_POINTS:
            removed = self._history.pop(0)
            old_key = (
                removed["sensor_timestamp_ms"],
                round(float(removed["sbp"]), 6),
                round(float(removed["dbp"]), 6),
                removed["source_status"],
            )
            self._history_keys.discard(old_key)

    def _publish(self, kind: str, payload: object) -> None:
        with self._changed:
            if kind == "snapshot" and isinstance(payload, PresentationSnapshot):
                self._snapshot = payload
                self._connection_state = (
                    "connected" if payload.transport_connected else "reconnecting"
                )
                self._error = ""
                self._append_history(payload)
            elif kind in {"error", "model_error"}:
                self._error = str(payload)
                self._connection_state = "error"
                self._snapshot = None
            elif kind == "stopped":
                if self._connection_state not in {"error", "idle"}:
                    self._connection_state = "idle"
                self._thread = None
                self._stop = None
            self._notify()

    def connect(self, payload: dict[str, Any]) -> dict[str, Any]:
        participant_id = str(payload.get("participant_id", "")).strip()
        if not PARTICIPANT_ID_PATTERN.fullmatch(participant_id):
            raise ValueError(
                "Participant ID must be 1–32 letters, numbers, underscores or hyphens."
            )
        transport = str(payload.get("transport", "serial"))
        device = str(payload.get("device", "")).strip()
        mode = str(payload.get("mode", "pending"))
        if mode not in {"pending", "model"}:
            raise ValueError("Mode must be pending or model.")
        model_dir = str(payload.get("model_dir", "")).strip() if mode == "model" else ""
        if mode == "model" and not model_dir:
            raise ValueError("Choose a saved model directory, or select pending mode.")
        args = build_connection_args(
            transport=transport,
            device=device,
            participant_id=participant_id,
            model_dir=model_dir,
            calibration_sbp=str(payload.get("calibration_sbp", "")),
            calibration_dbp=str(payload.get("calibration_dbp", "")),
            fast_window=bool(payload.get("fast_window", False)),
            allow_unvalidated=bool(payload.get("allow_unvalidated", False)),
        )
        context = load_viewer_context(args)
        if context.model_error:
            raise ValueError(f"Model incompatible: {context.model_error}")
        with self._changed:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A sensor session is already active. Stop it before reconnecting.")
            self._history.clear()
            self._history_keys.clear()
            self._snapshot = None
            self._error = ""
            self._connection_state = "connecting"
            self._session = {
                "transport": transport,
                "device": device or "BLE auto-discovery",
                "participant_id": participant_id,
                "mode": mode,
                "model_dir": model_dir,
                "calibration_sbp": context.calibration_sbp,
                "calibration_dbp": context.calibration_dbp,
                "fast_window": context.experimental_fast_window_seconds is not None,
                "allow_unvalidated": context.allow_unvalidated,
            }
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._worker_function,
                args=(args, context, self._stop, self._publish),
                daemon=True,
                name="ppg-bp-dashboard",
            )
            self._thread.start()
            self._notify()
            return dict(self._session)

    def disconnect(self, timeout: float = 3.0) -> None:
        with self._changed:
            stop = self._stop
            thread = self._thread
            if stop is not None:
                stop.set()
            if thread is None:
                self._connection_state = "idle"
                self._notify()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._changed:
            if thread is not None and thread.is_alive():
                self._error = "Sensor worker did not stop within the expected time."
                self._connection_state = "error"
            elif self._connection_state != "error":
                self._connection_state = "idle"
            self._notify()

    def reset_history(self) -> None:
        with self._changed:
            self._history.clear()
            self._history_keys.clear()
            self._notify()

    def state(self) -> dict[str, Any]:
        with self._lock:
            snapshot = asdict(self._snapshot) if self._snapshot is not None else None
            return {
                "revision": self._revision,
                "connection_state": self._connection_state,
                "error": self._error,
                "session": dict(self._session),
                "snapshot": snapshot,
                "history": list(self._history),
                "disclaimer": RESEARCH_DISCLAIMER,
            }

    def wait_for_change(self, after_revision: int, timeout: float = 15.0) -> dict[str, Any]:
        with self._changed:
            self._changed.wait_for(lambda: self._revision > after_revision, timeout=timeout)
            return self.state()

    def shutdown(self) -> None:
        self.disconnect()


def create_app(controller: DashboardSessionController | None = None) -> Flask:
    controller = controller or DashboardSessionController()
    app = Flask(__name__, static_folder=None)
    app.config["dashboard_controller"] = controller

    @app.after_request
    def local_security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def failure(message: str, status: int = 400):
        return jsonify({"ok": False, "error": message}), status

    @app.get("/")
    def index():
        return send_from_directory(ASSET_ROOT, "index.html")

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        return send_from_directory(ASSET_ROOT, filename)

    @app.get("/api/devices")
    def devices():
        transport = request.args.get("transport", "serial")
        try:
            if transport == "serial":
                found = discover_serial_devices()
            elif transport == "ble":
                timeout = min(15.0, max(1.0, float(request.args.get("timeout", "4"))))
                found = discover_ble_devices(timeout)
            else:
                return failure("transport must be serial or ble")
        except (RuntimeError, ValueError, OSError) as exc:
            return failure(str(exc), 503)
        return jsonify({"ok": True, "devices": found})

    @app.get("/api/models")
    def models():
        try:
            found = discover_models(request.args.get("participant_id", "").strip())
        except ValueError as exc:
            return failure(str(exc))
        return jsonify({"ok": True, "models": found})

    @app.post("/api/connect")
    def connect():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return failure("Expected a JSON connection request.")
        try:
            session = controller.connect(payload)
        except ValueError as exc:
            return failure(str(exc))
        except RuntimeError as exc:
            return failure(str(exc), 409)
        return jsonify({"ok": True, "session": session})

    @app.post("/api/disconnect")
    def disconnect():
        controller.disconnect()
        return jsonify({"ok": True})

    @app.post("/api/history/reset")
    def reset_history():
        controller.reset_history()
        return jsonify({"ok": True})

    @app.get("/api/state")
    def state():
        return jsonify(controller.state())

    @app.get("/api/events")
    def events():
        try:
            initial_revision = int(request.args.get("after", "-1"))
        except ValueError:
            initial_revision = -1

        @stream_with_context
        def generate():
            revision = initial_revision
            while True:
                current = controller.wait_for_change(revision)
                revision = int(current["revision"])
                yield f"id: {revision}\nevent: state\ndata: {json.dumps(current, allow_nan=False)}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def parse_dashboard_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local experimental BP dashboard.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_dashboard_args(argv)
    controller = DashboardSessionController()
    app = create_app(controller)
    url = f"http://{DEFAULT_HOST}:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Experimental BP dashboard: {url}")
    print("Press Ctrl+C to stop. The dashboard does not save study data.")
    try:
        app.run(host=DEFAULT_HOST, port=args.port, threaded=True, use_reloader=False)
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
