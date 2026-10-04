"""Regression checks for the display-only localhost BP dashboard."""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_bp_presentation import LiveBPPresentationEngine, PresentationSnapshot
from view_live_bp import ViewerContext
from view_live_bp_dashboard import (
    DashboardSessionController,
    create_app,
    discover_models,
)


def snapshot(**changes) -> PresentationSnapshot:
    base = PresentationSnapshot(
        mode="unavailable",
        sbp=None,
        dbp=None,
        age_s=None,
        sensor_timestamp_ms=None,
        status="Collecting clean PPG",
        current_status="Collecting clean PPG",
        reason="warming",
        current_reason="warming",
        source_status=None,
        buffer_s=10.0,
        target_s=85.0,
        accepted_windows=0,
        total_windows=0,
        clean_coverage_s=0.0,
        pulse_rate_bpm=None,
        motion="Still",
        intensity="Low",
        intensity_g=0.01,
        connection="Receiving",
        transport_connected=True,
        device="COM9",
        model_eligibility="Model pending",
        experimental_warning="",
        ppg_rate_hz=100.0,
        imu_rate_hz=100.0,
        ppg_i2c_errors=0,
        ppg_fifo_overflows=0,
        imu_i2c_errors=0,
        imu_fifo_overflows=0,
        ble_connected=None,
        ble_subscribed=None,
        ble_mtu=None,
        ble_dropped_records=None,
        ble_notify_errors=None,
        warnings=(),
        waveform_x_s=(-1.0, 0.0),
        waveform_ir=(-3.0, 3.0),
    )
    return replace(base, **changes)


class DashboardDiscoveryTests(unittest.TestCase):
    def test_model_discovery_filters_participant_and_reports_gate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for participant, eligible in (("P001", True), ("P002", False)):
                directory = root / "data" / "processed" / "bp" / "run1" / "single_subject" / participant
                directory.mkdir(parents=True)
                (directory / "model_manifest.json").write_text(
                    json.dumps(
                        {
                            "participant_id": participant,
                            "viewer_eligible": eligible,
                            "population_validated": False,
                            "calibration": {"label_group_id": "cal1", "sbp": 116, "dbp": 72},
                        }
                    ),
                    encoding="utf-8",
                )
            models = discover_models("P001", root)
            self.assertEqual(len(models), 1)
            self.assertTrue(models[0]["viewer_eligible"])
            self.assertEqual(models[0]["calibration_sbp"], 116)

    def test_invalid_participant_is_not_used_in_discovery(self):
        with self.assertRaisesRegex(ValueError, "letters"):
            discover_models("../P001")


class DashboardControllerTests(unittest.TestCase):
    def test_ble_health_uses_the_shared_firmware_parser(self):
        engine = LiveBPPresentationEngine(ViewerContext("P001", 116, 72, {}), 0.0)
        engine.ingest(
            "# ble_stats connected=true subscribed=true mtu=247 dropped_records=0 notify_errors=0",
            1.0,
        )
        result = engine.snapshot(1.0, "BLE", transport="ble", baud=115200)
        self.assertEqual(result.ble_connected, "true")
        self.assertEqual(result.ble_mtu, 247)
        self.assertEqual(result.ble_dropped_records, 0)

    def test_history_contains_only_unique_current_estimates(self):
        controller = DashboardSessionController()
        current = snapshot(
            mode="current", sbp=121.0, dbp=77.0, sensor_timestamp_ms=85000,
            source_status="prediction_ready",
        )
        controller._publish("snapshot", current)
        controller._publish("snapshot", current)
        controller._publish("snapshot", replace(current, mode="held", age_s=4.0))
        state = controller.state()
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["history"][0]["sbp"], 121.0)
        controller.reset_history()
        self.assertEqual(controller.state()["history"], [])

    def test_disconnected_snapshot_reports_reconnecting(self):
        controller = DashboardSessionController()
        controller._publish("snapshot", snapshot(transport_connected=False))
        self.assertEqual(controller.state()["connection_state"], "reconnecting")

    def test_history_is_bounded(self):
        controller = DashboardSessionController()
        for index in range(505):
            controller._publish(
                "snapshot",
                snapshot(
                    mode="current", sbp=120.0, dbp=75.0,
                    sensor_timestamp_ms=index, source_status="prediction_ready",
                ),
            )
        state = controller.state()
        self.assertEqual(len(state["history"]), 500)
        self.assertEqual(state["history"][0]["sensor_timestamp_ms"], 5)

    def test_connect_owns_one_worker_and_disconnect_stops_it(self):
        entered = threading.Event()

        def worker(_args, _context, stop, publish):
            entered.set()
            publish("snapshot", snapshot())
            stop.wait(2.0)
            publish("stopped", None)

        controller = DashboardSessionController(worker=worker)
        context = ViewerContext("P001", 116, 72, {})
        payload = {
            "transport": "serial", "device": "COM9", "participant_id": "P001",
            "mode": "pending", "calibration_sbp": "116", "calibration_dbp": "72",
        }
        with patch("view_live_bp_dashboard.load_viewer_context", return_value=context):
            controller.connect(payload)
            self.assertTrue(entered.wait(1.0))
            with self.assertRaisesRegex(RuntimeError, "already active"):
                controller.connect(payload)
        controller.disconnect()
        self.assertEqual(controller.state()["connection_state"], "idle")

    def test_new_connection_clears_history(self):
        def worker(_args, _context, stop, publish):
            stop.wait(1.0)
            publish("stopped", None)

        controller = DashboardSessionController(worker=worker)
        controller._publish(
            "snapshot",
            snapshot(mode="current", sbp=120, dbp=75, sensor_timestamp_ms=1),
        )
        context = ViewerContext("P001", 116, 72, {})
        with patch("view_live_bp_dashboard.load_viewer_context", return_value=context):
            controller.connect(
                {
                    "transport": "serial", "device": "COM9", "participant_id": "P001",
                    "mode": "pending", "calibration_sbp": "116", "calibration_dbp": "72",
                }
            )
        self.assertEqual(controller.state()["history"], [])
        controller.disconnect()

    def test_incompatible_model_does_not_start_worker(self):
        called = []
        controller = DashboardSessionController(worker=lambda *_args: called.append(True))
        context = ViewerContext("P001", 0, 0, {}, model_error="wrong participant")
        with patch("view_live_bp_dashboard.load_viewer_context", return_value=context):
            with self.assertRaisesRegex(ValueError, "Model incompatible"):
                controller.connect(
                    {
                        "transport": "serial", "device": "COM9", "participant_id": "P001",
                        "mode": "model", "model_dir": "model",
                    }
                )
        self.assertEqual(called, [])


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        self.controller = DashboardSessionController()
        self.app = create_app(self.controller)
        self.client = self.app.test_client()

    def tearDown(self):
        self.controller.shutdown()

    def test_index_and_assets_are_bundled_locally(self):
        index = self.client.get("/")
        script = self.client.get("/assets/dashboard.js")
        self.assertEqual(index.status_code, 200)
        self.assertIn(b"Blood Pressure Dashboard", index.data)
        self.assertNotIn(b"https://", index.data)
        self.assertEqual(script.status_code, 200)
        index.close()
        script.close()

    def test_devices_and_models_endpoints(self):
        with patch(
            "view_live_bp_dashboard.discover_serial_devices",
            return_value=[{"name": "COM9", "address": "COM9", "description": "USB"}],
        ):
            response = self.client.get("/api/devices?transport=serial")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["devices"][0]["address"], "COM9")
        self.assertEqual(self.client.get("/api/devices?transport=wifi").status_code, 400)

    def test_connect_rejects_malformed_requests(self):
        self.assertEqual(self.client.post("/api/connect", data="not json").status_code, 400)
        response = self.client.post(
            "/api/connect",
            json={
                "transport": "serial", "device": "COM9", "participant_id": "patient name",
                "mode": "pending", "calibration_sbp": 116, "calibration_dbp": 72,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Participant ID", response.get_json()["error"])

    def test_state_and_history_reset(self):
        self.controller._publish(
            "snapshot",
            snapshot(mode="current", sbp=119, dbp=74, sensor_timestamp_ms=90000),
        )
        state = self.client.get("/api/state").get_json()
        self.assertEqual(state["snapshot"]["mode"], "current")
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(self.client.post("/api/history/reset").status_code, 200)
        self.assertEqual(self.client.get("/api/state").get_json()["history"], [])

    def test_sse_emits_complete_state(self):
        response = self.client.get("/api/events?after=-1", buffered=False)
        first = next(response.response).decode("utf-8")
        response.close()
        self.assertIn("event: state", first)
        self.assertIn('"connection_state": "idle"', first)


if __name__ == "__main__":
    unittest.main()
