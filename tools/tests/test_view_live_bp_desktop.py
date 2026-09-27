"""Regression checks for the display-only desktop BP viewer."""

from __future__ import annotations

import threading
import sys
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bp_core.inference import BPInferenceResult
from view_live_bp import ViewerContext, resolve_display_bp
from view_live_bp_desktop import (
    DesktopBPApp,
    DesktopViewerEngine,
    build_connection_args,
    run_desktop_worker,
)


def _context(*, eligible=True, override=False, fast=False):
    return ViewerContext(
        "P001",
        116,
        72,
        {},
        bundle=SimpleNamespace(viewer_eligible=eligible, manifest={}),
        allow_unvalidated=override,
        experimental_fast_window_seconds=30.0 if fast else None,
        experimental_fast_bundle=SimpleNamespace() if fast else None,
    )


def _feed_stationary(engine: DesktopViewerEngine, seconds: int) -> None:
    for index in range(seconds * 100 + 1):
        engine.ingest(f"{index},{index * 10},4200,55000", index / 100)
    engine.ingest(
        f"# motion timestamp_ms={seconds * 1000} status=still activity_g=0.010",
        float(seconds),
    )


class DesktopArgsTests(unittest.TestCase):
    def test_pending_mode_cannot_produce_numeric_bp(self):
        args = build_connection_args(
            transport="serial", device="COM9", participant_id="P001",
            calibration_sbp="116", calibration_dbp="72",
        )
        self.assertEqual(args.port, "COM9")
        self.assertIsNone(args.model_dir)
        context = ViewerContext("P001", 116, 72, {})
        engine = DesktopViewerEngine(context, 0.0)
        _feed_stationary(engine, 86)
        snapshot = engine.snapshot(86.0, "COM9", transport="serial", baud=115200)
        self.assertEqual(snapshot.mode, "unavailable")
        self.assertIsNone(snapshot.sbp)
        self.assertEqual(snapshot.status, "Model validation pending")

    def test_model_modes_are_explicit_and_serial_ble_share_arguments(self):
        args = build_connection_args(
            transport="ble", device="PPG-LOGGER-4652B6", participant_id="P001",
            model_dir="data/processed/bp/model", fast_window=True, allow_unvalidated=True,
        )
        self.assertEqual(args.transport, "ble")
        self.assertEqual(args.ble_device, "PPG-LOGGER-4652B6")
        self.assertEqual(args.experimental_fast_window, 30)
        self.assertTrue(args.allow_unvalidated)
        auto = build_connection_args(
            transport="ble", device="", participant_id="P001",
            calibration_sbp="116", calibration_dbp="72",
        )
        self.assertIsNone(auto.ble_device)
        with self.assertRaisesRegex(ValueError, "require a saved model"):
            build_connection_args(
                transport="serial", device="COM9", participant_id="P001",
                calibration_sbp="116", calibration_dbp="72", fast_window=True,
            )
        with self.assertRaisesRegex(ValueError, "positive SBP"):
            build_connection_args(
                transport="serial", device="COM9", participant_id="P001",
                calibration_sbp="70", calibration_dbp="90",
            )


class DesktopEngineTests(unittest.TestCase):
    def test_incompatible_model_never_produces_a_number(self):
        context = _context()
        context.model_error = "participant or configuration mismatch"
        engine = DesktopViewerEngine(context, 0.0)
        _feed_stationary(engine, 86)
        snapshot = engine.snapshot(86.0, "COM9", transport="serial", baud=115200)
        self.assertEqual(snapshot.mode, "unavailable")
        self.assertEqual(snapshot.model_eligibility, "Model incompatible")
        self.assertIsNone(snapshot.sbp)

    def test_current_estimate_matches_terminal_display_decision(self):
        expected = BPInferenceResult(
            "prediction_ready", "accepted waveform", sbp=121, dbp=78,
            accepted_windows=7, total_windows=10, clean_coverage_s=64,
        )
        engine = DesktopViewerEngine(_context(), 0.0, predictor=lambda *_: expected)
        _feed_stationary(engine, 86)
        engine.ingest("# stats samples=8601 rate_hz=100 ovf=0 i2c_errors=0", 86.0)
        engine.ingest("# imu_stats samples=8601 rate_hz=100 fifo_overflows=0 i2c_errors=0", 86.0)
        snapshot = engine.snapshot(86.0, "COM9", transport="serial", baud=115200)
        terminal = resolve_display_bp(engine.state, engine.context, 86.0)
        self.assertEqual((snapshot.mode, snapshot.sbp, snapshot.dbp),
                         (terminal.mode, terminal.result.sbp, terminal.result.dbp))
        self.assertEqual(snapshot.ppg_rate_hz, 100.0)
        self.assertEqual(snapshot.imu_rate_hz, 100.0)
        self.assertLessEqual(len(snapshot.waveform_x_s), 750)
        self.assertAlmostEqual(snapshot.waveform_x_s[-1], 0.0)
        self.assertTrue(all(value == 0.0 for value in snapshot.waveform_ir))

    def test_moving_holds_but_disconnect_and_reconnect_hide(self):
        result = BPInferenceResult("prediction_ready", "accepted", sbp=120, dbp=75)
        engine = DesktopViewerEngine(_context(), 0.0, predictor=lambda *_: result)
        _feed_stationary(engine, 86)
        engine.snapshot(86.0, "BLE", transport="ble", baud=115200)
        engine.ingest("# motion timestamp_ms=87000 status=moving activity_g=0.1", 87.0)
        held = engine.snapshot(87.0, "BLE", transport="ble", baud=115200)
        self.assertEqual(held.mode, "held")
        self.assertEqual(held.sbp, 120)
        self.assertEqual(held.current_status, "Motion detected")
        engine.set_connected(False)
        disconnected = engine.snapshot(87.1, "BLE", transport="ble", baud=115200)
        self.assertEqual(disconnected.mode, "unavailable")
        self.assertIsNone(disconnected.sbp)
        engine.set_connected(True)
        reconnected = engine.snapshot(87.2, "BLE", transport="ble", baud=115200)
        self.assertEqual(reconnected.mode, "unavailable")
        # A new connection must not reveal an old estimate, even if the latest
        # pre-disconnect motion update has not yet become stale.
        self.assertIsNone(reconnected.sbp)

    def test_health_fault_and_stale_motion_do_not_show_held_bp(self):
        result = BPInferenceResult("prediction_ready", "accepted", sbp=120, dbp=75)
        engine = DesktopViewerEngine(_context(), 0.0, predictor=lambda *_: result)
        _feed_stationary(engine, 86)
        engine.snapshot(86.0, "COM9", transport="serial", baud=115200)
        engine.ingest("# stats samples=8601 rate_hz=100 ovf=0 i2c_errors=0", 86.1)
        engine.ingest("# stats samples=8602 rate_hz=100 ovf=0 i2c_errors=1", 86.2)
        fault = engine.snapshot(86.2, "COM9", transport="serial", baud=115200)
        self.assertEqual(fault.mode, "unavailable")
        self.assertIsNone(fault.sbp)
        stale = engine.snapshot(90.0, "COM9", transport="serial", baud=115200)
        self.assertEqual(stale.mode, "unavailable")

    def test_unvalidated_gate_and_fast_policy_use_terminal_inference(self):
        def predictor(bundle, _frame, _metadata):
            if bundle is fast_context.experimental_fast_bundle:
                return BPInferenceResult("prediction_ready", "accepted", sbp=119, dbp=74)
            return BPInferenceResult("model_validation_failed", "did not beat baseline")

        fast_context = _context(fast=True)
        fast = DesktopViewerEngine(fast_context, 0.0, predictor=predictor)
        _feed_stationary(fast, 31)
        short = fast.snapshot(31.0, "COM9", transport="serial", baud=115200)
        self.assertEqual(short.status, "EXPERIMENTAL FAST ESTIMATE")
        self.assertEqual(short.sbp, 119)
        self.assertIn("30-SECOND", short.experimental_warning)

        rejected = DesktopViewerEngine(
            _context(eligible=False), 0.0,
            predictor=lambda *_: BPInferenceResult("model_validation_failed", "baseline gate failed"),
        )
        _feed_stationary(rejected, 86)
        hidden = rejected.snapshot(86.0, "COM9", transport="serial", baud=115200)
        self.assertEqual(hidden.mode, "unavailable")
        self.assertIsNone(hidden.sbp)

        override = DesktopViewerEngine(
            _context(eligible=False, override=True), 0.0,
            predictor=lambda *_: BPInferenceResult("unvalidated_estimate", "development only", sbp=118, dbp=73),
        )
        _feed_stationary(override, 86)
        shown = override.snapshot(86.0, "COM9", transport="serial", baud=115200)
        self.assertEqual(shown.mode, "current")
        self.assertIn("UNVALIDATED", shown.experimental_warning)

    def test_malformed_rows_do_not_crash(self):
        engine = DesktopViewerEngine(ViewerContext("P001", 116, 72, {}), 0.0)
        engine.ingest("nonsense\n", 0.1)
        engine.ingest("imu,not-a-sequence,broken\n", 0.2)
        engine.ingest("# warning event=bad_test\n", 0.3)
        snapshot = engine.snapshot(0.3, "COM9", transport="serial", baud=115200)
        self.assertIsNone(snapshot.sbp)
        self.assertIn("bad_test", " ".join(snapshot.warnings))


class FakeLineSource:
    def __init__(self, stop):
        self.stop = stop
        self.connected = True
        self.display_name = "fake-ble"
        self.closed = False
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def set_buffer_size(self, **_kwargs):
        pass

    def readline(self):
        self.reads += 1
        if self.reads == 1:
            return b"0,0,4200,55000\n"
        if self.reads == 2:
            return b"# motion timestamp_ms=10 status=still activity_g=0.01\n"
        self.stop.set()
        return b""


class DesktopWorkerTests(unittest.TestCase):
    def test_incompatible_model_is_rejected_before_opening_sensor(self):
        events = []
        source_factory = Mock()
        context = ViewerContext("P001", 0, 0, {}, model_error="wrong participant")
        with patch("view_live_bp_desktop.load_viewer_context", return_value=context):
            run_desktop_worker(
                SimpleNamespace(transport="ble", baud=115200), None,
                threading.Event(), lambda kind, payload: events.append((kind, payload)),
                source_factory=source_factory,
            )
        source_factory.assert_not_called()
        self.assertEqual(events[0], ("model_error", "wrong participant"))
        self.assertEqual(events[-1][0], "stopped")

    def test_window_constructs_and_switches_setup_modes(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Desktop display unavailable: {exc}")
        try:
            root.withdraw()
            app = DesktopBPApp(root)
            self.assertEqual(app.mode_var.get(), "pending")
            app.transport_var.set("ble")
            app._change_transport()
            app.mode_var.set("model")
            app._change_mode()
            root.update_idletasks()
            self.assertTrue(app.ble_box.winfo_ismapped() or not root.winfo_ismapped())
            app._show_unavailable("Model incompatible", "wrong participant")
            self.assertEqual(app.bp_label.cget("text"), "-- / --")
        finally:
            root.destroy()

    def test_worker_publishes_snapshot_and_closes_transport(self):
        stop = threading.Event()
        source = FakeLineSource(stop)
        messages = []
        ticks = iter([0.0, 0.1, 0.7, 1.3])
        args = SimpleNamespace(transport="ble", baud=115200)
        run_desktop_worker(
            args, ViewerContext("P001", 116, 72, {}), stop,
            lambda kind, payload: messages.append((kind, payload)),
            source_factory=lambda *_args, **_kwargs: source,
            clock=lambda: next(ticks),
        )
        self.assertTrue(source.closed)
        self.assertTrue(any(kind == "snapshot" for kind, _ in messages))
        self.assertEqual(messages[-1][0], "stopped")

    def test_stop_and_close_do_not_require_tk_event_loop(self):
        app = DesktopBPApp.__new__(DesktopBPApp)
        app.stop_event = threading.Event()
        app._show_unavailable = Mock()
        app.stop_button = Mock()
        app._stop()
        self.assertTrue(app.stop_event.is_set())
        app.worker = Mock()
        app.closing = False
        app._on_close()
        self.assertTrue(app.closing)
        self.assertTrue(app.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
