import unittest
from argparse import Namespace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from view_live_upper_arm_hr import (
    MINIMUM_ANALYSIS_SECONDS,
    UpperArmViewerState,
    buffer_duration_s,
    build_validation_record,
    display_hr,
    maybe_analyze,
    parse_args,
    render_screen,
    run_viewer,
    update_state_from_line,
)


def add_still_samples(state: UpperArmViewerState, duration_s: float = 40.0) -> None:
    update_state_from_line(
        state,
        "# motion timestamp_ms=0 status=still activity_g=0.018 threshold_g=0.050",
        now=0.0,
    )
    sample_count = int(duration_s * 100) + 1
    for sequence in range(sample_count):
        timestamp_ms = sequence * 10
        update_state_from_line(
            state,
            f"{sequence},{timestamp_ms},85000,140000",
            now=timestamp_ms / 1000.0,
        )
    update_state_from_line(
        state,
        f"# motion timestamp_ms={int(duration_s * 1000)} status=still activity_g=0.018 threshold_g=0.050",
        now=duration_s,
    )


def fake_result(
    *,
    bpm=72.4,
    status="usable",
    latest_window_end=35.0,
    status_reason="accepted",
):
    windows = [
        SimpleNamespace(status="accepted", end_s=25.0),
        SimpleNamespace(status="accepted", end_s=30.0),
        SimpleNamespace(status="accepted", end_s=latest_window_end),
    ]
    return SimpleNamespace(
        bpm=bpm,
        status=status,
        status_reason=status_reason,
        windows=windows,
        time_s=list(range(41)),
        detected_peak_count=24,
        accepted_window_count=3,
        clean_coverage_s=30.0,
    )


class UpperArmViewerParsingTests(unittest.TestCase):
    def test_buffers_ppg_and_ignores_firmware_finger_hr(self):
        state = UpperArmViewerState(started_at=0.0)

        self.assertTrue(update_state_from_line(state, "1,10,85000,140000", now=1.0))
        self.assertTrue(
            update_state_from_line(
                state,
                "# hr timestamp_ms=10000 bpm=150.0 status=stable beats=6",
                now=1.1,
            )
        )

        self.assertEqual(len(state.ppg_samples), 1)
        self.assertIsNone(state.preview.bpm)
        self.assertNotIn("150.0", render_screen(state, 1.2, "COM5", 115200))

    def test_raw_imu_and_malformed_rows_never_appear(self):
        state = UpperArmViewerState(started_at=0.0)

        self.assertTrue(update_state_from_line(state, "imu,1,10,-2,0,257", now=1.0))
        self.assertFalse(update_state_from_line(state, "bad,row", now=1.1))
        screen = render_screen(state, 1.2, "COM5", 115200)

        self.assertNotIn("-2,0,257", screen)
        self.assertNotIn("bad,row", screen)

    def test_cli_supports_current_com_port(self):
        args = parse_args(["--port", "COM5", "--baud", "115200", "--refresh", "0.5"])
        self.assertEqual(args.port, "COM5")
        self.assertEqual(args.baud, 115200)
        self.assertEqual(args.refresh, 0.5)


class UpperArmViewerAnalysisTests(unittest.TestCase):
    def test_waits_for_forty_seconds_of_still_data(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state, duration_s=MINIMUM_ANALYSIS_SECONDS - 1.0)
        analyzer = Mock()

        self.assertFalse(maybe_analyze(state, now=39.0, analyzer=analyzer))
        analyzer.assert_not_called()
        self.assertEqual(state.preview.status, "warming_up")
        self.assertIn("39.0/40 s", state.preview.reason)

    def test_reports_stable_only_after_accepted_recent_consensus(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state)

        self.assertTrue(maybe_analyze(state, now=40.0, analyzer=lambda _df, _metadata: fake_result()))

        self.assertEqual(display_hr(state, now=40.1), ("72.4", "Stable"))
        self.assertEqual(state.preview.accepted_windows, 3)
        self.assertEqual(state.preview.clean_coverage_s, 30.0)

    def test_builds_machine_readable_stable_validation_snapshot(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state)
        update_state_from_line(state, "# stats rate_hz=99.8 ovf=0 i2c_errors=0", now=40.0)
        maybe_analyze(state, now=40.0, analyzer=lambda _df, _metadata: fake_result())

        record = build_validation_record(state, now=40.2, elapsed_s=40.2)

        self.assertEqual(record["status"], "stable")
        self.assertEqual(record["bpm"], 72.4)
        self.assertEqual(record["analysis_timestamp_ms"], 40000)
        self.assertEqual(record["ppg_rate_hz"], 99.8)
        self.assertEqual(record["ppg_i2c_errors"], 0)

    def test_validation_screen_reports_that_data_is_saved(self):
        state = UpperArmViewerState(started_at=0.0)
        screen = render_screen(state, now=0.0, port="COM5", baud=115200, saving=True)

        self.assertIn("are saved", screen)
        self.assertNotIn("No data is being saved", screen)

    def test_rejects_numeric_consensus_when_recent_window_failed(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state)

        maybe_analyze(
            state,
            now=40.0,
            analyzer=lambda _df, _metadata: fake_result(latest_window_end=30.0),
        )

        self.assertEqual(state.preview.status, "recent_window_rejected")
        self.assertEqual(display_hr(state, now=40.1)[0], "--")

    def test_motion_immediately_hides_bpm_and_clears_buffer(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state)
        maybe_analyze(state, now=40.0, analyzer=lambda _df, _metadata: fake_result())
        self.assertEqual(display_hr(state, now=40.1)[0], "72.4")

        update_state_from_line(
            state,
            "# motion timestamp_ms=41000 status=moving activity_g=0.100 threshold_g=0.050",
            now=41.0,
        )

        self.assertEqual(len(state.ppg_samples), 0)
        self.assertEqual(display_hr(state, now=41.1), ("--", "Motion detected"))

    def test_stale_motion_status_hides_previous_bpm(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state)
        maybe_analyze(state, now=40.0, analyzer=lambda _df, _metadata: fake_result())

        self.assertEqual(display_hr(state, now=43.1), ("--", "Motion update stale"))

    def test_analysis_receives_health_counters_and_motion_updates(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state)
        update_state_from_line(state, "# stats rate_hz=100.0 ovf=2 i2c_errors=1", now=40.0)
        update_state_from_line(
            state,
            "# imu_stats rate_hz=100.0 fifo_overflows=3 i2c_errors=4",
            now=40.0,
        )
        captured = {}

        def analyzer(_df, metadata):
            captured.update(metadata)
            return fake_result(bpm=None, status="invalid_timing", status_reason="sensor errors")

        maybe_analyze(state, now=40.0, analyzer=analyzer)

        self.assertEqual(captured["firmware_fifo_overflow_count"], 2)
        self.assertEqual(captured["firmware_i2c_error_count"], 1)
        self.assertEqual(captured["imu_firmware_fifo_overflow_count"], 3)
        self.assertEqual(captured["imu_firmware_i2c_error_count"], 4)
        self.assertEqual(state.preview.status, "invalid_timing")

    def test_rolling_buffer_is_bounded(self):
        state = UpperArmViewerState(started_at=0.0)
        add_still_samples(state, duration_s=65.0)

        self.assertLessEqual(buffer_duration_s(state), 60.0)
        self.assertGreater(buffer_duration_s(state), 59.9)


class FakeSerialException(Exception):
    pass


class InterruptingPort:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def reset_input_buffer(self):
        pass

    def readline(self):
        raise KeyboardInterrupt


class FakeSerialModule:
    SerialException = FakeSerialException

    @staticmethod
    def Serial(port, baud, timeout):
        return InterruptingPort()


class UpperArmViewerRuntimeTests(unittest.TestCase):
    def test_ctrl_c_exits_cleanly_without_saving(self):
        args = Namespace(port="COM5", baud=115200, refresh=1.0)
        output = StringIO()

        with patch("sys.stdout", output):
            result = run_viewer(args, FakeSerialModule, clock=lambda: 0.0, sleep=lambda _seconds: None)

        self.assertEqual(result, 0)
        self.assertIn("No data was saved", output.getvalue())


if __name__ == "__main__":
    unittest.main()
