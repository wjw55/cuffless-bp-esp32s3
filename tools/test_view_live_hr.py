import unittest
from argparse import Namespace
from io import StringIO
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from view_live_hr import (
    ViewerState,
    connection_status,
    hr_display,
    motion_display,
    parse_args,
    render_screen,
    run_viewer,
    update_state_from_line,
)


class ViewerParsingTests(unittest.TestCase):
    def test_parses_stable_and_non_numeric_hr_updates(self):
        state = ViewerState(started_at=0.0)

        update_state_from_line(
            state,
            "# hr timestamp_ms=10420 bpm=na status=warming_up beats=2",
            now=1.0,
        )
        self.assertEqual(hr_display(state, now=1.1), ("--", "Warming up", "2"))

        update_state_from_line(
            state,
            "# hr timestamp_ms=20420 bpm=72.4 status=stable beats=6",
            now=2.0,
        )
        self.assertEqual(hr_display(state, now=2.1), ("72.4", "Stable", "6"))

    def test_parses_ppg_imu_health_and_warning(self):
        state = ViewerState(started_at=0.0)

        update_state_from_line(
            state,
            "# stats rate_hz=99.8 ovf=0 i2c_errors=0",
            now=1.0,
        )
        update_state_from_line(
            state,
            "# imu_stats rate_hz=100.1 fifo_overflows=0 i2c_errors=1",
            now=1.1,
        )
        update_state_from_line(
            state,
            "# warning event=imu_fifo_read_failed i2c_errors=1",
            now=1.2,
        )
        screen = render_screen(state, now=1.3, port="COM3", baud=115200)

        self.assertIn("99.8 Hz", screen)
        self.assertIn("100.1 Hz", screen)
        self.assertIn("imu_fifo_read_failed", screen)

    def test_parses_and_displays_motion_status(self):
        state = ViewerState(started_at=0.0)
        self.assertTrue(
            update_state_from_line(
                state,
                "# motion timestamp_ms=20420 status=still activity_g=0.018 threshold_g=0.050",
                now=1.0,
            )
        )

        self.assertEqual(motion_display(state, now=1.1), ("Still", "0.018 g"))
        screen = render_screen(state, now=1.1, port="COM3", baud=115200)
        self.assertIn("Motion: Still", screen)
        self.assertNotIn("# motion", screen)

    def test_motion_unavailable_and_stale_states(self):
        state = ViewerState(started_at=0.0)
        update_state_from_line(
            state,
            "# motion timestamp_ms=1000 status=imu_unavailable activity_g=na threshold_g=na",
            now=1.0,
        )
        self.assertEqual(motion_display(state, now=1.1), ("IMU unavailable", "--"))
        self.assertEqual(motion_display(state, now=4.1), ("Motion update stale", "--"))

    def test_raw_rows_and_malformed_lines_never_appear(self):
        state = ViewerState(started_at=0.0)

        self.assertFalse(update_state_from_line(state, "42,12345,50001,60002", now=1.0))
        self.assertFalse(update_state_from_line(state, "imu,42,12345,-2,0,257", now=1.1))
        self.assertFalse(update_state_from_line(state, "# nonsense without fields", now=1.2))
        screen = render_screen(state, now=1.3, port="COM3", baud=115200)

        self.assertNotIn("50001", screen)
        self.assertNotIn("60002", screen)
        self.assertNotIn("-2,0,257", screen)
        self.assertNotIn("nonsense", screen)


class ViewerStateTests(unittest.TestCase):
    def test_hr_becomes_stale_after_three_seconds(self):
        state = ViewerState(started_at=0.0)
        update_state_from_line(
            state,
            "# hr timestamp_ms=20420 bpm=72.4 status=stable beats=6",
            now=2.0,
        )

        self.assertEqual(hr_display(state, now=5.1), ("--", "Heart-rate update stale", "6"))

    def test_connection_reports_waiting_receiving_and_stale(self):
        state = ViewerState(started_at=0.0)
        self.assertEqual(connection_status(state, now=2.0), "Waiting for ESP32")
        self.assertEqual(connection_status(state, now=6.0), "No serial data")

        update_state_from_line(state, "1,10,20,30", now=7.0)
        self.assertEqual(connection_status(state, now=7.1), "Receiving")
        self.assertIn("Stale", connection_status(state, now=13.0))

    def test_port_baud_and_refresh_are_configurable(self):
        args = parse_args(["--port", "COM9", "--baud", "230400", "--refresh", "0.5"])
        self.assertEqual(args.port, "COM9")
        self.assertEqual(args.baud, 230400)
        self.assertEqual(args.refresh, 0.5)


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


class ViewerRuntimeTests(unittest.TestCase):
    def test_ctrl_c_exits_cleanly_without_saving(self):
        args = Namespace(port="COM3", baud=115200, refresh=1.0)
        output = StringIO()

        with patch("sys.stdout", output):
            result = run_viewer(args, FakeSerialModule, clock=lambda: 0.0, sleep=lambda _seconds: None)

        self.assertEqual(result, 0)
        self.assertIn("No data was saved", output.getvalue())


if __name__ == "__main__":
    unittest.main()
