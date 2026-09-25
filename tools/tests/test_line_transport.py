import asyncio
import struct
import sys
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from line_transport import (
    BLE_TX_CHARACTERISTIC_UUID,
    BLE_FRAME_IMU_ABSOLUTE,
    BLE_FRAME_IMU_BATCH_BASE,
    BLE_FRAME_IMU_SELF_CONTAINED_BASE,
    BLE_FRAME_PPG_ABSOLUTE,
    BLE_FRAME_PPG_BATCH_BASE,
    BLE_FRAME_PPG_SELF_CONTAINED_BASE,
    BLE_FRAME_STATUS,
    BLE_FRAME_STATUS_FRAGMENT,
    BleLineSource,
    BleTelemetryDecoder,
    LineTransportDisconnected,
    LineTransportError,
    add_transport_arguments,
    validate_transport_arguments,
)


def pack_bits(opcode, fields):
    bits = []
    for value, width in fields:
        bits.extend((value >> bit) & 1 for bit in range(width))
    payload = bytearray([opcode])
    payload.extend(b"\x00" * ((len(bits) + 7) // 8))
    for index, value in enumerate(bits):
        if value:
            payload[1 + (index // 8)] |= 1 << (index % 8)
    return bytes(payload)


class FakeScanner:
    devices = []

    @classmethod
    async def discover(cls, timeout):
        del timeout
        return cls.devices


class FailingClient:
    def __init__(self, *_args, **_kwargs):
        self.is_connected = False

    async def connect(self):
        raise RuntimeError("connection refused")

    async def disconnect(self):
        self.is_connected = False


class DisconnectingClient:
    def __init__(self, *_args, **_kwargs):
        self.is_connected = False

    async def connect(self):
        self.is_connected = True

    async def start_notify(self, _uuid, _callback):
        self.is_connected = False

    async def disconnect(self):
        self.is_connected = False


class LineTransportTests(unittest.TestCase):
    def test_self_contained_ppg_batches_do_not_depend_on_previous_packet(self):
        decoder = BleTelemetryDecoder()
        first = pack_bits(
            BLE_FRAME_PPG_SELF_CONTAINED_BASE | 2,
            [
                (500, 32), (1000, 32),
                (50123, 18), (60456, 18),
                (50125, 18), (60459, 18), (2, 3),
            ],
        )
        later = pack_bits(
            BLE_FRAME_PPG_SELF_CONTAINED_BASE | 2,
            [
                (510, 32), (1100, 32),
                (50200, 18), (60500, 18),
                (50203, 18), (60504, 18), (1, 3),
            ],
        )

        self.assertEqual(
            decoder.decode(first),
            [b"500,1000,50123,60456\n", b"501,1010,50125,60459\n"],
        )
        self.assertEqual(
            decoder.decode(later),
            [b"510,1100,50200,60500\n", b"511,1109,50203,60504\n"],
        )

    def test_self_contained_imu_batch_preserves_signed_values(self):
        decoder = BleTelemetryDecoder()
        packet = pack_bits(
            BLE_FRAME_IMU_SELF_CONTAINED_BASE | 2,
            [
                (700, 32), (2000, 32),
                ((-1024) & 0x7FF, 11), (0, 11), (1023, 11),
                ((-100) & 0x7FF, 11), (200, 11), ((-300) & 0x7FF, 11), (2, 3),
            ],
        )

        self.assertEqual(
            decoder.decode(packet),
            [b"imu,700,2000,-1024,0,1023\n", b"imu,701,2010,-100,200,-300\n"],
        )

    def test_status_fragment_gap_discards_incomplete_line(self):
        decoder = BleTelemetryDecoder()

        self.assertEqual(
            decoder.decode(
                bytes([BLE_FRAME_STATUS_FRAGMENT]) + struct.pack("<H", 4) + b"# bad"
            ),
            [],
        )
        self.assertEqual(
            decoder.decode(
                bytes([BLE_FRAME_STATUS_FRAGMENT]) + struct.pack("<H", 6) + b"# good\n"
            ),
            [b"# good\n"],
        )

    def test_compact_ble_ppg_packets_reconstruct_exact_text_rows(self):
        decoder = BleTelemetryDecoder()
        absolute = bytes([BLE_FRAME_PPG_ABSOLUTE]) + struct.pack(
            "<IIII", 100, 2000, 50123, 60456
        )
        batch = pack_bits(
            BLE_FRAME_PPG_BATCH_BASE | 3,
            [
                (50125, 18), (60459, 18), (2, 3),
                (50130, 18), (60463, 18), (1, 3),
                (50128, 18), (60461, 18), (3, 3),
            ],
        )

        self.assertEqual(decoder.decode(absolute), [b"100,2000,50123,60456\n"])
        self.assertEqual(
            decoder.decode(batch),
            [
                b"101,2010,50125,60459\n",
                b"102,2019,50130,60463\n",
                b"103,2030,50128,60461\n",
            ],
        )

    def test_compact_ble_imu_packets_preserve_signed_values_and_timestamps(self):
        decoder = BleTelemetryDecoder()
        absolute = bytes([BLE_FRAME_IMU_ABSOLUTE]) + struct.pack(
            "<IIhhh", 300, 4000, -1024, 0, 1023
        )
        batch = pack_bits(
            BLE_FRAME_IMU_BATCH_BASE | 2,
            [
                ((-100) & 0x7FF, 11), (200, 11), ((-300) & 0x7FF, 11), (2, 3),
                ((-101) & 0x7FF, 11), (201, 11), ((-301) & 0x7FF, 11), (3, 3),
            ],
        )

        self.assertEqual(decoder.decode(absolute), [b"imu,300,4000,-1024,0,1023\n"])
        self.assertEqual(
            decoder.decode(batch),
            [b"imu,301,4010,-100,200,-300\n", b"imu,302,4021,-101,201,-301\n"],
        )

    def test_compact_status_frames_still_reassemble_newline_records(self):
        source = BleLineSource(None, 1.0, 0.01, reconnect=False)
        source._notification(None, bytearray([BLE_FRAME_STATUS]) + bytearray(b"# hr status="))
        source._notification(None, bytearray([BLE_FRAME_STATUS]) + bytearray(b"stable\n"))

        self.assertEqual(source.readline(), b"# hr status=stable\n")

    def test_malformed_compact_packet_fails_safely(self):
        source = BleLineSource(None, 1.0, 0.01, reconnect=False)
        source._notification(None, bytearray([BLE_FRAME_PPG_BATCH_BASE | 1, 0]))

        with self.assertRaisesRegex(LineTransportError, "before an absolute PPG"):
            source.readline()

    def test_ble_notifications_reassemble_fragmented_and_combined_lines(self):
        source = BleLineSource(None, 1.0, 0.01, reconnect=False)
        source._notification(None, bytearray(b"1,10,20"))
        source._notification(None, bytearray(b",30\nimu,2,20,1,2,3\npartial"))

        self.assertEqual(source.readline(), b"1,10,20,30\n")
        self.assertEqual(source.readline(), b"imu,2,20,1,2,3\n")
        self.assertEqual(source.readline(), b"")

    def test_discovery_without_selector_requires_exactly_one_logger(self):
        FakeScanner.devices = [
            SimpleNamespace(name="PPG-LOGGER-AABBCC", address="AA:BB:CC"),
            SimpleNamespace(name="headphones", address="11:22:33"),
        ]
        source = BleLineSource(
            None, 1.0, 0.01, reconnect=False, scanner_class=FakeScanner, client_class=object
        )
        device = asyncio.run(source._discover_device())
        self.assertEqual(device.name, "PPG-LOGGER-AABBCC")

        FakeScanner.devices.append(
            SimpleNamespace(name="PPG-LOGGER-DDEEFF", address="DD:EE:FF")
        )
        with self.assertRaisesRegex(LineTransportError, "found 2"):
            asyncio.run(source._discover_device())

    def test_discovery_accepts_exact_name_or_address(self):
        FakeScanner.devices = [
            SimpleNamespace(name="PPG-LOGGER-AABBCC", address="AA:BB:CC")
        ]
        by_name = BleLineSource(
            "PPG-LOGGER-AABBCC", 1.0, 0.01, reconnect=False,
            scanner_class=FakeScanner, client_class=object,
        )
        by_address = BleLineSource(
            "aa:bb:cc", 1.0, 0.01, reconnect=False,
            scanner_class=FakeScanner, client_class=object,
        )
        self.assertEqual(asyncio.run(by_name._discover_device()).address, "AA:BB:CC")
        self.assertEqual(asyncio.run(by_address._discover_device()).name, "PPG-LOGGER-AABBCC")

    def test_initial_connection_failure_is_reported(self):
        FakeScanner.devices = [
            SimpleNamespace(name="PPG-LOGGER-AABBCC", address="AA:BB:CC")
        ]
        source = BleLineSource(
            None,
            0.1,
            0.01,
            reconnect=False,
            scanner_class=FakeScanner,
            client_class=FailingClient,
        )

        with self.assertRaisesRegex(LineTransportError, "connection refused"):
            source.__enter__()

    def test_mock_client_disconnect_stops_non_reconnecting_source(self):
        FakeScanner.devices = [
            SimpleNamespace(name="PPG-LOGGER-AABBCC", address="AA:BB:CC")
        ]
        source = BleLineSource(
            None,
            0.1,
            0.05,
            reconnect=False,
            scanner_class=FakeScanner,
            client_class=DisconnectingClient,
        )

        with source:
            with self.assertRaises(LineTransportDisconnected):
                source.readline()

    def test_non_reconnecting_disconnect_raises(self):
        source = BleLineSource(None, 1.0, 0.01, reconnect=False)
        source._chunks.put(source._DISCONNECTED)
        with self.assertRaises(LineTransportDisconnected):
            source.readline()

    def test_reconnect_discards_partial_line_from_old_connection(self):
        source = BleLineSource(None, 1.0, 0.01, reconnect=True)
        source._notification(None, bytearray(b"old,partial"))
        source._chunks.put(source._STREAM_RESET)
        source._notification(None, bytearray(b"10,100,200,300\n"))

        self.assertEqual(source.readline(), b"10,100,200,300\n")

    def test_transport_cli_preserves_serial_default_and_validates_ble(self):
        parser = ArgumentParser()
        add_transport_arguments(parser)
        serial_args = parser.parse_args(["--port", "COM9"])
        self.assertEqual(validate_transport_arguments(serial_args).transport, "serial")

        ble_args = parser.parse_args(["--transport", "ble", "--ble-device", "PPG-LOGGER-AABBCC"])
        self.assertEqual(validate_transport_arguments(ble_args).transport, "ble")
        self.assertIsNone(ble_args.port)

        with self.assertRaises(SystemExit), patch("sys.stderr"):
            validate_transport_arguments(parser.parse_args([]), parser)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            validate_transport_arguments(
                parser.parse_args(["--transport", "ble", "--port", "COM9"]), parser
            )

    def test_characteristic_uuid_is_fixed(self):
        self.assertEqual(
            BLE_TX_CHARACTERISTIC_UUID,
            "9f5c0002-6f5a-4b7d-9a21-3d4e5f607182",
        )


if __name__ == "__main__":
    unittest.main()
