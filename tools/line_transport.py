"""Shared blocking line transports for USB serial and BLE telemetry."""

from __future__ import annotations

import asyncio
import queue
import struct
import threading
from dataclasses import dataclass
from typing import Any


BLE_DEVICE_PREFIX = "PPG-LOGGER-"
BLE_SERVICE_UUID = "9f5c0001-6f5a-4b7d-9a21-3d4e5f607182"
BLE_TX_CHARACTERISTIC_UUID = "9f5c0002-6f5a-4b7d-9a21-3d4e5f607182"
DEFAULT_BLE_SCAN_TIMEOUT_SECONDS = 10.0
DEFAULT_BLE_CONNECT_TIMEOUT_SECONDS = 15.0

BLE_FRAME_STATUS = 0x80
BLE_FRAME_PPG_ABSOLUTE = 0x81
BLE_FRAME_IMU_ABSOLUTE = 0x82
BLE_FRAME_PPG_BATCH_BASE = 0x90
BLE_FRAME_IMU_BATCH_BASE = 0xA0
BLE_FRAME_PPG_SELF_CONTAINED_BASE = 0xB0
BLE_FRAME_IMU_SELF_CONTAINED_BASE = 0xC0
BLE_FRAME_STATUS_FRAGMENT = 0xD0


class BleTelemetryDecoder:
    """Decode compact BLE packets back into the logger's existing text rows."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._ppg_sequence: int | None = None
        self._ppg_timestamp_ms: int | None = None
        self._imu_sequence: int | None = None
        self._imu_timestamp_ms: int | None = None
        self._status_sequence: int | None = None
        self._status_buffer = bytearray()

    @staticmethod
    def _get_bits(data: bytes, bit_offset: int, bit_count: int) -> tuple[int, int]:
        value = 0
        for bit in range(bit_count):
            source_bit = bit_offset + bit
            if data[source_bit // 8] & (1 << (source_bit % 8)):
                value |= 1 << bit
        return value, bit_offset + bit_count

    @staticmethod
    def _signed(value: int, bits: int) -> int:
        sign = 1 << (bits - 1)
        return value - (1 << bits) if value & sign else value

    @staticmethod
    def _require_length(data: bytes, expected: int, frame_name: str) -> None:
        if len(data) != expected:
            raise ValueError(
                f"invalid {frame_name} packet length: expected {expected}, got {len(data)}"
            )

    def decode(self, packet: bytes) -> list[bytes]:
        if not packet:
            return []
        frame = packet[0]
        if frame == BLE_FRAME_STATUS:
            return [packet[1:]] if len(packet) > 1 else []
        if frame == BLE_FRAME_STATUS_FRAGMENT:
            if len(packet) < 4:
                raise ValueError("invalid status fragment packet length")
            sequence = struct.unpack_from("<H", packet, 1)[0]
            if self._status_sequence is not None and sequence != self._status_sequence:
                self._status_buffer.clear()
            self._status_sequence = (sequence + 1) & 0xFFFF
            self._status_buffer.extend(packet[3:])
            lines = []
            while True:
                newline = self._status_buffer.find(b"\n")
                if newline < 0:
                    break
                lines.append(bytes(self._status_buffer[: newline + 1]))
                del self._status_buffer[: newline + 1]
            if len(self._status_buffer) > 4096:
                self._status_buffer.clear()
                raise ValueError("status fragment buffer exceeded safe limit")
            return lines
        if frame == BLE_FRAME_PPG_ABSOLUTE:
            self._require_length(packet, 17, "absolute PPG")
            sequence, timestamp_ms, red, ir = struct.unpack_from("<IIII", packet, 1)
            self._ppg_sequence = sequence
            self._ppg_timestamp_ms = timestamp_ms
            return [f"{sequence},{timestamp_ms},{red},{ir}\n".encode("ascii")]
        if frame == BLE_FRAME_IMU_ABSOLUTE:
            self._require_length(packet, 15, "absolute IMU")
            sequence, timestamp_ms = struct.unpack_from("<II", packet, 1)
            x, y, z = struct.unpack_from("<hhh", packet, 9)
            self._imu_sequence = sequence
            self._imu_timestamp_ms = timestamp_ms
            return [f"imu,{sequence},{timestamp_ms},{x},{y},{z}\n".encode("ascii")]

        family = frame & 0xF0
        count = frame & 0x0F
        if family == BLE_FRAME_PPG_SELF_CONTAINED_BASE:
            if not 1 <= count <= 15:
                raise ValueError(f"invalid self-contained PPG batch count: {count}")
            expected = (108 + (39 * (count - 1)) + 7) // 8
            self._require_length(packet, expected, "self-contained PPG batch")
            sequence, timestamp_ms = struct.unpack_from("<II", packet, 1)
            bit_offset = 72
            lines = []
            red, bit_offset = self._get_bits(packet, bit_offset, 18)
            ir, bit_offset = self._get_bits(packet, bit_offset, 18)
            lines.append(f"{sequence},{timestamp_ms},{red},{ir}\n".encode("ascii"))
            for _ in range(1, count):
                red, bit_offset = self._get_bits(packet, bit_offset, 18)
                ir, bit_offset = self._get_bits(packet, bit_offset, 18)
                delta_code, bit_offset = self._get_bits(packet, bit_offset, 3)
                sequence += 1
                timestamp_ms += 8 + delta_code
                lines.append(f"{sequence},{timestamp_ms},{red},{ir}\n".encode("ascii"))
            self._ppg_sequence = sequence
            self._ppg_timestamp_ms = timestamp_ms
            return lines
        if family == BLE_FRAME_IMU_SELF_CONTAINED_BASE:
            if not 1 <= count <= 15:
                raise ValueError(f"invalid self-contained IMU batch count: {count}")
            expected = (105 + (36 * (count - 1)) + 7) // 8
            self._require_length(packet, expected, "self-contained IMU batch")
            sequence, timestamp_ms = struct.unpack_from("<II", packet, 1)
            bit_offset = 72
            lines = []
            x, bit_offset = self._get_bits(packet, bit_offset, 11)
            y, bit_offset = self._get_bits(packet, bit_offset, 11)
            z, bit_offset = self._get_bits(packet, bit_offset, 11)
            lines.append(
                (
                    f"imu,{sequence},{timestamp_ms},{self._signed(x, 11)},"
                    f"{self._signed(y, 11)},{self._signed(z, 11)}\n"
                ).encode("ascii")
            )
            for _ in range(1, count):
                x, bit_offset = self._get_bits(packet, bit_offset, 11)
                y, bit_offset = self._get_bits(packet, bit_offset, 11)
                z, bit_offset = self._get_bits(packet, bit_offset, 11)
                delta_code, bit_offset = self._get_bits(packet, bit_offset, 3)
                sequence += 1
                timestamp_ms += 8 + delta_code
                lines.append(
                    (
                        f"imu,{sequence},{timestamp_ms},{self._signed(x, 11)},"
                        f"{self._signed(y, 11)},{self._signed(z, 11)}\n"
                    ).encode("ascii")
                )
            self._imu_sequence = sequence
            self._imu_timestamp_ms = timestamp_ms
            return lines
        if family == BLE_FRAME_PPG_BATCH_BASE:
            if not 1 <= count <= 15:
                raise ValueError(f"invalid PPG batch count: {count}")
            if self._ppg_sequence is None or self._ppg_timestamp_ms is None:
                raise ValueError("PPG batch arrived before an absolute PPG sample")
            expected = (8 + (39 * count) + 7) // 8
            self._require_length(packet, expected, "PPG batch")
            bit_offset = 8
            lines = []
            for _ in range(count):
                red, bit_offset = self._get_bits(packet, bit_offset, 18)
                ir, bit_offset = self._get_bits(packet, bit_offset, 18)
                delta_code, bit_offset = self._get_bits(packet, bit_offset, 3)
                self._ppg_sequence += 1
                self._ppg_timestamp_ms += 8 + delta_code
                lines.append(
                    f"{self._ppg_sequence},{self._ppg_timestamp_ms},{red},{ir}\n".encode(
                        "ascii"
                    )
                )
            return lines
        if family == BLE_FRAME_IMU_BATCH_BASE:
            if not 1 <= count <= 15:
                raise ValueError(f"invalid IMU batch count: {count}")
            if self._imu_sequence is None or self._imu_timestamp_ms is None:
                raise ValueError("IMU batch arrived before an absolute IMU sample")
            expected = (8 + (36 * count) + 7) // 8
            self._require_length(packet, expected, "IMU batch")
            bit_offset = 8
            lines = []
            for _ in range(count):
                x, bit_offset = self._get_bits(packet, bit_offset, 11)
                y, bit_offset = self._get_bits(packet, bit_offset, 11)
                z, bit_offset = self._get_bits(packet, bit_offset, 11)
                delta_code, bit_offset = self._get_bits(packet, bit_offset, 3)
                self._imu_sequence += 1
                self._imu_timestamp_ms += 8 + delta_code
                lines.append(
                    (
                        f"imu,{self._imu_sequence},{self._imu_timestamp_ms},"
                        f"{self._signed(x, 11)},{self._signed(y, 11)},"
                        f"{self._signed(z, 11)}\n"
                    ).encode("ascii")
                )
            return lines

        # Compatibility with the first BLE firmware, which sent unframed text.
        return [packet]


class LineTransportError(RuntimeError):
    """Raised when a selected line transport cannot continue."""


class LineTransportDisconnected(LineTransportError):
    """Raised when a non-reconnecting transport disconnects."""


@dataclass(frozen=True)
class TransportMetadata:
    transport: str
    device_name: str
    device_address: str | None = None
    service_uuid: str | None = None
    disconnect_count: int = 0


def add_transport_arguments(parser, *, default_baud: int = 115200) -> None:
    parser.add_argument(
        "--transport",
        choices=("serial", "ble"),
        default="serial",
        help="sensor transport; serial preserves the existing USB workflow",
    )
    parser.add_argument("--port", help="USB serial port, for example COM9")
    parser.add_argument(
        "--ble-device",
        help=(
            "exact BLE device name or address; when omitted, exactly one "
            f"{BLE_DEVICE_PREFIX}* device must be discoverable"
        ),
    )
    parser.add_argument(
        "--ble-scan-timeout",
        type=float,
        default=DEFAULT_BLE_SCAN_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"BLE discovery timeout (default: {DEFAULT_BLE_SCAN_TIMEOUT_SECONDS:g} seconds)",
    )
    parser.add_argument("--baud", type=int, default=default_baud)


def validate_transport_arguments(args, parser=None):
    transport = getattr(args, "transport", "serial")
    port = getattr(args, "port", None)
    ble_device = getattr(args, "ble_device", None)
    scan_timeout = getattr(args, "ble_scan_timeout", DEFAULT_BLE_SCAN_TIMEOUT_SECONDS)
    error = None
    if transport == "serial" and not port:
        error = "--port is required when --transport serial is selected"
    elif transport == "serial" and ble_device:
        error = "--ble-device requires --transport ble"
    elif transport == "ble" and port:
        error = "--port cannot be combined with --transport ble"
    elif not isinstance(scan_timeout, (int, float)) or scan_timeout <= 0:
        error = "--ble-scan-timeout must be greater than 0"
    if error:
        if parser is not None:
            parser.error(error)
        raise SystemExit(f"ERROR: {error}")
    return args


class SerialLineSource:
    def __init__(self, port: str, baud: int, timeout: float, serial_module=None):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial_module = serial_module
        self._serial = None

    def __enter__(self):
        if self._serial_module is None:
            try:
                import serial as serial_module  # type: ignore
            except ImportError as exc:
                raise LineTransportError(
                    "pyserial is required for USB serial. Install it with: python -m pip install pyserial"
                ) from exc
            self._serial_module = serial_module
        try:
            self._serial = self._serial_module.Serial(self.port, self.baud, timeout=self.timeout)
            if hasattr(self._serial, "__enter__"):
                self._serial.__enter__()
        except Exception as exc:
            raise LineTransportError(
                f"Could not open or read {self.port} at {self.baud} baud: {exc}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._serial is None:
            return False
        try:
            if hasattr(self._serial, "__exit__"):
                return bool(self._serial.__exit__(exc_type, exc, traceback))
            self._serial.close()
        except Exception:
            pass
        return False

    def readline(self) -> bytes:
        try:
            return self._serial.readline()
        except Exception as exc:
            raise LineTransportError(f"USB serial read failed: {exc}") from exc

    def reset_input_buffer(self) -> None:
        self._serial.reset_input_buffer()

    def set_buffer_size(self, **kwargs) -> None:
        self._serial.set_buffer_size(**kwargs)

    @property
    def display_name(self) -> str:
        return self.port

    @property
    def connected(self) -> bool:
        return self._serial is not None

    @property
    def metadata(self) -> TransportMetadata:
        return TransportMetadata("serial", self.port)


class BleLineSource:
    """Expose BLE notification bytes as a blocking newline-delimited reader."""

    _DISCONNECTED = object()
    _STREAM_RESET = object()

    def __init__(
        self,
        device_selector: str | None,
        scan_timeout: float,
        timeout: float,
        *,
        reconnect: bool,
        scanner_class=None,
        client_class=None,
    ):
        self.device_selector = device_selector
        self.scan_timeout = float(scan_timeout)
        self.timeout = float(timeout)
        self.reconnect = reconnect
        self._scanner_class = scanner_class
        self._client_class = client_class
        self._chunks: queue.Queue[Any] = queue.Queue()
        self._buffer = bytearray()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._initial_error: Exception | None = None
        self._device_name = device_selector or "BLE auto-discovery"
        self._device_address: str | None = None
        self._disconnect_count = 0
        self._ever_connected = False
        self._connected = False
        self._decoder = BleTelemetryDecoder()

    def _load_bleak(self) -> None:
        if self._scanner_class is not None and self._client_class is not None:
            return
        try:
            from bleak import BleakClient, BleakScanner  # type: ignore
        except ImportError as exc:
            raise LineTransportError(
                "BLE transport requires Bleak. Install it with: python -m pip install bleak"
            ) from exc
        self._scanner_class = BleakScanner
        self._client_class = BleakClient

    @staticmethod
    def _format_devices(devices) -> str:
        entries = []
        for device in devices:
            name = getattr(device, "name", None) or "<unnamed>"
            address = getattr(device, "address", "<unknown>")
            entries.append(f"{name} [{address}]")
        return ", ".join(entries) if entries else "none"

    async def _discover_device(self):
        devices = await self._scanner_class.discover(timeout=self.scan_timeout)
        if self.device_selector:
            selector = self.device_selector.casefold()
            matches = [
                device
                for device in devices
                if selector
                in {
                    str(getattr(device, "name", "")).casefold(),
                    str(getattr(device, "address", "")).casefold(),
                }
            ]
        else:
            matches = [
                device
                for device in devices
                if str(getattr(device, "name", "")).startswith(BLE_DEVICE_PREFIX)
            ]
        if len(matches) != 1:
            requirement = (
                f"exactly one match for {self.device_selector!r}"
                if self.device_selector
                else f"exactly one {BLE_DEVICE_PREFIX}* device"
            )
            raise LineTransportError(
                f"BLE discovery requires {requirement}; found {len(matches)}. "
                f"Discovered devices: {self._format_devices(devices)}"
            )
        return matches[0]

    def _notification(self, _sender, data: bytearray) -> None:
        if not data:
            return
        try:
            for decoded in self._decoder.decode(bytes(data)):
                if decoded:
                    self._chunks.put(decoded)
        except ValueError as exc:
            self._chunks.put(LineTransportError(f"BLE telemetry decode failed: {exc}"))

    async def _worker(self) -> None:
        first_attempt = True
        while not self._stop.is_set():
            client = None
            stage = "discovery"
            try:
                device = await self._discover_device()
                self._device_name = getattr(device, "name", None) or self._device_name
                self._device_address = getattr(device, "address", None)
                stage = "client creation"
                client = self._client_class(
                    device,
                    disconnected_callback=lambda _client: None,
                    timeout=DEFAULT_BLE_CONNECT_TIMEOUT_SECONDS,
                )
                stage = "connection"
                await client.connect()
                stage = "notification subscription"
                await client.start_notify(BLE_TX_CHARACTERISTIC_UUID, self._notification)
                self._ever_connected = True
                self._connected = True
                first_attempt = False
                self._ready.set()
                while not self._stop.is_set() and bool(client.is_connected):
                    await asyncio.sleep(0.1)
                if not self._stop.is_set():
                    self._connected = False
                    self._disconnect_count += 1
                    if not self.reconnect:
                        self._chunks.put(self._DISCONNECTED)
                        return
                    self._chunks.put(self._STREAM_RESET)
            except Exception as exc:
                if not isinstance(exc, LineTransportError):
                    exc = LineTransportError(f"BLE {stage} failed: {exc}")
                self._connected = False
                if first_attempt:
                    self._initial_error = exc
                    self._ready.set()
                    return
                self._disconnect_count += 1
                if not self.reconnect:
                    self._chunks.put(exc)
                    return
            finally:
                self._connected = False
                if client is not None:
                    try:
                        if bool(client.is_connected):
                            await client.disconnect()
                    except Exception:
                        pass
            if not self._stop.is_set():
                await asyncio.sleep(1.0)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._worker())
        except Exception as exc:
            self._initial_error = self._initial_error or exc
            self._ready.set()

    def __enter__(self):
        self._load_bleak()
        self._thread = threading.Thread(target=self._thread_main, name="ppg-ble", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.scan_timeout + DEFAULT_BLE_CONNECT_TIMEOUT_SECONDS + 2.0):
            self.close()
            raise LineTransportError("Timed out while starting the BLE transport")
        if self._initial_error is not None:
            self.close()
            if isinstance(self._initial_error, LineTransportError):
                raise self._initial_error
            raise LineTransportError(f"Could not connect over BLE: {self._initial_error}") from self._initial_error
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
        return False

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def readline(self) -> bytes:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                result = bytes(self._buffer[: newline + 1])
                del self._buffer[: newline + 1]
                return result
            try:
                item = self._chunks.get(timeout=self.timeout)
            except queue.Empty:
                return b""
            if item is self._DISCONNECTED:
                raise LineTransportDisconnected("BLE device disconnected during collection")
            if item is self._STREAM_RESET:
                # Never combine a partial line from the old connection with data
                # received after reconnection. Sequence gaps in subsequent complete
                # records cause viewers to restart their clean acquisition buffer.
                self._buffer.clear()
                self._decoder.reset()
                continue
            if isinstance(item, Exception):
                raise LineTransportError(f"BLE receive failed: {item}") from item
            self._buffer.extend(item)

    def reset_input_buffer(self) -> None:
        self._buffer.clear()
        self._decoder.reset()
        while True:
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                return

    def set_buffer_size(self, **_kwargs) -> None:
        return None

    @property
    def display_name(self) -> str:
        if self._device_address:
            return f"{self._device_name} [{self._device_address}]"
        return self._device_name

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def metadata(self) -> TransportMetadata:
        return TransportMetadata(
            "ble",
            self._device_name,
            self._device_address,
            BLE_SERVICE_UUID,
            self._disconnect_count,
        )


def create_line_source(
    args,
    *,
    timeout: float,
    reconnect: bool,
    serial_module=None,
    scanner_class=None,
    client_class=None,
):
    if getattr(args, "transport", "serial") == "ble":
        return BleLineSource(
            getattr(args, "ble_device", None),
            getattr(args, "ble_scan_timeout", DEFAULT_BLE_SCAN_TIMEOUT_SECONDS),
            timeout,
            reconnect=reconnect,
            scanner_class=scanner_class,
            client_class=client_class,
        )
    return SerialLineSource(args.port, args.baud, timeout, serial_module=serial_module)
