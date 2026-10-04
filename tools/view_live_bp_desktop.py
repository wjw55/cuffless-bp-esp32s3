"""Display the existing quality-gated BP preview in a local desktop window.

The worker owns all acquisition and inference state. Tkinter only receives copied
presentation snapshots, so no sensor or model work runs on the UI thread.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from line_transport import BLE_DEVICE_PREFIX, create_line_source
from live_bp_presentation import (
    LiveBPPresentationEngine as DesktopViewerEngine,
    PLOT_SECONDS,
    PresentationSnapshot as DesktopSnapshot,
    RESEARCH_DISCLAIMER,
    build_connection_args,
    run_presentation_worker,
)
from view_live_bp import (
    ViewerContext,
    load_viewer_context,
)


UI_REFRESH_SECONDS = 0.5
def run_desktop_worker(
    args: argparse.Namespace,
    context: ViewerContext | None,
    stop: threading.Event,
    publish: Callable[[str, object], None],
    *,
    source_factory=create_line_source,
    clock=time.monotonic,
) -> None:
    """Read and analyze away from Tkinter; publish snapshots or errors only."""
    run_presentation_worker(
        args,
        context,
        stop,
        publish,
        source_factory=source_factory,
        context_loader=load_viewer_context,
        clock=clock,
        refresh_seconds=UI_REFRESH_SECONDS,
    )


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = max(0, int(seconds))
    return f"{total} s ago" if total < 60 else f"{total // 60} min {total % 60:02d} s ago"


def _format_rate(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f} Hz"


def _format_count(value: int | None) -> str:
    return "--" if value is None else str(value)


class DesktopBPApp:
    def __init__(self, root: tk.Tk):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.root = root
        self.root.title("Experimental Upper-Arm BP Viewer")
        self.root.geometry("1100x850")
        self.root.minsize(850, 650)
        self.events: queue.Queue[tuple[int, str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.session_token = 0
        self.closing = False

        self.transport_var = tk.StringVar(value="serial")
        self.serial_port_var = tk.StringVar()
        self.ble_device_var = tk.StringVar()
        self.participant_var = tk.StringVar(value="P001")
        self.mode_var = tk.StringVar(value="pending")
        self.model_var = tk.StringVar()
        self.sbp_var = tk.StringVar()
        self.dbp_var = tk.StringVar()
        self.fast_var = tk.BooleanVar(value=False)
        self.unvalidated_var = tk.BooleanVar(value=False)

        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        self._build_controls(outer)
        self._build_readout(outer)

        chart_frame = ttk.LabelFrame(outer, text="Raw IR waveform · last 15 seconds · baseline removed")
        chart_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 8))
        figure = Figure(figsize=(8, 2.4), dpi=100)
        self.axes = figure.add_subplot(111)
        self.axes.set_xlabel("Seconds before latest sample")
        self.axes.set_ylabel("IR counts")
        self.axes.set_xlim(-PLOT_SECONDS, 0)
        self.axes.grid(alpha=0.25)
        (self.wave_line,) = self.axes.plot([], [], color="#157ca5", linewidth=1.3)
        figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        footer = ttk.Frame(outer)
        footer.grid(row=3, column=0, sticky="ew")
        self.health_label = ttk.Label(footer, text="PPG -- · IMU -- · I²C/FIFO --", wraplength=1000)
        self.health_label.pack(anchor="w")
        self.warning_label = ttk.Label(footer, text="Warnings: none", wraplength=1000)
        self.warning_label.pack(anchor="w", pady=(3, 0))
        ttk.Label(footer, text=RESEARCH_DISCLAIMER, foreground="#9a3412").pack(anchor="w", pady=(7, 0))

        self._change_transport()
        self._change_mode()
        self._refresh_serial_ports()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_events)

    def _build_controls(self, outer: ttk.Frame) -> None:
        controls = ttk.LabelFrame(outer, text="Connection and participant", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        for column in range(6):
            controls.columnconfigure(column, weight=1 if column in {1, 3} else 0)

        ttk.Label(controls, text="Transport").grid(row=0, column=0, sticky="w")
        self.transport_box = ttk.Combobox(
            controls, textvariable=self.transport_var, values=("serial", "ble"), state="readonly", width=10
        )
        self.transport_box.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.transport_box.bind("<<ComboboxSelected>>", lambda _event: self._change_transport())
        ttk.Label(controls, text="Device").grid(row=0, column=2, sticky="w")
        self.serial_box = ttk.Combobox(controls, textvariable=self.serial_port_var, width=28)
        self.serial_box.grid(row=0, column=3, sticky="ew", padx=(6, 8))
        self.ble_box = ttk.Combobox(controls, textvariable=self.ble_device_var, width=28)
        self.ble_box.grid(row=0, column=3, sticky="ew", padx=(6, 8))
        self.scan_button = ttk.Button(controls, text="Refresh / Scan", command=self._refresh_devices)
        self.scan_button.grid(row=0, column=4, sticky="w")
        ttk.Label(controls, text="Participant").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.participant_entry = ttk.Entry(controls, textvariable=self.participant_var, width=12)
        self.participant_entry.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        choice = ttk.Frame(controls)
        choice.grid(row=1, column=2, columnspan=3, sticky="w", pady=(8, 0))
        self.model_radio = ttk.Radiobutton(choice, text="Saved model", variable=self.mode_var, value="model", command=self._change_mode)
        self.model_radio.pack(side="left")
        self.pending_radio = ttk.Radiobutton(choice, text="Pending (no model)", variable=self.mode_var, value="pending", command=self._change_mode)
        self.pending_radio.pack(side="left", padx=(14, 0))

        self.model_frame = ttk.Frame(controls)
        self.model_frame.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(8, 0))
        self.model_frame.columnconfigure(1, weight=1)
        ttk.Label(self.model_frame, text="Model directory").grid(row=0, column=0, sticky="w")
        self.model_entry = ttk.Entry(self.model_frame, textvariable=self.model_var)
        self.model_entry.grid(row=0, column=1, sticky="ew", padx=8)
        self.browse_button = ttk.Button(self.model_frame, text="Browse…", command=self._browse_model)
        self.browse_button.grid(row=0, column=2)

        self.pending_frame = ttk.Frame(controls)
        self.pending_frame.grid(row=2, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Label(self.pending_frame, text="Calibration SBP").pack(side="left")
        self.sbp_entry = ttk.Entry(self.pending_frame, textvariable=self.sbp_var, width=7)
        self.sbp_entry.pack(side="left", padx=(6, 18))
        ttk.Label(self.pending_frame, text="DBP").pack(side="left")
        self.dbp_entry = ttk.Entry(self.pending_frame, textvariable=self.dbp_var, width=7)
        self.dbp_entry.pack(side="left", padx=6)

        advanced = ttk.LabelFrame(controls, text="Advanced / experimental", padding=5)
        advanced.grid(row=3, column=0, columnspan=5, sticky="ew", pady=(9, 0))
        self.fast_check = ttk.Checkbutton(advanced, text="Try 30-second policy", variable=self.fast_var)
        self.fast_check.pack(side="left")
        self.unvalidated_check = ttk.Checkbutton(
            advanced, text="Allow unvalidated model (development only)", variable=self.unvalidated_var
        )
        self.unvalidated_check.pack(side="left", padx=(20, 0))

        buttons = ttk.Frame(controls)
        buttons.grid(row=4, column=0, columnspan=5, sticky="w", pady=(10, 0))
        self.connect_button = ttk.Button(buttons, text="Connect", command=self._connect)
        self.connect_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.connection_label = ttk.Label(buttons, text="Not connected")
        self.connection_label.pack(side="left", padx=(18, 0))

        self.setup_widgets = [
            self.transport_box, self.serial_box, self.ble_box, self.scan_button,
            self.participant_entry, self.model_radio, self.pending_radio,
            self.model_entry, self.browse_button, self.sbp_entry, self.dbp_entry,
            self.fast_check, self.unvalidated_check,
        ]

    def _build_readout(self, outer: ttk.Frame) -> None:
        readout = ttk.LabelFrame(outer, text="Experimental BP preview", padding=12)
        readout.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        readout.columnconfigure(0, weight=1)
        readout.columnconfigure(1, weight=1)
        self.bp_label = tk.Label(readout, text="-- / --", font=("Segoe UI", 38, "bold"), fg="#27384d")
        self.bp_label.grid(row=0, column=0, sticky="w")
        ttk.Label(readout, text="mmHg").grid(row=0, column=1, sticky="w")
        self.mode_label = ttk.Label(readout, text="No valid estimate", font=("Segoe UI", 13, "bold"))
        self.mode_label.grid(row=1, column=0, columnspan=2, sticky="w")
        self.age_label = ttk.Label(readout, text="Estimate age: --")
        self.age_label.grid(row=2, column=0, columnspan=2, sticky="w")
        self.status_label = ttk.Label(readout, text="Status: Not connected", wraplength=1000)
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.reason_label = ttk.Label(readout, text="", wraplength=1000)
        self.reason_label.grid(row=4, column=0, columnspan=2, sticky="w")
        self.progress_label = ttk.Label(readout, text="Clean buffer: --")
        self.progress_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.motion_label = ttk.Label(readout, text="Motion: --")
        self.motion_label.grid(row=6, column=0, columnspan=2, sticky="w")
        self.model_label = ttk.Label(readout, text="Model: --")
        self.model_label.grid(row=7, column=0, columnspan=2, sticky="w")
        self.experimental_label = ttk.Label(readout, text="", foreground="#b42318", font=("Segoe UI", 11, "bold"))
        self.experimental_label.grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _change_transport(self) -> None:
        if self.transport_var.get() == "ble":
            self.serial_box.grid_remove()
            self.ble_box.grid()
        else:
            self.ble_box.grid_remove()
            self.serial_box.grid()

    def _change_mode(self) -> None:
        if self.mode_var.get() == "pending":
            self.model_frame.grid_remove()
            self.pending_frame.grid()
        else:
            self.pending_frame.grid_remove()
            self.model_frame.grid()

    def _browse_model(self) -> None:
        selected = filedialog.askdirectory(title="Choose saved BP model directory")
        if selected:
            self.model_var.set(selected)

    def _refresh_serial_ports(self) -> None:
        try:
            from serial.tools import list_ports

            ports = sorted(port.device for port in list_ports.comports())
        except (ImportError, OSError):
            ports = []
        self.serial_box["values"] = ports
        if ports and not self.serial_port_var.get():
            self.serial_port_var.set(ports[0])

    def _refresh_devices(self) -> None:
        if self.transport_var.get() == "serial":
            self._refresh_serial_ports()
            return
        self.scan_button.configure(state="disabled")
        self.connection_label.configure(text="Scanning for PPG-LOGGER devices…")
        token = self.session_token

        def scan() -> None:
            try:
                from bleak import BleakScanner

                devices = asyncio.run(BleakScanner.discover(timeout=4.0))
                names = sorted({device.name for device in devices if device.name and device.name.startswith(BLE_DEVICE_PREFIX)})
                self.events.put((token, "devices", names))
            except Exception as exc:
                self.events.put((token, "scan_error", str(exc)))

        threading.Thread(target=scan, daemon=True, name="ppg-ble-scan").start()

    def _set_setup_enabled(self, enabled: bool) -> None:
        for widget in self.setup_widgets:
            if widget is self.transport_box:
                widget.configure(state="readonly" if enabled else "disabled")
            elif widget in {self.serial_box, self.ble_box}:
                widget.configure(state="normal" if enabled else "disabled")
            else:
                widget.configure(state="normal" if enabled else "disabled")
        self.connect_button.configure(state="normal" if enabled else "disabled")
        self.stop_button.configure(state="disabled" if enabled else "normal")

    def _connect(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        device = self.serial_port_var.get() if self.transport_var.get() == "serial" else self.ble_device_var.get()
        try:
            if self.mode_var.get() == "model" and not self.model_var.get().strip():
                raise ValueError("Choose a saved model directory, or select pending mode.")
            args = build_connection_args(
                transport=self.transport_var.get(),
                device=device,
                participant_id=self.participant_var.get(),
                model_dir=self.model_var.get() if self.mode_var.get() == "model" else "",
                calibration_sbp=self.sbp_var.get(),
                calibration_dbp=self.dbp_var.get(),
                fast_window=self.fast_var.get(),
                allow_unvalidated=self.unvalidated_var.get(),
            )
        except (OSError, ValueError, SystemExit) as exc:
            messagebox.showerror("Cannot connect", str(exc))
            return
        self.session_token += 1
        token = self.session_token
        self.stop_event = threading.Event()
        self._set_setup_enabled(False)
        self._show_unavailable("Connecting…", "Waiting for sensor data")
        self.worker = threading.Thread(
            target=run_desktop_worker,
            args=(args, None, self.stop_event, lambda kind, payload: self.events.put((token, kind, payload))),
            daemon=True,
            name="ppg-bp-desktop",
        )
        self.worker.start()

    def _stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self._show_unavailable("Stopping…", "Closing the sensor connection")
        self.stop_button.configure(state="disabled")

    def _show_unavailable(self, status: str, reason: str = "") -> None:
        self.bp_label.configure(text="-- / --", fg="#27384d")
        self.mode_label.configure(text="No valid estimate")
        self.age_label.configure(text="Estimate age: --")
        self.status_label.configure(text=f"Status: {status}")
        self.reason_label.configure(text=reason)
        self.connection_label.configure(text=status)
        self.progress_label.configure(text="Clean buffer: --")
        self.motion_label.configure(text="Motion: --")
        self.model_label.configure(text=f"Model: {status}" if status == "Model incompatible" else "Model: --")
        self.health_label.configure(text="PPG -- · IMU -- · I²C/FIFO --")
        self.experimental_label.configure(text="")
        self.wave_line.set_data([], [])
        self.canvas.draw_idle()

    def _render(self, snapshot: DesktopSnapshot) -> None:
        if snapshot.mode == "unavailable" or snapshot.sbp is None or snapshot.dbp is None:
            self.bp_label.configure(text="-- / --", fg="#27384d")
            self.mode_label.configure(text="No valid estimate")
        else:
            self.bp_label.configure(
                text=f"{snapshot.sbp:.0f} / {snapshot.dbp:.0f}",
                fg="#9a3412" if snapshot.mode == "held" else "#116466",
            )
            self.mode_label.configure(
                text="LAST VALIDATED · not a new measurement" if snapshot.mode == "held" else "CURRENT CLEAN ESTIMATE"
            )
        self.age_label.configure(text=f"Estimate age: {_format_age(snapshot.age_s)}")
        self.status_label.configure(text=f"Status: {snapshot.status} · Current signal: {snapshot.current_status}")
        self.reason_label.configure(text=snapshot.current_reason if snapshot.mode == "held" else snapshot.reason)
        self.progress_label.configure(
            text=(
                f"Clean buffer: {snapshot.buffer_s:.1f}/{snapshot.target_s:.0f} s  ·  "
                f"Accepted windows: {snapshot.accepted_windows}/{snapshot.total_windows}  ·  "
                f"Clean coverage: {snapshot.clean_coverage_s:.1f} s"
            )
        )
        intensity = snapshot.intensity
        if snapshot.intensity_g is not None:
            intensity += f" ({snapshot.intensity_g:.3f} g)"
        self.motion_label.configure(text=f"Motion: {snapshot.motion} · Intensity: {intensity}")
        self.model_label.configure(text=f"Model: {snapshot.model_eligibility}")
        self.experimental_label.configure(text=snapshot.experimental_warning)
        self.connection_label.configure(text=f"{snapshot.device} · {snapshot.connection}")
        self.health_label.configure(
            text=(
                f"PPG {_format_rate(snapshot.ppg_rate_hz)} · I²C {_format_count(snapshot.ppg_i2c_errors)} "
                f"· FIFO {_format_count(snapshot.ppg_fifo_overflows)}    |    "
                f"IMU {_format_rate(snapshot.imu_rate_hz)} · I²C {_format_count(snapshot.imu_i2c_errors)} "
                f"· FIFO {_format_count(snapshot.imu_fifo_overflows)}"
            )
        )
        self.warning_label.configure(text="Warnings: " + (" | ".join(snapshot.warnings) if snapshot.warnings else "none"))
        self.wave_line.set_data(snapshot.waveform_x_s, snapshot.waveform_ir)
        if snapshot.waveform_ir:
            low, high = min(snapshot.waveform_ir), max(snapshot.waveform_ir)
            margin = max(1.0, (high - low) * 0.12)
            self.axes.set_ylim(low - margin, high + margin)
        self.canvas.draw_idle()

    def _drain_events(self) -> None:
        try:
            while True:
                token, kind, payload = self.events.get_nowait()
                if token != self.session_token:
                    continue
                if kind == "snapshot" and self.stop_event is not None and not self.stop_event.is_set():
                    self._render(payload)
                elif kind == "error":
                    if self.stop_event is None or not self.stop_event.is_set():
                        self._show_unavailable("Connection error", str(payload))
                        self.warning_label.configure(text=f"Warnings: {payload}")
                elif kind == "model_error":
                    self._show_unavailable("Model incompatible", str(payload))
                    self.warning_label.configure(text=f"Warnings: {payload}")
                elif kind == "stopped":
                    self.worker = None
                    self.stop_event = None
                    self._set_setup_enabled(True)
                    if self.closing:
                        self.root.destroy()
                        return
                    if self.status_label.cget("text") not in {"Status: Connection error", "Status: Model incompatible"}:
                        self._show_unavailable("Not connected")
                elif kind == "devices":
                    self.scan_button.configure(state="normal")
                    self.ble_box["values"] = payload
                    if len(payload) == 1:
                        self.ble_device_var.set(payload[0])
                    self.connection_label.configure(
                        text=f"Found {len(payload)} PPG-LOGGER device(s)" if payload else "No PPG-LOGGER devices found"
                    )
                elif kind == "scan_error":
                    self.scan_button.configure(state="normal")
                    self.connection_label.configure(text=f"BLE scan failed: {payload}")
        except queue.Empty:
            pass
        if not self.closing or self.worker is not None:
            self.root.after(100, self._drain_events)

    def _on_close(self) -> None:
        if self.worker is None:
            self.root.destroy()
            return
        self.closing = True
        self._stop()


def main() -> int:
    root = tk.Tk()
    DesktopBPApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
