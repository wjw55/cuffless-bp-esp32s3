"""Frozen motion-quality classifier inference for PC-side shadow operation."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from motion_quality import extract_synchronized_window_features, load_config


SHADOW_HISTORY_SECONDS = 90.0
SHADOW_COLUMNS = [
    "elapsed_s",
    "window_start_timestamp_ms",
    "window_end_timestamp_ms",
    "prediction",
    "usable",
    "unusable_probability",
    "decision_threshold",
    "status",
    "reason",
    "ppg_sample_count",
    "imu_sample_count",
    "ppg_completeness",
    "imu_completeness",
    "model_name",
    "model_sha256",
    "shadow_mode",
    "affects_bp_or_hr",
]


class ShadowModelCompatibilityError(ValueError):
    """Raised when a frozen classifier cannot safely consume live features."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MotionQualityShadowBundle:
    model: Any
    model_name: str
    feature_columns: tuple[str, ...]
    decision_threshold: float
    config: dict[str, Any]
    config_sha256: str
    model_sha256: str
    model_path: Path


@dataclass
class MotionQualityShadowResult:
    status: str = "warming_up"
    reason: str = "waiting for synchronized PPG and IMU"
    prediction: str | None = None
    unusable_probability: float | None = None
    window_start_timestamp_ms: float | None = None
    window_end_timestamp_ms: float | None = None
    ppg_sample_count: int = 0
    imu_sample_count: int = 0
    ppg_completeness: float = 0.0
    imu_completeness: float = 0.0

    @property
    def usable(self) -> bool | None:
        if self.prediction is None:
            return None
        return self.prediction == "usable"


@dataclass
class MotionQualityShadowState:
    bundle: MotionQualityShadowBundle
    ppg_samples: deque[tuple[int, int, int, int]] = field(default_factory=deque)
    imu_samples: deque[tuple[int, int, int, int, int]] = field(default_factory=deque)
    result: MotionQualityShadowResult = field(default_factory=MotionQualityShadowResult)
    last_window_end_ms: float | None = None
    prediction_count: int = 0
    usable_count: int = 0
    unusable_count: int = 0
    health_fault: str | None = None

    def _reset(self, reason: str) -> None:
        self.ppg_samples.clear()
        self.imu_samples.clear()
        self.last_window_end_ms = None
        self.result = MotionQualityShadowResult(status="invalid_timing", reason=reason)

    def add_ppg(self, row: tuple[int, int, int, int]) -> None:
        if self.ppg_samples:
            previous = self.ppg_samples[-1]
            if row[0] != previous[0] + 1 or row[1] <= previous[1] or row[1] - previous[1] > 40:
                self._reset("PPG continuity fault; shadow history restarted")
        self.ppg_samples.append(row)
        self._prune()

    def add_imu(self, row: tuple[int, int, int, int, int]) -> None:
        if self.imu_samples:
            previous = self.imu_samples[-1]
            if row[0] != previous[0] + 1 or row[1] <= previous[1] or row[1] - previous[1] > 40:
                self._reset("IMU continuity fault; shadow history restarted")
        self.imu_samples.append(row)
        self._prune()

    def add_health(self, kind: str, fields: dict[str, Any]) -> None:
        """Latch firmware sensor-health faults for the current serial session."""
        keys = ("i2c_errors", "ovf") if kind == "stats" else ("i2c_errors", "fifo_overflows")
        faults = []
        for key in keys:
            try:
                value = int(fields.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 1
            if value:
                faults.append(f"{kind}.{key}={value}")
        if faults:
            self.health_fault = ";".join(faults)
            self._reset(f"sensor health fault: {self.health_fault}")

    def _prune(self) -> None:
        latest = max(
            self.ppg_samples[-1][1] if self.ppg_samples else 0,
            self.imu_samples[-1][1] if self.imu_samples else 0,
        )
        cutoff = latest - int(SHADOW_HISTORY_SECONDS * 1000)
        while self.ppg_samples and self.ppg_samples[0][1] < cutoff:
            self.ppg_samples.popleft()
        while self.imu_samples and self.imu_samples[0][1] < cutoff:
            self.imu_samples.popleft()


def load_shadow_bundle(model_path: str | Path, config_path: str | Path) -> MotionQualityShadowBundle:
    model_path = Path(model_path)
    config_path = Path(config_path)
    if not model_path.exists():
        raise ShadowModelCompatibilityError(f"Motion-quality model does not exist: {model_path}")
    if not config_path.exists():
        raise ShadowModelCompatibilityError(f"Motion-quality config does not exist: {config_path}")
    try:
        package = joblib.load(model_path)
    except Exception as exc:
        raise ShadowModelCompatibilityError(f"Cannot load motion-quality model: {exc}") from exc
    if not isinstance(package, dict):
        raise ShadowModelCompatibilityError("Motion-quality model package must be a dictionary")
    required = {"model", "model_name", "feature_columns", "decision_threshold", "config_sha256"}
    missing = sorted(required - set(package))
    if missing:
        raise ShadowModelCompatibilityError("Motion-quality model is missing fields: " + ", ".join(missing))
    config_hash = _sha256(config_path)
    if str(package["config_sha256"]) != config_hash:
        raise ShadowModelCompatibilityError("Motion-quality model and config checksums do not match")
    model = package["model"]
    if not hasattr(model, "predict_proba"):
        raise ShadowModelCompatibilityError("Motion-quality model does not support probability prediction")
    classes = np.asarray(getattr(model, "classes_", []))
    if not np.array_equal(classes, np.asarray([0, 1])):
        raise ShadowModelCompatibilityError("Motion-quality model classes must be usable=0, unusable=1")
    threshold = float(package["decision_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ShadowModelCompatibilityError("Motion-quality decision threshold must be between zero and one")
    feature_columns = tuple(str(value) for value in package["feature_columns"])
    if not feature_columns:
        raise ShadowModelCompatibilityError("Motion-quality model has no feature columns")
    return MotionQualityShadowBundle(
        model=model,
        model_name=str(package["model_name"]),
        feature_columns=feature_columns,
        decision_threshold=threshold,
        config=load_config(config_path),
        config_sha256=config_hash,
        model_sha256=_sha256(model_path),
        model_path=model_path.resolve(),
    )


def _median_period_ms(rows: deque, timestamp_index: int = 1) -> float:
    recent = list(rows)[-101:]
    values = np.asarray([row[timestamp_index] for row in recent], dtype=float)
    return float(np.median(np.diff(values))) if len(values) >= 2 else 10.0


def maybe_score_shadow(state: MotionQualityShadowState) -> bool:
    """Score the newest completed 8-second window, at most once per 4 seconds."""
    window_ms = float(state.bundle.config["window_seconds"]) * 1000.0
    step_ms = float(state.bundle.config["window_step_seconds"]) * 1000.0
    if state.health_fault:
        state.result = MotionQualityShadowResult(
            status="invalid_health",
            reason=f"sensor health fault: {state.health_fault}",
        )
        return False
    if len(state.ppg_samples) < 4 or len(state.imu_samples) < 4:
        state.result = MotionQualityShadowResult(reason="waiting for synchronized PPG and IMU")
        return False
    ppg_end = state.ppg_samples[-1][1] + _median_period_ms(state.ppg_samples)
    imu_end = state.imu_samples[-1][1] + _median_period_ms(state.imu_samples)
    end_ms = min(ppg_end, imu_end)
    start_ms = end_ms - window_ms
    common_start = max(state.ppg_samples[0][1], state.imu_samples[0][1])
    if start_ms < common_start:
        available = max(0.0, (end_ms - common_start) / 1000.0)
        state.result = MotionQualityShadowResult(
            status="warming_up",
            reason=f"collecting synchronized data: {available:.1f}/{window_ms / 1000.0:.0f} s",
        )
        return False
    if state.last_window_end_ms is not None and end_ms - state.last_window_end_ms < step_ms - 1e-6:
        return False

    ppg = pd.DataFrame(list(state.ppg_samples), columns=["sample_seq", "timestamp_ms", "red", "ir"])
    imu = pd.DataFrame(
        list(state.imu_samples), columns=["imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"]
    )
    state.last_window_end_ms = end_ms
    try:
        features, diagnostics = extract_synchronized_window_features(
            ppg, imu, state.bundle.config, start_ms, end_ms
        )
        if features is None:
            reasons = diagnostics.get("rejection_reasons", [])
            state.result = MotionQualityShadowResult(
                status="invalid_window",
                reason=";".join(str(value) for value in reasons) or "window feature extraction rejected",
                window_start_timestamp_ms=start_ms,
                window_end_timestamp_ms=end_ms,
                ppg_sample_count=int(diagnostics.get("ppg_sample_count", 0)),
                imu_sample_count=int(diagnostics.get("imu_sample_count", 0)),
                ppg_completeness=float(diagnostics.get("ppg_completeness", 0.0)),
                imu_completeness=float(diagnostics.get("imu_completeness", 0.0)),
            )
            return True
        missing = [column for column in state.bundle.feature_columns if column not in features]
        if missing:
            raise ShadowModelCompatibilityError("Live feature schema is missing: " + ", ".join(missing))
        feature_frame = pd.DataFrame(
            [{column: features[column] for column in state.bundle.feature_columns}],
            columns=list(state.bundle.feature_columns),
        )
        probability_matrix = np.asarray(state.bundle.model.predict_proba(feature_frame), dtype=float)
        if probability_matrix.shape != (1, 2) or not np.isfinite(probability_matrix).all():
            raise ValueError("classifier returned invalid probabilities")
        probability = float(probability_matrix[0, 1])
        prediction = "unusable" if probability >= state.bundle.decision_threshold else "usable"
        state.result = MotionQualityShadowResult(
            status=prediction,
            reason="shadow prediction only; BP/HR behavior unchanged",
            prediction=prediction,
            unusable_probability=probability,
            window_start_timestamp_ms=start_ms,
            window_end_timestamp_ms=end_ms,
            ppg_sample_count=int(diagnostics["ppg_sample_count"]),
            imu_sample_count=int(diagnostics["imu_sample_count"]),
            ppg_completeness=float(diagnostics["ppg_completeness"]),
            imu_completeness=float(diagnostics["imu_completeness"]),
        )
        state.prediction_count += 1
        if prediction == "usable":
            state.usable_count += 1
        else:
            state.unusable_count += 1
    except Exception as exc:
        state.result = MotionQualityShadowResult(
            status="analysis_error",
            reason=str(exc),
            window_start_timestamp_ms=start_ms,
            window_end_timestamp_ms=end_ms,
        )
    return True


def build_shadow_record(state: MotionQualityShadowState, elapsed_s: float) -> dict[str, Any]:
    result = state.result
    row = {
        "elapsed_s": round(max(0.0, elapsed_s), 3),
        "window_start_timestamp_ms": result.window_start_timestamp_ms,
        "window_end_timestamp_ms": result.window_end_timestamp_ms,
        "prediction": result.prediction,
        "usable": result.usable,
        "unusable_probability": (
            round(result.unusable_probability, 6) if result.unusable_probability is not None else None
        ),
        "decision_threshold": state.bundle.decision_threshold,
        "status": result.status,
        "reason": result.reason,
        "ppg_sample_count": result.ppg_sample_count,
        "imu_sample_count": result.imu_sample_count,
        "ppg_completeness": round(result.ppg_completeness, 6),
        "imu_completeness": round(result.imu_completeness, 6),
        "model_name": state.bundle.model_name,
        "model_sha256": state.bundle.model_sha256,
        "shadow_mode": True,
        "affects_bp_or_hr": False,
    }
    return {column: row[column] for column in SHADOW_COLUMNS}
