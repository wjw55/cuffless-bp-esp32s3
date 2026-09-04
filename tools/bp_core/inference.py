from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .datasets import Recording, SignalData
from .features import aggregate_recording_features, process_signal


MODEL_MANIFEST_SCHEMA_VERSION = 1


class ModelCompatibilityError(ValueError):
    pass


@dataclass
class BPModelBundle:
    model_dir: Path
    participant_id: str
    calibration_id: str
    calibration_sbp: float
    calibration_dbp: float
    calibration_features: dict[str, float]
    config: dict[str, Any]
    packages: dict[str, dict[str, Any]]
    viewer_eligible: bool
    allow_unvalidated: bool
    manifest: dict[str, Any]


@dataclass
class BPInferenceResult:
    status: str
    reason: str
    sbp: float | None = None
    dbp: float | None = None
    delta_sbp: float | None = None
    delta_dbp: float | None = None
    accepted_windows: int = 0
    total_windows: int = 0
    pulse_rate_bpm: float | None = None
    model_eligible: bool = False

    @property
    def numeric_available(self) -> bool:
        return self.sbp is not None and self.dbp is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_bundle(
    model_dir: str | Path,
    *,
    expected_participant_id: str | None = None,
    allow_unvalidated: bool = False,
) -> BPModelBundle:
    root = Path(model_dir).resolve()
    manifest_path = root / "model_manifest.json"
    config_path = root / "config_snapshot.json"
    if not manifest_path.exists() or not config_path.exists():
        raise ModelCompatibilityError("model directory must contain model_manifest.json and config_snapshot.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError(f"cannot read model manifest/configuration: {exc}") from exc
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA_VERSION:
        raise ModelCompatibilityError("unsupported or missing model manifest schema_version")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise ModelCompatibilityError("model configuration checksum does not match config_snapshot.json")
    participant_id = str(manifest.get("participant_id", ""))
    if not participant_id:
        raise ModelCompatibilityError("model manifest has no participant_id")
    if expected_participant_id is not None and participant_id != str(expected_participant_id):
        raise ModelCompatibilityError(
            f"model participant {participant_id!r} does not match requested participant {expected_participant_id!r}"
        )
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ModelCompatibilityError("model manifest has no calibration record")
    calibration_id = str(calibration.get("label_group_id", ""))
    calibration_features = calibration.get("features")
    if not calibration_id or not isinstance(calibration_features, dict):
        raise ModelCompatibilityError("model calibration ID or morphology features are missing")
    try:
        calibration_sbp = float(calibration["sbp"])
        calibration_dbp = float(calibration["dbp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelCompatibilityError("model calibration BP is invalid") from exc
    if (
        not math.isfinite(calibration_sbp)
        or not math.isfinite(calibration_dbp)
        or calibration_sbp <= calibration_dbp
    ):
        raise ModelCompatibilityError("model calibration BP is non-finite or SBP is not greater than DBP")

    packages: dict[str, dict[str, Any]] = {}
    model_entries = manifest.get("models")
    if not isinstance(model_entries, dict):
        raise ModelCompatibilityError("model manifest has no models object")
    for target in ("sbp", "dbp"):
        entry = model_entries.get(target)
        if not isinstance(entry, dict) or not entry.get("file"):
            raise ModelCompatibilityError(f"model manifest is missing {target.upper()} model")
        path = root / str(entry["file"])
        if not path.exists():
            raise ModelCompatibilityError(f"model file does not exist: {path}")
        try:
            package = joblib.load(path)
        except Exception as exc:
            raise ModelCompatibilityError(f"cannot load {target.upper()} model: {exc}") from exc
        if not isinstance(package, dict) or package.get("target") != target:
            raise ModelCompatibilityError(f"invalid {target.upper()} model package")
        if str(package.get("participant_id")) != participant_id:
            raise ModelCompatibilityError(f"{target.upper()} model participant does not match manifest")
        if str(package.get("calibration_label_group_id")) != calibration_id:
            raise ModelCompatibilityError(f"{target.upper()} model calibration does not match manifest")
        columns = package.get("feature_columns")
        if not isinstance(columns, list) or len(columns) != int(entry.get("feature_count", -1)):
            raise ModelCompatibilityError(f"{target.upper()} feature schema does not match manifest")
        if columns != entry.get("feature_columns"):
            raise ModelCompatibilityError(f"{target.upper()} feature columns do not match manifest")
        if package.get("model_manifest_schema_version") != MODEL_MANIFEST_SCHEMA_VERSION:
            raise ModelCompatibilityError(f"{target.upper()} model schema does not match manifest")
        if package.get("config_sha256") != manifest.get("config_sha256"):
            raise ModelCompatibilityError(f"{target.upper()} model configuration does not match manifest")
        if not hasattr(package.get("estimator"), "predict"):
            raise ModelCompatibilityError(f"{target.upper()} package has no usable estimator")
        packages[target] = package

    viewer_eligible = bool(manifest.get("viewer_eligible", False))
    target_gate = all(
        manifest["models"][target].get("beats_zero_change_on_locked_test") is True
        for target in ("sbp", "dbp")
    )
    if viewer_eligible != target_gate or bool(manifest.get("passes_both_targets", False)) != target_gate:
        raise ModelCompatibilityError("viewer eligibility is inconsistent with per-target baseline results")
    try:
        normalized_calibration_features = {
            str(key): float(value) for key, value in calibration_features.items()
        }
    except (TypeError, ValueError) as exc:
        raise ModelCompatibilityError("model calibration morphology features are invalid") from exc
    if not normalized_calibration_features or not all(
        math.isfinite(value) for value in normalized_calibration_features.values()
    ):
        raise ModelCompatibilityError("model calibration morphology features are empty or non-finite")
    return BPModelBundle(
        model_dir=root,
        participant_id=participant_id,
        calibration_id=calibration_id,
        calibration_sbp=calibration_sbp,
        calibration_dbp=calibration_dbp,
        calibration_features=normalized_calibration_features,
        config=config,
        packages=packages,
        viewer_eligible=viewer_eligible,
        allow_unvalidated=bool(allow_unvalidated),
        manifest=manifest,
    )


def signal_from_ppg_frame(frame: pd.DataFrame, metadata: dict[str, Any]) -> SignalData:
    required = ["sample_seq", "timestamp_ms", "red", "ir"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"PPG data is missing columns: {', '.join(missing)}")
    timestamp = pd.to_numeric(frame["timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    if not len(timestamp):
        raise ValueError("PPG data is empty")
    local_metadata = dict(metadata)
    local_metadata["bp_pipeline_first_timestamp_ms"] = float(timestamp[0])
    return SignalData(
        time_s=(timestamp - timestamp[0]) / 1000.0,
        ir=pd.to_numeric(frame["ir"], errors="coerce").to_numpy(dtype=float),
        red=pd.to_numeric(frame["red"], errors="coerce").to_numpy(dtype=float),
        acceleration_m_s2=None,
        sample_sequence=pd.to_numeric(frame["sample_seq"], errors="coerce").to_numpy(dtype=float),
        metadata=local_metadata,
    )


def _recording_context(participant_id: str) -> Recording:
    return Recording(
        dataset_id="local_upper_arm",
        participant_id=participant_id,
        session_id="live_bp",
        recording_id="live",
        label_group_id=f"local_upper_arm:{participant_id}:live_bp:live",
        chronological_order="live",
        sensor_site="upper_arm",
        sample_rate_hz=100.0,
        ppg_path="",
        imu_path=None,
        sbp=math.nan,
        dbp=math.nan,
        reference_hr=None,
        label_source="none",
        label_timing=None,
        quality_status="live",
        source_format="live_ppg",
    )


def extract_current_features(
    participant_id: str,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    for key in (
        "firmware_i2c_error_count",
        "firmware_fifo_overflow_count",
        "imu_firmware_i2c_error_count",
        "imu_firmware_fifo_overflow_count",
    ):
        try:
            if int(metadata.get(key, 0) or 0) > 0:
                raise ValueError(f"sensor_health_error:{key}={metadata[key]}")
        except (TypeError, ValueError) as exc:
            if str(exc).startswith("sensor_health_error"):
                raise
            raise ValueError(f"invalid_sensor_health_counter:{key}") from exc
    recording = _recording_context(participant_id)
    segments = pd.DataFrame(process_signal(recording, signal_from_ppg_frame(frame, metadata), config))
    occasion = aggregate_recording_features(recording, segments, config)
    return occasion, segments


def _raw_feature_name(short_name: str) -> str:
    if short_name.startswith("median__"):
        return short_name.replace("median__", "median__feature__", 1)
    if short_name.startswith("iqr__"):
        return short_name.replace("iqr__", "iqr__feature__", 1)
    raise ModelCompatibilityError(f"unsupported model feature name: {short_name}")


def _model_input(bundle: BPModelBundle, occasion: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    values: dict[str, float] = {}
    for column in columns:
        if column == "baseline_sbp":
            values[column] = bundle.calibration_sbp
            continue
        if column == "baseline_dbp":
            values[column] = bundle.calibration_dbp
            continue
        prefix, separator, short_name = column.partition("__")
        if not separator or prefix not in {"current", "calibration", "change"}:
            raise ModelCompatibilityError(f"unsupported model input column: {column}")
        raw_name = _raw_feature_name(short_name)
        if raw_name not in occasion or raw_name not in bundle.calibration_features:
            raise ModelCompatibilityError(f"required morphology feature is missing: {raw_name}")
        current = float(occasion[raw_name])
        calibration = float(bundle.calibration_features[raw_name])
        values[column] = {"current": current, "calibration": calibration, "change": current - calibration}[prefix]
    return pd.DataFrame([values], columns=columns)


def _quality_status(segments: pd.DataFrame) -> tuple[str, str]:
    if segments.empty:
        return "insufficient_clean_data", "no analysis windows were available"
    rejected = segments[segments["accepted"] != True]  # noqa: E712
    reasons = ";".join(rejected.get("rejection_reason", pd.Series(dtype=str)).fillna("").astype(str))
    if "motion" in reasons:
        return "motion_contaminated", "movement affected the analysis windows"
    if "contact_step" in reasons or "poor_contact" in reasons:
        return "contact_artifact", "contact quality affected the analysis windows"
    if "missing" in reasons or "incomplete" in reasons:
        return "invalid_timing", "sample timing or completeness failed"
    return "poor_waveform_quality", "too few waveform windows passed quality checks"


def predict_frame(
    bundle: BPModelBundle,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> BPInferenceResult:
    try:
        occasion, segments = extract_current_features(
            bundle.participant_id, frame, metadata, bundle.config
        )
    except (ValueError, FloatingPointError) as exc:
        text = str(exc)
        status = "invalid_timing" if "sequence" in text or "timestamp" in text or "sensor_health" in text else "analysis_error"
        return BPInferenceResult(status=status, reason=text, model_eligible=bundle.viewer_eligible)
    accepted = int(occasion["accepted_segment_count"])
    total = int(occasion["total_segment_count"])
    pulse = occasion.get("median__feature__pulse_rate_bpm")
    pulse_rate = float(pulse) if pulse is not None and pd.notna(pulse) else None
    if not bool(occasion["occasion_usable"]):
        status, reason = _quality_status(segments)
        return BPInferenceResult(
            status=status,
            reason=reason,
            accepted_windows=accepted,
            total_windows=total,
            pulse_rate_bpm=pulse_rate,
            model_eligible=bundle.viewer_eligible,
        )
    if not bundle.viewer_eligible and not bundle.allow_unvalidated:
        return BPInferenceResult(
            status="model_validation_failed",
            reason="saved SBP and DBP models did not both beat zero-change",
            accepted_windows=accepted,
            total_windows=total,
            pulse_rate_bpm=pulse_rate,
            model_eligible=False,
        )
    predicted: dict[str, float] = {}
    deltas: dict[str, float] = {}
    try:
        for target in ("sbp", "dbp"):
            package = bundle.packages[target]
            delta = float(package["estimator"].predict(_model_input(bundle, occasion, package["feature_columns"]))[0])
            deltas[target] = delta
            predicted[target] = (bundle.calibration_sbp if target == "sbp" else bundle.calibration_dbp) + delta
    except (KeyError, TypeError, ValueError, FloatingPointError, ModelCompatibilityError) as exc:
        return BPInferenceResult(
            status="model_incompatible",
            reason=str(exc),
            accepted_windows=accepted,
            total_windows=total,
            pulse_rate_bpm=pulse_rate,
            model_eligible=bundle.viewer_eligible,
        )
    if not all(math.isfinite(value) for value in predicted.values()) or predicted["sbp"] <= predicted["dbp"]:
        return BPInferenceResult(
            status="invalid_model_output",
            reason="model output was non-finite or SBP was not greater than DBP",
            accepted_windows=accepted,
            total_windows=total,
            pulse_rate_bpm=pulse_rate,
            model_eligible=bundle.viewer_eligible,
        )
    return BPInferenceResult(
        status="prediction_ready" if bundle.viewer_eligible else "unvalidated_estimate",
        reason="quality-gated personalized research estimate",
        sbp=predicted["sbp"],
        dbp=predicted["dbp"],
        delta_sbp=deltas["sbp"],
        delta_dbp=deltas["dbp"],
        accepted_windows=accepted,
        total_windows=total,
        pulse_rate_bpm=pulse_rate,
        model_eligible=bundle.viewer_eligible,
    )
