from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import resolve_path


ONE_MONTH_REQUIRED_COLUMNS = [
    "Time [sec]",
    "Acceleration X [m/sec2]",
    "Acceleration Y [m/sec2]",
    "Acceleration Z [m/sec2]",
    "PPG Red [-]",
    "PPG IR [-]",
    "SBP [mmHg]",
    "DBP [mmHg]",
    "HR [bpm]",
]
LOCAL_PPG_COLUMNS = ["sample_seq", "timestamp_ms", "red", "ir"]
LOCAL_IMU_COLUMNS = ["imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"]


@dataclass
class Recording:
    dataset_id: str
    participant_id: str
    session_id: str
    recording_id: str
    label_group_id: str
    chronological_order: str
    sensor_site: str
    sample_rate_hz: float | None
    ppg_path: str
    imu_path: str | None
    sbp: float
    dbp: float
    reference_hr: float | None
    label_source: str
    label_timing: str | None
    quality_status: str
    calibration_occasion: bool = False
    source_format: str = ""
    metadata_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["extra"] = json.dumps(self.extra, sort_keys=True)
        return row


@dataclass
class SignalData:
    time_s: np.ndarray
    ir: np.ndarray
    red: np.ndarray | None
    acceleration_m_s2: np.ndarray | None
    sample_sequence: np.ndarray | None
    metadata: dict[str, Any]


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _constant_number(series: pd.Series, name: str, path: Path) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{path}: expected one constant {name}, found {values.tolist()}")
    return float(values[0])


def _sampling_rate(time_s: np.ndarray) -> float | None:
    if len(time_s) < 2:
        return None
    differences = np.diff(time_s.astype(float))
    valid = differences[np.isfinite(differences) & (differences > 0)]
    return float(1.0 / np.median(valid)) if len(valid) else None


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_one_month(dataset: dict[str, Any]) -> list[Recording]:
    root = resolve_path(dataset["root"])
    recordings: list[Recording] = []
    for path in sorted(root.rglob("ESP*(PPG-IMU).csv")):
        relative = path.relative_to(root)
        if len(relative.parts) < 5:
            continue
        participant, week, session, timestamp = relative.parts[:4]
        frame = pd.read_csv(path, usecols=lambda value: value in ONE_MONTH_REQUIRED_COLUMNS)
        missing = [column for column in ONE_MONTH_REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        time_s = pd.to_numeric(frame["Time [sec]"], errors="coerce").to_numpy(dtype=float)
        sbp = _constant_number(frame["SBP [mmHg]"], "SBP", path)
        dbp = _constant_number(frame["DBP [mmHg]"], "DBP", path)
        hr = _constant_number(frame["HR [bpm]"], "HR", path)
        session_id = f"{week}_{session}_{timestamp}"
        recordings.append(
            Recording(
                dataset_id="one_month_wrist",
                participant_id=participant,
                session_id=session_id,
                recording_id=session_id,
                label_group_id=f"one_month_wrist:{participant}:{session_id}",
                chronological_order=timestamp,
                sensor_site=dataset.get("sensor_site", "wrist"),
                sample_rate_hz=_sampling_rate(time_s),
                ppg_path=str(path.resolve()),
                imu_path=str(path.resolve()),
                sbp=sbp,
                dbp=dbp,
                reference_hr=hr,
                label_source="session_cuff",
                label_timing="dataset_session_label",
                quality_status="unreviewed",
                source_format="one_month_raw_csv",
                extra={"week": week, "session": session, "timestamp_folder": timestamp},
            )
        )
    return recordings


def _load_label_rows(labels_dir: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    labels: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in sorted(labels_dir.glob("*_labels.csv")) if labels_dir.exists() else []:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                key = (
                    str(row.get("subject", "")).strip(),
                    str(row.get("session", "")).strip(),
                    str(row.get("trial_id", "")).strip(),
                )
                if not all(key):
                    raise ValueError(f"{path}:{line_number}: incomplete label identity")
                if key in labels:
                    raise ValueError(f"Duplicate local label for {key} in {path}")
                labels[key] = row
    return labels


def _resolve_local_path(raw_dir: Path, metadata_path: Path, value: Any, fallback_suffix: str) -> Path:
    candidates: list[Path] = []
    if value:
        supplied = Path(str(value))
        candidates.extend([supplied, raw_dir / supplied.name])
    candidates.append(raw_dir / metadata_path.name.replace("_metadata.json", fallback_suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _merge_local_value(metadata: dict[str, Any], label: dict[str, str] | None, metadata_key: str, label_key: str) -> Any:
    metadata_value = metadata.get(metadata_key)
    label_value = None if label is None else label.get(label_key)
    if label_value is not None and str(label_value).strip() != "":
        if metadata_value not in (None, ""):
            left = _number(metadata_value)
            right = _number(label_value)
            matches = math.isclose(left, right, abs_tol=1e-9) if left is not None and right is not None else str(metadata_value).strip() == str(label_value).strip()
            if not matches:
                raise ValueError(
                    f"Conflicting local {metadata_key}: metadata={metadata_value!r}, label={label_value!r}"
                )
        return label_value
    return metadata_value


def _discover_local(dataset: dict[str, Any]) -> list[Recording]:
    raw_dir = resolve_path(dataset["raw_dir"])
    labels = _load_label_rows(resolve_path(dataset["labels_dir"]))
    required_profile = str(dataset.get("required_profile", ""))
    recordings: list[Recording] = []
    for metadata_path in sorted(raw_dir.glob("*_metadata.json")) if raw_dir.exists() else []:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if required_profile and metadata.get("ppg_profile") != required_profile:
            continue
        participant = str(metadata.get("subject_id", "")).strip()
        session = str(metadata.get("session_id", "")).strip()
        trial = str(metadata.get("trial_id", "")).strip()
        if not all((participant, session, trial)):
            continue
        label = labels.get((participant, session, trial))
        sbp = _number(_merge_local_value(metadata, label, "systolic_mmHg", "sbp"))
        dbp = _number(_merge_local_value(metadata, label, "diastolic_mmHg", "dbp"))
        if sbp is None or dbp is None:
            continue
        reference_hr = _number(_merge_local_value(metadata, label, "cuff_hr_bpm", "omron_hr"))
        ppg_path = _resolve_local_path(raw_dir, metadata_path, metadata.get("output_csv_path"), "_ppg.csv")
        imu_path = _resolve_local_path(raw_dir, metadata_path, metadata.get("output_imu_csv_path"), "_imu.csv")
        recording_time = str(metadata.get("recording_start_time") or f"{session}:{trial}")
        label_timing = _merge_local_value(metadata, label, "cuff_timing", "omron_timing")
        quality = str((label or {}).get("quality") or metadata.get("timing_quality") or "unreviewed")
        recordings.append(
            Recording(
                dataset_id="local_upper_arm",
                participant_id=participant,
                session_id=session,
                recording_id=trial,
                label_group_id=f"local_upper_arm:{participant}:{session}:{trial}",
                chronological_order=recording_time,
                sensor_site=str(metadata.get("sensor_location") or dataset.get("sensor_site", "upper_arm")),
                sample_rate_hz=_number(metadata.get("approximate_sampling_rate_hz")),
                ppg_path=str(ppg_path),
                imu_path=str(imu_path) if imu_path.exists() else None,
                sbp=sbp,
                dbp=dbp,
                reference_hr=reference_hr,
                label_source="omron",
                label_timing=str(label_timing) if label_timing else None,
                quality_status=quality,
                source_format="local_ppg_csv",
                metadata_path=str(metadata_path.resolve()),
                extra={"ppg_profile": metadata.get("ppg_profile")},
            )
        )
    return recordings


def _find_ppg_bp_workbook(root: Path) -> Path | None:
    candidates = [path for path in root.rglob("*.xlsx") if not path.name.startswith("~$")]
    return sorted(candidates, key=lambda path: ("ppg" not in path.name.lower(), len(path.parts)))[0] if candidates else None


def _normalized_column(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _find_column(columns: Iterable[Any], terms: tuple[str, ...]) -> Any | None:
    normalized_columns = [(original, _normalized_column(original)) for original in columns]
    for term in terms:
        for original, normalized in normalized_columns:
            if normalized == term or term in normalized:
                return original
    return None


def _read_ppg_bp_table(workbook: Path) -> pd.DataFrame:
    raw = pd.read_excel(workbook, header=None)
    header_index = None
    for index, row in raw.head(20).iterrows():
        normalized = {_normalized_column(value) for value in row if not pd.isna(value)}
        if any("subject_id" in value for value in normalized) and any("systolic" in value for value in normalized):
            header_index = int(index)
            break
    if header_index is None:
        raise ValueError(f"{workbook}: cannot locate the physiological-information header row")
    frame = raw.iloc[header_index + 1 :].copy()
    frame.columns = [str(value).strip() for value in raw.iloc[header_index]]
    return frame.dropna(how="all").reset_index(drop=True)


def _discover_ppg_bp(dataset: dict[str, Any]) -> list[Recording]:
    root = resolve_path(dataset["root"])
    workbook = _find_ppg_bp_workbook(root)
    if workbook is None:
        return []
    frame = _read_ppg_bp_table(workbook)
    id_column = _find_column(frame.columns, ("subject_id", "subject", "id", "num"))
    sbp_column = _find_column(frame.columns, ("systolic", "sbp"))
    dbp_column = _find_column(frame.columns, ("diastolic", "dbp"))
    hr_column = _find_column(frame.columns, ("heart_rate", "heart rate", "hr"))
    if id_column is None or sbp_column is None or dbp_column is None:
        raise ValueError(f"{workbook}: cannot identify subject, SBP, and DBP columns")
    info: dict[str, pd.Series] = {}
    for _, row in frame.iterrows():
        raw_id = row[id_column]
        if pd.isna(raw_id):
            continue
        numeric = _number(raw_id)
        key = str(int(numeric)) if numeric is not None and numeric.is_integer() else str(raw_id).strip()
        info[key] = row

    recordings: list[Recording] = []
    for path in sorted(root.rglob("*.txt")):
        if "subject" not in str(path.parent).lower() or "data" in path.stem.lower():
            continue
        match = re.match(r"(\d+)[_-](\d+)", path.stem)
        if not match:
            continue
        participant, repetition = match.groups()
        row = info.get(participant)
        if row is None:
            continue
        sbp = _number(row[sbp_column])
        dbp = _number(row[dbp_column])
        if sbp is None or dbp is None:
            continue
        recordings.append(
            Recording(
                dataset_id="ppg_bp",
                participant_id=f"PPGBP{int(participant):03d}",
                session_id="single_cuff_occasion",
                recording_id=f"segment_{repetition}",
                label_group_id=f"ppg_bp:{participant}",
                chronological_order=repetition,
                sensor_site=dataset.get("sensor_site", "fingertip"),
                sample_rate_hz=1000.0,
                ppg_path=str(path.resolve()),
                imu_path=None,
                sbp=sbp,
                dbp=dbp,
                reference_hr=_number(row[hr_column]) if hr_column is not None else None,
                label_source="subject_cuff",
                label_timing="before_ppg",
                quality_status="dataset_screened",
                source_format="ppg_bp_txt",
                metadata_path=str(workbook.resolve()),
            )
        )
    return recordings


def discover_recordings(config: dict[str, Any]) -> tuple[list[Recording], list[dict[str, Any]]]:
    recordings: list[Recording] = []
    statuses: list[dict[str, Any]] = []
    adapters = {
        "one_month_wrist": _discover_one_month,
        "ppg_bp": _discover_ppg_bp,
        "local_upper_arm": _discover_local,
    }
    for dataset_id, adapter in adapters.items():
        dataset = config["datasets"].get(dataset_id, {})
        if not dataset.get("enabled", False):
            statuses.append({"dataset_id": dataset_id, "status": "disabled", "recording_count": 0})
            continue
        root_key = "root" if "root" in dataset else "raw_dir"
        root = resolve_path(dataset[root_key])
        if not root.exists():
            statuses.append(
                {
                    "dataset_id": dataset_id,
                    "status": "not_installed" if dataset.get("optional", False) else "missing_required",
                    "recording_count": 0,
                    "path": str(root),
                }
            )
            continue
        try:
            found = adapter(dataset)
            recordings.extend(found)
            statuses.append(
                {"dataset_id": dataset_id, "status": "available", "recording_count": len(found), "path": str(root)}
            )
        except Exception as exc:
            statuses.append(
                {"dataset_id": dataset_id, "status": "error", "recording_count": 0, "path": str(root), "error": str(exc)}
            )
    _mark_calibration_occasions(recordings)
    return recordings, statuses


def _mark_calibration_occasions(recordings: list[Recording]) -> None:
    grouped: dict[tuple[str, str], list[Recording]] = {}
    for recording in recordings:
        grouped.setdefault((recording.dataset_id, recording.participant_id), []).append(recording)
    for participant_recordings in grouped.values():
        label_groups: dict[str, list[Recording]] = {}
        for recording in participant_recordings:
            label_groups.setdefault(recording.label_group_id, []).append(recording)
        first_group = min(label_groups.values(), key=lambda group: min(row.chronological_order for row in group))
        for recording in first_group:
            recording.calibration_occasion = True


def load_recording(recording: Recording) -> SignalData:
    path = Path(recording.ppg_path)
    metadata: dict[str, Any] = {}
    if recording.metadata_path and recording.source_format == "local_ppg_csv":
        metadata = json.loads(Path(recording.metadata_path).read_text(encoding="utf-8"))

    if recording.source_format == "one_month_raw_csv":
        frame = pd.read_csv(path, usecols=ONE_MONTH_REQUIRED_COLUMNS)
        time_s = pd.to_numeric(frame["Time [sec]"], errors="coerce").to_numpy(dtype=float)
        acceleration = frame[
            ["Acceleration X [m/sec2]", "Acceleration Y [m/sec2]", "Acceleration Z [m/sec2]"]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return SignalData(
            time_s=time_s,
            ir=pd.to_numeric(frame["PPG IR [-]"], errors="coerce").to_numpy(dtype=float),
            red=pd.to_numeric(frame["PPG Red [-]"], errors="coerce").to_numpy(dtype=float),
            acceleration_m_s2=acceleration,
            sample_sequence=np.arange(len(frame), dtype=float),
            metadata=metadata,
        )

    if recording.source_format == "local_ppg_csv":
        frame = pd.read_csv(path)
        missing = [column for column in LOCAL_PPG_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        timestamp = pd.to_numeric(frame["timestamp_ms"], errors="coerce").to_numpy(dtype=float)
        if len(timestamp):
            metadata["bp_pipeline_first_timestamp_ms"] = float(timestamp[0])
        acceleration = None
        if recording.imu_path and Path(recording.imu_path).exists():
            imu = pd.read_csv(recording.imu_path)
            if all(column in imu.columns for column in LOCAL_IMU_COLUMNS):
                imu_time = pd.to_numeric(imu["timestamp_ms"], errors="coerce").to_numpy(dtype=float)
                axes = imu[["x_raw", "y_raw", "z_raw"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                axes *= float(metadata.get("imu_scale_g_per_lsb", 0.0039)) * 9.80665
                acceleration = np.column_stack(
                    [np.interp(timestamp, imu_time, axes[:, axis], left=np.nan, right=np.nan) for axis in range(3)]
                )
        return SignalData(
            time_s=(timestamp - timestamp[0]) / 1000.0 if len(timestamp) else timestamp,
            ir=pd.to_numeric(frame["ir"], errors="coerce").to_numpy(dtype=float),
            red=pd.to_numeric(frame["red"], errors="coerce").to_numpy(dtype=float),
            acceleration_m_s2=acceleration,
            sample_sequence=pd.to_numeric(frame["sample_seq"], errors="coerce").to_numpy(dtype=float),
            metadata=metadata,
        )

    if recording.source_format == "ppg_bp_txt":
        values = np.loadtxt(path, dtype=float).reshape(-1)
        return SignalData(
            time_s=np.arange(len(values), dtype=float) / 1000.0,
            ir=values,
            red=None,
            acceleration_m_s2=None,
            sample_sequence=np.arange(len(values), dtype=float),
            metadata=metadata,
        )
    raise ValueError(f"Unsupported source format: {recording.source_format}")


def audit_datasets(config: dict[str, Any], recordings: list[Recording], statuses: list[dict[str, Any]]) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "datasets": [],
        "processed_schema_checks": {},
        "candidate_datasets": [
            {"name": "MIMIC-BP", "role": "optional_stage_2_population_pretraining", "required": False},
            {"name": "Aurora-BP", "role": "restricted_wrist_candidate", "required": False},
            {"name": "CUHK-BP", "role": "request_only_upper_arm_candidate", "required": False},
            {"name": "Pulse Transit Time PPG", "role": "optional_motion_quality_testing", "required": False},
        ],
    }
    by_dataset: dict[str, list[Recording]] = {}
    for recording in recordings:
        by_dataset.setdefault(recording.dataset_id, []).append(recording)
    for status in statuses:
        rows = by_dataset.get(status["dataset_id"], [])
        item = dict(status)
        dataset_config = config["datasets"].get(status["dataset_id"], {})
        item["source_url"] = dataset_config.get("source_url")
        item["licence"] = dataset_config.get("licence")
        if rows:
            item.update(
                {
                    "participant_count": len({row.participant_id for row in rows}),
                    "label_group_count": len({row.label_group_id for row in rows}),
                    "sbp_min": min(row.sbp for row in rows),
                    "sbp_max": max(row.sbp for row in rows),
                    "dbp_min": min(row.dbp for row in rows),
                    "dbp_max": max(row.dbp for row in rows),
                    "sample_rate_hz_median": float(np.median([row.sample_rate_hz for row in rows if row.sample_rate_hz])),
                    "sensor_sites": sorted({row.sensor_site for row in rows}),
                }
            )
        if status["dataset_id"] == "one_month_wrist" and status.get("status") == "available":
            root = resolve_path(config["datasets"]["one_month_wrist"]["root"])
            item["source_commit"] = _git_commit(root)
            processed = next(root.rglob("processed.csv"), None)
            processed_feature = next(root.rglob("processed_feature.csv"), None)
            checks: dict[str, Any] = {}
            for name, path in (("processed.csv", processed), ("processed_feature.csv", processed_feature)):
                if path is not None:
                    with path.open(encoding="utf-8") as handle:
                        first_line = handle.readline().strip()
                    checks[name] = {
                        "column_count": len(first_line.split(",")),
                        "has_header": any(character.isalpha() for character in first_line),
                        "training_allowed": False,
                    }
            if processed is not None and rows:
                with processed.open(encoding="utf-8") as handle:
                    values = np.fromstring(handle.readline(), sep=",")
                matching_raw = next(processed.parent.glob("ESP*(PPG-IMU).csv"), None)
                source = pd.read_csv(matching_raw, nrows=1) if matching_raw is not None else pd.DataFrame()
                checks["processed_label_order"] = {
                    "observed_last_three": values[-3:].tolist() if len(values) >= 3 else [],
                    "raw_hr_sbp_dbp": [
                        _number(source.iloc[0]["HR [bpm]"]) if not source.empty else None,
                        _number(source.iloc[0]["SBP [mmHg]"]) if not source.empty else None,
                        _number(source.iloc[0]["DBP [mmHg]"]) if not source.empty else None,
                    ],
                    "matches_hr_sbp_dbp": bool(
                        not source.empty
                        and len(values) >= 3
                        and np.allclose(
                            values[-3:],
                            [source.iloc[0]["HR [bpm]"], source.iloc[0]["SBP [mmHg]"], source.iloc[0]["DBP [mmHg]"]],
                        )
                    ),
                    "note": "Headerless processed files are audit-only; raw labelled columns are authoritative.",
                }
            audit["processed_schema_checks"] = checks
        if status["dataset_id"] == "ppg_bp" and status.get("status") == "available":
            root = resolve_path(config["datasets"]["ppg_bp"]["root"])
            archive = next(root.glob("*.zip"), None)
            item["source_version"] = dataset_config.get("source_version")
            item["expected_archive_md5"] = dataset_config.get("expected_archive_md5")
            if archive is not None:
                item["archive_sha256"] = sha256_file(archive)
        audit["datasets"].append(item)
    return audit
