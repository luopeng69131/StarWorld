"""Prepare causal prediction windows from official StarNet measurement logs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Iterator
import zipfile

import numpy as np

DATASET_URL = "https://github.com/ConnectedSystemsLab/StarNet"
DATASET_DOI = "https://doi.org/10.1145/3768971"
LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.\d+)?\s+TP:\s*([-+\d.eE]+)")
SEGMENT_FIELDS = ("trace_id", "source_run_id", "region", "duration_s", "start_time")
WINDOW_FIELDS = ("trace_id", "source_run_id", "window_start_s", "window_end_s",
                 "history_mean_mbps", "history_p05_mbps", "history_p25_mbps", "history_std_mbps",
                 "point_prediction_mbps", "heuristic_safe_capacity_mbps", "future_mean_mbps",
                 "future_p05_mbps", "future_min_mbps", "overestimation_label", "high_risk_label")


@dataclass(frozen=True)
class MeasurementRun:
    region: str
    name: str
    text: str


def region_from_name(name: str) -> str:
    value = name.lower()
    if any(word in value for word in ("victoria", "canada", "_vic")):
        return "VIC"
    if any(word in value for word in ("osn", "germany")):
        return "OSN"
    if any(word in value for word in ("chicago", "_us", "usa")):
        return "US"
    raise ValueError(f"cannot infer region for {name}; supply --region")


def iter_measurement_runs(root: Path, region: str | None = None) -> Iterator[MeasurementRun]:
    """Read extracted official logs and logs inside ZIP archives."""
    for archive in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            for name in sorted(handle.namelist()):
                if Path(name).name != "network_measurements.log":
                    continue
                label = f"{archive.relative_to(root)}/{name}"
                yield MeasurementRun(region or region_from_name(label), label,
                                     handle.read(name).decode("utf-8", errors="replace"))
    for path in sorted(root.rglob("network_measurements.log")):
        label = str(path.relative_to(root))
        yield MeasurementRun(region or region_from_name(label), label,
                             path.read_text(encoding="utf-8", errors="replace"))


def parse_throughput(text: str, max_gap_s: int = 2) -> list[tuple[datetime, np.ndarray]]:
    """Average repeated seconds; interpolate short interior gaps; split long gaps."""
    if max_gap_s < 1:
        raise ValueError("max_gap_s must be at least one")
    grouped: dict[datetime, list[float]] = {}
    for line in text.splitlines():
        match = LOG_LINE.match(line.strip())
        if match:
            value = float(match[2])
            if np.isfinite(value) and value >= 0:
                instant = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S")
                grouped.setdefault(instant, []).append(value)
    blocks: list[tuple[datetime, np.ndarray]] = []
    values: list[float] = []
    start = previous = None
    for instant in sorted(grouped):
        value = float(np.mean(grouped[instant]))
        gap = int((instant - previous).total_seconds()) if previous else 0
        if previous is not None and gap > max_gap_s:
            blocks.append((start, np.asarray(values, dtype=np.float32)))
            values, start = [], None
        elif previous is not None and gap > 1:
            left = values[-1]
            values.extend(left + (value - left) * i / gap for i in range(1, gap))
        if start is None:
            start = instant
        values.append(value)
        previous = instant
    if values:
        blocks.append((start, np.asarray(values, dtype=np.float32)))
    return blocks


def causal_windows(values: np.ndarray, trace_id: str, run_id: str,
                   history_s: int = 60, step_s: int = 10) -> list[dict]:
    """Derive features from [t-history,t), and labels separately from [t,t+step)."""
    rows = []
    for t in range(history_s, len(values) - step_s + 1, step_s):
        past, future = values[t - history_s:t], values[t:t + step_s]
        point, safe = float(past.mean()), float(np.quantile(past, .1))
        rows.append(dict(zip(WINDOW_FIELDS, (
            trace_id, run_id, t, t + step_s, point, float(np.quantile(past, .05)),
            float(np.quantile(past, .25)), float(past.std()), point, safe,
            float(future.mean()), float(np.quantile(future, .05)), float(future.min()),
            int(point > 1.25 * max(float(future.mean()), 1e-3)),
            int(np.quantile(future, .05) < 10),
        ))))
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_starnet(input_root: Path, output: Path, *, region: str | None = None,
                    segment_s: int = 600, history_s: int = 60, step_s: int = 10,
                    max_gap_s: int = 2, max_runs: int | None = None,
                    replay_index: Path | None = None) -> dict:
    if not input_root.is_dir():
        raise ValueError("input_root must be a directory containing StarNet logs or ZIPs")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")
    if step_s != 10 or segment_s <= history_s + step_s or history_s < 1:
        raise ValueError("use 10-s states and a segment longer than history plus one step")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    segments, windows, provenance, seen = [], [], [], set()
    for run in iter_measurement_runs(input_root, region):
        digest = hashlib.sha256(run.text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        run_id = f"{run.region}_{digest[:16]}"
        provenance.append({"source_run_id": run_id, "source": run.name, "sha256": digest})
        number = 0
        for start, block in parse_throughput(run.text, max_gap_s):
            for offset in range(0, len(block) - segment_s + 1, segment_s):
                trace_id = f"{run_id}_segment_{number:05d}"
                number += 1
                segments.append(dict(zip(SEGMENT_FIELDS, (
                    trace_id, run_id, run.region, segment_s,
                    str(start + timedelta(seconds=offset)),
                ))))
                windows.extend(causal_windows(block[offset:offset + segment_s], trace_id,
                                              run_id, history_s, step_s))
        if max_runs is not None and len(provenance) >= max_runs:
            break
    if not segments:
        raise ValueError("no complete measurement segments; check the input format and segment length")
    branches = []
    if replay_index is not None:
        with replay_index.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                log = Path(row["log_path"])
                log = log if log.is_absolute() else replay_index.resolve().parent / log
                if not log.is_file():
                    raise ValueError(f"replay log does not exist: {log}")
                branches.append({"trace_id": row["trace_id"], "controller": row["controller"],
                                 "log_path": str(log.resolve())})
    _write_csv(output / "segments.csv", SEGMENT_FIELDS, segments)
    _write_csv(output / "prediction_windows.csv", WINDOW_FIELDS, windows)
    _write_csv(output / "replay_index.csv", ("trace_id", "controller", "log_path"), branches)
    config = {"data": {"segments_csv": "segments.csv", "prediction_csv": "prediction_windows.csv",
                       "abr_outcomes_csv": "replay_index.csv"}, "device": "cpu"}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"dataset": DATASET_URL, "dataset_paper": DATASET_DOI,
              "segments": len(segments), "prediction_windows": len(windows),
              "replay_branches": len(branches), "state_interval_s": step_s,
              "segment_s": segment_s, "history_s": history_s, "runs": provenance,
              "point_predictor": "causal historical mean", "safe_capacity": "causal historical 0.1 quantile"}
    (output / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
