from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
else:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None  # type: ignore[assignment]


REQUIRED_SEGMENT_COLUMNS = {"trace_id", "source_run_id", "region", "duration_s"}
REQUIRED_PREDICTION_COLUMNS = {
    "trace_id",
    "source_run_id",
    "window_start_s",
    "window_end_s",
    "history_mean_mbps",
    "history_p05_mbps",
    "point_prediction_mbps",
    "future_mean_mbps",
}
REQUIRED_ABR_COLUMNS = {"trace_id", "controller", "log_path"}


def _require_pandas() -> Any:
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required to build a dataset from CSV source tables; "
            "install pandas or load an existing episode archive"
        )
    return pd


def _load_checked(path: str | Path, required: set[str], label: str) -> pd.DataFrame:
    pandas = _require_pandas()
    frame = pandas.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} table is missing columns: {sorted(missing)}")
    return frame


def load_source_tables(
    segments_csv: str | Path,
    prediction_csv: str | Path,
    abr_outcomes_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segments = _load_checked(segments_csv, REQUIRED_SEGMENT_COLUMNS, "segment")
    prediction = _load_checked(prediction_csv, REQUIRED_PREDICTION_COLUMNS, "prediction")
    abr = _load_checked(abr_outcomes_csv, REQUIRED_ABR_COLUMNS, "ABR outcome")
    return segments, prediction, abr
