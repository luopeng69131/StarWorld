from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from starworld.schemas import Episode


def load_episode_archive(
    path: str | Path,
) -> tuple[list[Episode], dict[str, Any]]:
    """Load a built episode archive without dataset-construction dependencies."""
    root = Path(path)
    arrays = np.load(root / "episodes.npz")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    episodes: list[Episode] = []
    for row in index:
        prefix = row["prefix"]
        targets = {
            name: arrays[f"{prefix}_target_{name}"]
            for name in row["target_names"]
        }
        episode = Episode(
            episode_id=row["episode_id"],
            source_run_id=row["source_run_id"],
            trace_id=row["trace_id"],
            branch_id=row["branch_id"],
            region=row["region"],
            controller=row["controller"],
            time_s=arrays[f"{prefix}_time_s"],
            network=arrays[f"{prefix}_network"],
            prediction=arrays[f"{prefix}_prediction"],
            streaming=arrays[f"{prefix}_streaming"],
            action=arrays[f"{prefix}_action"],
            config=arrays[f"{prefix}_config"],
            context=arrays[f"{prefix}_context"],
            targets=targets,
            metadata=row["metadata"],
        )
        episode.validate()
        episodes.append(episode)
    audit_path = root / "audit.json"
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.exists()
        else {}
    )
    return episodes, audit
