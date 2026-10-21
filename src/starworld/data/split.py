from __future__ import annotations

from collections.abc import Iterable
import math
import re

import numpy as np

from starworld.schemas import Episode, SplitManifest


def grouped_source_run_split(
    episodes: Iterable[Episode],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> SplitManifest:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be below one")

    episode_list = list(episodes)
    groups = sorted({episode.source_run_id for episode in episode_list})
    if len(groups) < 3:
        raise ValueError("at least three source runs are required for grouped splitting")
    shuffled = np.asarray(groups, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = max(1, int(round(len(shuffled) * train_fraction)))
    validation_end = max(train_end + 1, int(round(len(shuffled) * (train_fraction + validation_fraction))))
    validation_end = min(validation_end, len(shuffled) - 1)
    split_by_group = {
        group: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, group in enumerate(shuffled.tolist())
    }
    ids = {"train": [], "validation": [], "test": []}
    for episode in episode_list:
        ids[split_by_group[episode.source_run_id]].append(episode.episode_id)
    return SplitManifest(
        split_by_source_run=split_by_group,
        train_ids=ids["train"],
        validation_ids=ids["validation"],
        test_ids=ids["test"],
        excluded_ids=[],
        protocol="source_run_random",
    )


def region_trace_balanced_source_run_split(
    episodes: Iterable[Episode],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> SplitManifest:
    """Keep source runs disjoint while balancing regions and trace counts.

    Source runs in this dataset differ from one to hundreds of traces. This
    protocol first fixes region-stratified source-run quotas, then balances
    trace counts within those quotas.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction <= 0:
        raise ValueError("train + validation fractions must be below one")
    episode_list = list(episodes)
    network_only = [
        episode
        for episode in episode_list
        if episode.controller == "NetworkOnly"
    ]
    count_source = network_only if network_only else episode_list
    traces_by_group: dict[str, set[str]] = {}
    region_by_group: dict[str, str] = {}
    for episode in count_source:
        traces_by_group.setdefault(episode.source_run_id, set()).add(
            episode.trace_id
        )
        region_by_group[episode.source_run_id] = episode.region
    if len(traces_by_group) < 3:
        raise ValueError("at least three source runs are required")

    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for region in sorted(set(region_by_group.values())):
        groups = [
            group
            for group, value in region_by_group.items()
            if value == region
        ]
        if len(groups) < 3:
            raise ValueError(
                f"region {region} needs at least three source runs"
            )
        tie_break = {group: float(rng.random()) for group in groups}
        groups.sort(
            key=lambda group: (
                -len(traces_by_group[group]),
                tie_break[group],
            )
        )
        total = float(sum(len(traces_by_group[group]) for group in groups))
        targets = {name: total * fraction for name, fraction in fractions.items()}
        raw_group_targets = {
            name: len(groups) * fraction
            for name, fraction in fractions.items()
        }
        group_targets = {
            name: max(1, int(math.floor(target)))
            for name, target in raw_group_targets.items()
        }
        while sum(group_targets.values()) < len(groups):
            destination = max(
                fractions,
                key=lambda name: (
                    raw_group_targets[name] - group_targets[name],
                    fractions[name],
                    name,
                ),
            )
            group_targets[destination] += 1
        while sum(group_targets.values()) > len(groups):
            destination = max(
                (
                    name
                    for name in fractions
                    if group_targets[name] > 1
                ),
                key=lambda name: (
                    group_targets[name] - raw_group_targets[name],
                    -fractions[name],
                    name,
                ),
            )
            group_targets[destination] -= 1
        counts = {name: 0.0 for name in fractions}
        group_counts = {name: 0 for name in fractions}
        for group in groups:
            weight = float(len(traces_by_group[group]))
            choices = [
                name
                for name in fractions
                if group_counts[name] < group_targets[name]
            ]

            def objective(destination: str) -> float:
                candidate = dict(counts)
                candidate[destination] += weight
                trace_error = sum(
                    (
                        (
                            candidate[name]
                            - targets[name]
                        )
                        / max(targets[name], 1.0)
                    )
                    ** 2
                    for name in fractions
                )
                return float(trace_error)

            destination = min(
                choices,
                key=lambda name: (
                    objective(name),
                    counts[name] / max(targets[name], 1.0),
                    name,
                ),
            )
            assignment[group] = destination
            counts[destination] += weight
            group_counts[destination] += 1

    manifest = _manifest_from_assignments(episode_list, assignment)
    manifest.protocol = "source_run_region_trace_balanced"
    return manifest


def assert_no_group_leakage(episodes: Iterable[Episode], manifest: SplitManifest) -> None:
    assignment: dict[str, set[str]] = {}
    ids = {
        **{episode_id: "train" for episode_id in manifest.train_ids},
        **{episode_id: "validation" for episode_id in manifest.validation_ids},
        **{episode_id: "test" for episode_id in manifest.test_ids},
    }
    for episode in episodes:
        split = ids.get(episode.episode_id)
        if split is None:
            continue
        assignment.setdefault(episode.source_run_id, set()).add(split)
    leaking = {group: splits for group, splits in assignment.items() if len(splits) > 1}
    if leaking:
        raise AssertionError(f"source-run leakage detected: {leaking}")


def region_holdout_split(
    episodes: Iterable[Episode],
    test_region: str,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> SplitManifest:
    episode_list = list(episodes)
    region = test_region.lower()
    test_groups = {
        episode.source_run_id
        for episode in episode_list
        if episode.region.lower() == region
    }
    remaining = sorted({episode.source_run_id for episode in episode_list}.difference(test_groups))
    if not test_groups or len(remaining) < 2:
        raise ValueError(f"region holdout {test_region} does not leave viable train/validation groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    validation_count = max(1, int(round(len(remaining) * validation_fraction)))
    validation_groups = set(remaining[:validation_count])
    split_by_group = {
        group: (
            "test"
            if group in test_groups
            else "validation"
            if group in validation_groups
            else "train"
        )
        for group in sorted({episode.source_run_id for episode in episode_list})
    }
    manifest = _manifest_from_assignments(episode_list, split_by_group)
    manifest.protocol = "region_holdout"
    manifest.holdout_value = test_region
    return manifest


def profile_holdout_split(
    episodes: Iterable[Episode],
    profile_term: str,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> SplitManifest:
    episode_list = list(episodes)
    term = profile_term.lower()
    test_groups = {
        episode.source_run_id
        for episode in episode_list
        if term in str(episode.metadata.get("profile_labels", "")).lower()
    }
    remaining = sorted({episode.source_run_id for episode in episode_list}.difference(test_groups))
    if not test_groups or len(remaining) < 2:
        raise ValueError(f"profile holdout {profile_term} does not leave viable groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    validation_count = max(1, int(round(len(remaining) * validation_fraction)))
    validation_groups = set(remaining[:validation_count])
    split_by_group = {
        group: (
            "test"
            if group in test_groups
            else "validation"
            if group in validation_groups
            else "train"
        )
        for group in sorted({episode.source_run_id for episode in episode_list})
    }
    manifest = _manifest_from_assignments(episode_list, split_by_group)
    manifest.protocol = "profile_holdout"
    manifest.holdout_value = profile_term
    return manifest


def chronological_holdout_split(
    episodes: Iterable[Episode],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> SplitManifest:
    episode_list = list(episodes)
    def timestamp_key(value: str) -> tuple[int, int, int, int, int, str]:
        match = re.search(
            r"data_(\d{4})[-_](\d{2})[-_](\d{2})_(\d{1,2})_(\d{2})",
            value,
        )
        if not match:
            return (0, 0, 0, 0, 0, value)
        return (*[int(part) for part in match.groups()], value)

    groups = sorted({episode.source_run_id for episode in episode_list}, key=timestamp_key)
    if len(groups) < 3:
        raise ValueError("chronological holdout requires at least three source runs")
    test_count = max(1, int(round(len(groups) * test_fraction)))
    validation_count = max(1, int(round(len(groups) * validation_fraction)))
    if test_count + validation_count >= len(groups):
        validation_count = 1
        test_count = 1
    test_groups = set(groups[-test_count:])
    validation_groups = set(groups[-(test_count + validation_count) : -test_count])
    split_by_group = {
        group: (
            "test"
            if group in test_groups
            else "validation"
            if group in validation_groups
            else "train"
        )
        for group in groups
    }
    manifest = _manifest_from_assignments(episode_list, split_by_group)
    manifest.protocol = "chronological_holdout"
    return manifest


def action_combination_holdout_split(
    episodes: Iterable[Episode],
    heldout_controller: str,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> SplitManifest:
    """Hold out an action family without crossing source-run boundaries.

    A regular source-run split is constructed first. The held-out controller is
    removed from train/validation, while test retains that controller and the
    corresponding NetworkOnly measurement episodes. Other test-controller
    branches are excluded from this protocol. Thus action novelty and source
    novelty are evaluated together without exposing a test source run during
    training.
    """
    episode_list = list(episodes)
    controller = heldout_controller.lower()
    known = {
        episode.controller.lower()
        for episode in episode_list
        if episode.controller != "NetworkOnly"
    }
    if controller not in known:
        raise ValueError(
            f"unknown held-out controller {heldout_controller}; "
            f"available={sorted(known)}"
        )
    base = grouped_source_run_split(
        episode_list,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    assignment = base.split_by_source_run
    selected = {"train": [], "validation": [], "test": []}
    excluded: list[str] = []
    for episode in episode_list:
        split = assignment[episode.source_run_id]
        name = episode.controller.lower()
        keep = (
            (split in {"train", "validation"} and name != controller)
            or (
                split == "test"
                and (name == controller or episode.controller == "NetworkOnly")
            )
        )
        if keep:
            selected[split].append(episode.episode_id)
        else:
            excluded.append(episode.episode_id)
    if not selected["train"] or not selected["validation"] or not selected["test"]:
        raise ValueError(
            f"action holdout {heldout_controller} produced an empty split"
        )
    manifest = SplitManifest(
        split_by_source_run=assignment,
        train_ids=selected["train"],
        validation_ids=selected["validation"],
        test_ids=selected["test"],
        excluded_ids=excluded,
        protocol="action_combination_holdout",
        holdout_value=heldout_controller,
    )
    assert_no_group_leakage(episode_list, manifest)
    return manifest


def long_horizon_stress_split(
    episodes: Iterable[Episode],
    minimum_test_steps: int = 12,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 17,
) -> SplitManifest:
    """Hold out source runs with trajectories long enough for stress rollout.

    Source-run isolation is preserved first. Short branches belonging to a
    selected test source are excluded rather than moved to another split.
    Training and validation retain their natural episode-length distribution.
    """
    if minimum_test_steps < 2:
        raise ValueError("minimum_test_steps must be at least two")
    episode_list = list(episodes)
    groups = sorted({episode.source_run_id for episode in episode_list})
    if len(groups) < 3:
        raise ValueError("long-horizon stress split requires at least three source runs")
    maximum_transitions = {
        group: max(
            episode.length - 1
            for episode in episode_list
            if episode.source_run_id == group
        )
        for group in groups
    }
    eligible = [
        group
        for group in groups
        if maximum_transitions[group] >= minimum_test_steps
    ]
    if not eligible:
        raise ValueError(
            f"no source run has an episode of at least {minimum_test_steps} steps"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    test_count = min(
        len(eligible),
        max(1, int(round(len(groups) * test_fraction))),
    )
    test_groups = set(eligible[:test_count])
    remaining = [group for group in groups if group not in test_groups]
    if len(remaining) < 2:
        raise ValueError("long-horizon holdout does not leave train/validation groups")
    rng.shuffle(remaining)
    validation_count = min(
        len(remaining) - 1,
        max(1, int(round(len(groups) * validation_fraction))),
    )
    validation_groups = set(remaining[:validation_count])
    split_by_group = {
        group: (
            "test"
            if group in test_groups
            else "validation"
            if group in validation_groups
            else "train"
        )
        for group in groups
    }

    selected = {"train": [], "validation": [], "test": []}
    excluded: list[str] = []
    for episode in episode_list:
        split = split_by_group[episode.source_run_id]
        if (
            split == "test"
            and episode.length - 1 < minimum_test_steps
        ):
            excluded.append(episode.episode_id)
        else:
            selected[split].append(episode.episode_id)
    if not all(selected.values()):
        raise ValueError("long-horizon stress split produced an empty split")
    manifest = SplitManifest(
        split_by_source_run=split_by_group,
        train_ids=selected["train"],
        validation_ids=selected["validation"],
        test_ids=selected["test"],
        excluded_ids=excluded,
        protocol="long_horizon_stress",
        holdout_value=str(minimum_test_steps),
    )
    assert_no_group_leakage(episode_list, manifest)
    return manifest


def build_split_manifest(
    episodes: Iterable[Episode],
    protocol: str = "source_run_random",
    holdout_value: str | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> SplitManifest:
    episode_list = list(episodes)
    if protocol == "source_run_random":
        return grouped_source_run_split(
            episode_list,
            train_fraction,
            validation_fraction,
            seed,
        )
    if protocol == "source_run_region_trace_balanced":
        return region_trace_balanced_source_run_split(
            episode_list,
            train_fraction,
            validation_fraction,
            seed,
        )
    if protocol == "region_holdout":
        if not holdout_value:
            raise ValueError("region_holdout requires holdout_value")
        return region_holdout_split(
            episode_list,
            holdout_value,
            validation_fraction,
            seed,
        )
    if protocol == "profile_holdout":
        if not holdout_value:
            raise ValueError("profile_holdout requires holdout_value")
        return profile_holdout_split(
            episode_list,
            holdout_value,
            validation_fraction,
            seed,
        )
    if protocol == "chronological_holdout":
        return chronological_holdout_split(
            episode_list,
            validation_fraction=validation_fraction,
            test_fraction=max(1.0 - train_fraction - validation_fraction, 1e-6),
        )
    if protocol == "action_combination_holdout":
        if not holdout_value:
            raise ValueError(
                "action_combination_holdout requires a controller holdout_value"
            )
        return action_combination_holdout_split(
            episode_list,
            holdout_value,
            train_fraction,
            validation_fraction,
            seed,
        )
    if protocol == "long_horizon_stress":
        minimum_steps = 12 if holdout_value is None else int(holdout_value)
        return long_horizon_stress_split(
            episode_list,
            minimum_test_steps=minimum_steps,
            validation_fraction=validation_fraction,
            test_fraction=max(
                1.0 - train_fraction - validation_fraction,
                1e-6,
            ),
            seed=seed,
        )
    raise ValueError(f"unknown split protocol {protocol}")


def _manifest_from_assignments(
    episodes: list[Episode],
    split_by_group: dict[str, str],
) -> SplitManifest:
    ids = {"train": [], "validation": [], "test": []}
    for episode in episodes:
        ids[split_by_group[episode.source_run_id]].append(episode.episode_id)
    manifest = SplitManifest(
        split_by_source_run=split_by_group,
        train_ids=ids["train"],
        validation_ids=ids["validation"],
        test_ids=ids["test"],
        excluded_ids=[],
    )
    assert_no_group_leakage(episodes, manifest)
    return manifest
