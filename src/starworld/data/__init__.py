from .archive import load_episode_archive
from .split import (
    chronological_holdout_split,
    grouped_source_run_split,
    profile_holdout_split,
    region_holdout_split,
)


def __getattr__(name: str):
    if name == "CrossLayerExperienceConstructor":
        from .episodes import CrossLayerExperienceConstructor

        return CrossLayerExperienceConstructor
    raise AttributeError(name)


__all__ = [
    "CrossLayerExperienceConstructor",
    "chronological_holdout_split",
    "grouped_source_run_split",
    "load_episode_archive",
    "profile_holdout_split",
    "region_holdout_split",
]
