"""Load the YAML run configuration and resolve paths relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Read a YAML config file and turn relative paths into absolute ones.

    Paths in the config are written relative to the project root (the folder
    containing ``config/``), so the pipeline runs the same from any working
    directory.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file, e.g. ``config/arakawa_2022.yaml``.

    Returns
    -------
    dict
        Parsed configuration with an extra ``root`` key and absolute paths.
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = path.parent.parent
    cfg["root"] = root
    for key in ("gsi_flood", "osm_water"):
        cfg["reference"][key] = root / cfg["reference"][key]
    for key in ("raw", "outputs"):
        cfg["paths"][key] = root / cfg["paths"][key]

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Fail early with a clear message instead of deep inside the pipeline."""
    det = cfg["detection"]
    if det["speckle_filter"] not in ("boxcar", "lee"):
        raise ValueError("detection.speckle_filter must be 'boxcar' or 'lee'")
    if det["ratio_threshold"] <= 1.0:
        raise ValueError("detection.ratio_threshold must be > 1 (before/after ratio)")
    if cfg["grid"]["resolution_m"] <= 0:
        raise ValueError("grid.resolution_m must be positive")
    for pol in cfg["sar"]["polarizations"]:
        if pol not in ("VV", "VH"):
            raise ValueError(f"Unsupported polarization: {pol}")
