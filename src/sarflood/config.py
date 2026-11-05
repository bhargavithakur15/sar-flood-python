"""Load the YAML run configuration and resolve paths relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """
    Read a YAML config file and turn relative paths into absolute ones.

    Paths in the config are written relative to the project root (the folder
    containing ``config/``), so the pipeline runs the same from any working
    directory.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file, e.g. ``config/arakawa_2022.yaml``.
    overrides : list of str, optional
        ``"section.key=value"`` strings applied on top of the file for this
        run only, e.g. ``["masks.slope_max_percent=5", "paths.outputs=outputs/slope5"]``.
        The YAML file itself is never changed.

    Returns
    -------
    dict
        Parsed configuration with an extra ``root`` key and absolute paths.
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        apply_override(cfg, item)

    root = path.parent.parent
    cfg["root"] = root
    for key in ("gsi_flood", "osm_water"):
        cfg["reference"][key] = root / cfg["reference"][key]
    for key in ("raw", "outputs"):
        cfg["paths"][key] = root / cfg["paths"][key]

    _validate(cfg)
    return cfg


def apply_override(cfg: dict[str, Any], item: str) -> None:
    """
    Set one nested value from a ``"section.key=value"`` string. The value is
    parsed as YAML, so ``5`` becomes an int, ``null`` becomes None and
    ``[VH]`` a list. Only keys that already exist can be overridden, so a typo
    fails loudly instead of being silently ignored.
    """
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    dotted, raw_value = item.split("=", 1)
    keys = dotted.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Unknown config section in override: {dotted}")
        node = node[key]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise KeyError(f"Unknown config key in override: {dotted}")
    node[keys[-1]] = yaml.safe_load(raw_value)


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
