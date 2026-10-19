"""
Command line entry point.

    python -m sarflood download --config config/arakawa_2022.yaml
    python -m sarflood run      --config config/arakawa_2022.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sarflood", description="Sentinel-1 flood mapping")
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--overwrite", action="store_true", help="re-download cached inputs")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.command == "download":
        from .acquire import download_all
        from .grid import analysis_grid

        ref = cfg["reference"]
        grid = analysis_grid(ref["gsi_flood"], cfg["grid"]["crs"], cfg["grid"]["resolution_m"], ref["aoi_buffer_m"])
        download_all(cfg, grid, overwrite=args.overwrite)
    else:
        from .pipeline import run

        table = run(cfg)
        cols = ["iou", "precision", "recall", "recall_cropland", "recall_built_up", "raw_candidates_iou"]
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(table[[c for c in cols if c in table.columns]])
        print(f"\nOutputs written to {cfg['paths']['outputs']}")


if __name__ == "__main__":
    main()
