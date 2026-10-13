"""Run the per-query cutoff experiment. Caches let a restart resume."""
from __future__ import annotations

import time

import limits  # noqa: F401

from collect import collect_dev, collect_fit
from develop import score_development
from experiment import experiment_settings
from limits import SETTINGS_PATH, guard_memory, log, write_json
from regress import select_and_fit
from report_text import write_findings


def main() -> None:
    started = time.perf_counter()
    write_json(SETTINGS_PATH, experiment_settings())
    collect_fit()
    guard_memory()
    collect_dev()
    guard_memory()
    chosen = select_and_fit()
    development = score_development(chosen)
    write_findings(chosen, development)
    log(f"done seconds={time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
