"""Merge verified CBM3D worker reports for the requested shot/seed grid.

Uses the same CBM3D_OUTPUT_ROOT, CBM3D_METHODS, CBM3D_SHOTS and CBM3D_SEEDS
settings as run_expc_cbm3d_unknown.py (seed 43 by default).
"""
import json
import os
from pathlib import Path

from fewshot_protocol import METHODS, SHOTS, parse_list, write_reports
from run_diffpir_transfer_experiments import ROOT


def main():
    output = Path(os.getenv("CBM3D_OUTPUT_ROOT", str(ROOT / "experiment_runs/expc_cbm3d_unknown")))
    methods = parse_list("CBM3D_METHODS", METHODS)
    shots = [int(value) for value in parse_list("CBM3D_SHOTS", SHOTS)]
    seeds = [int(value) for value in parse_list("CBM3D_SEEDS", [43])]
    paths = sorted(output.glob("reports_worker*/raw_rows.json"))
    if not paths:
        paths = [output / "reports/raw_rows.json"]
    rows_by_key = {}
    common_contract = None
    for path in paths:
        contract = json.loads(path.with_name("manifest.json").read_text())
        comparable = {key: value for key, value in contract.items() if key != "rank"}
        if common_contract is None:
            common_contract = comparable
        elif comparable != common_contract:
            raise RuntimeError("Worker reports have different run contracts.")
        for row in json.loads(path.read_text()):
            key = row["method"], int(row["shot"]), int(row["seed"])
            if key in rows_by_key:
                raise RuntimeError(f"Duplicate CBM3D cell across reports: {key}")
            rows_by_key[key] = row
    expected = {(method, shot, seed) for method in methods for shot in shots for seed in seeds}
    missing = expected - set(rows_by_key)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested CBM3D cells: {sorted(missing)[:10]}")
    rows = [rows_by_key[key] for key in sorted(expected)]
    write_reports(output / "reports", rows, "cbm3d")
    (output / "reports/aggregation_manifest.json").write_text(json.dumps({
        "workers": [str(path) for path in paths], "methods": methods, "shots": shots,
        "seeds": seeds, "common_contract": common_contract,
    }, indent=2) + "\n")
    print(f"Merged {len(rows)} CBM3D cells -> {output / 'reports'}")


if __name__ == "__main__":
    main()
