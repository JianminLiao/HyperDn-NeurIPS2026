"""Print or execute the WED/PSNR DiffPIR and CBM3D oracle searches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_recipes():
    return json.loads((ROOT / "configs/oracles.json").read_text())["banks"]


def commands_for(bank, output_root):
    output = Path(output_root) / Path(bank["output"]).relative_to("dataset_labels")
    if not output.is_absolute():
        output = ROOT / output
    env = dict(bank["env"])
    env["SAVE_DIR"] = str(output.parent)
    env["RUN_NAME"] = output.stem
    command = [sys.executable, str(ROOT / bank["generator"])]
    if bank["group"] == "cbm3d":
        env["RUN_NAME"] = output.stem.removesuffix("_merged")
        return [(command, {**env, "RUN_MODE": mode}) for mode in ("search", "merge")]
    return [(command, env)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("diffpir", "cbm3d", "all"), default="all")
    parser.add_argument("--bank", help="Run a single ID, e.g. diffpir_train or cbm3d_val")
    parser.add_argument("--list", action="store_true", help="Show IDs, counts, and required output paths")
    parser.add_argument("--output-root", default="dataset_labels_regenerated", help="Directory for newly generated labels")
    parser.add_argument("--execute", action="store_true", help="Actually run the expensive searches; default only prints commands")
    args = parser.parse_args()
    banks = [bank for bank in load_recipes() if (args.group == "all" or bank["group"] == args.group) and (not args.bank or bank["id"] == args.bank)]
    if not banks:
        parser.error("No bank matches this selection")
    if args.list:
        for bank in banks:
            print(f"{bank['id']:15s} {bank['record_count']:6d} records  {bank['output']}")
        return
    for bank in banks:
        print(f"\n# {bank['id']}: {bank['record_count']} records", flush=True)
        for command, env in commands_for(bank, args.output_root):
            print(shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *command]), flush=True)
            if args.execute:
                Path(env["SAVE_DIR"]).mkdir(parents=True, exist_ok=True)
                subprocess.run(command, cwd=ROOT, env={**os.environ, **env}, check=True)
    if not args.execute:
        print("\nDry run only. Add --execute to launch searches after preparing the data and solver dependencies.")


if __name__ == "__main__":
    main()
