"""Verify and unpack the 17 training and evaluation label banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def verify(path, entry):
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT, help="Parent directory for dataset_labels/")
    parser.add_argument("--check", action="store_true", help="Verify already extracted files without unpacking")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/label_banks.json").read_text())
    banks = manifest["banks"]
    if not args.check:
        archive = ROOT / manifest["archive"]["path"]
        verify(archive, manifest["archive"])
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            expected = [bank["path"] for bank in banks]
            if [member.name for member in members] != expected or not all(member.isfile() for member in members):
                raise RuntimeError("Archive contents do not match the 17-bank manifest")
            for bank, member in zip(banks, members):
                path = args.output_root / bank["path"]
                if path.exists():
                    verify(path, bank)
                    continue
                raw = tar.extractfile(member).read()
                if len(raw) != bank["bytes"] or hashlib.sha256(raw).hexdigest() != bank["sha256"]:
                    raise RuntimeError(f"Checksum mismatch in archive: {bank['path']}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    for bank in banks:
        verify(args.output_root / bank["path"], bank)
    print(f"Verified {len(banks)} label banks ({sum(bank['record_count'] for bank in banks):,} records) in {args.output_root.resolve() / 'dataset_labels'}")


if __name__ == "__main__":
    main()
