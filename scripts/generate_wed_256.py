from pathlib import Path
import argparse
import json

from PIL import Image


def center_crop_256(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def load_split_stems(split_root: Path, split_name: str) -> list[str]:
    split_dir = split_root / split_name
    stems = sorted(path.stem for path in split_dir.glob("*.png"))
    if not stems:
        raise FileNotFoundError(f"No split members found in {split_dir}")
    return stems


def write_split(source_root: Path, output_root: Path, split_name: str, stems: list[str], crop_size: int) -> None:
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(stems, start=1):
        source_path = source_root / f"{stem}.bmp"
        target_path = split_dir / f"{stem}.png"
        with Image.open(source_path) as image:
            cropped = center_crop_256(image.convert("RGB"), crop_size)
            cropped.save(target_path)
        if index % 500 == 0 or index == len(stems):
            print(f"[{split_name}] wrote {index}/{len(stems)}")


def validate_output(source_root: Path, output_root: Path, split_map: dict[str, list[str]], crop_size: int) -> None:
    all_source_stems = sorted(path.stem for path in source_root.glob("*.bmp"))
    all_output_stems = []
    for split_name, stems in split_map.items():
        split_dir = output_root / split_name
        output_stems = sorted(path.stem for path in split_dir.glob("*.png"))
        if output_stems != stems:
            raise RuntimeError(f"Split membership mismatch for {split_name}")
        for sample_path in sorted(split_dir.glob("*.png"))[:3]:
            with Image.open(sample_path) as image:
                if image.size != (crop_size, crop_size):
                    raise RuntimeError(f"Unexpected size for {sample_path}: {image.size}")
        all_output_stems.extend(output_stems)
        print(f"validated {split_name}: {len(output_stems)} images")
    if sorted(all_output_stems) != all_source_stems:
        raise RuntimeError("Union of WED_256 stems does not match raw WED stems")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data/WED_256 from raw WED BMP files.")
    parser.add_argument("--source-root", default="data/WED")
    parser.add_argument("--split-root", default=None, help="Optional existing train/val PNG tree; otherwise use the frozen filename manifest.")
    parser.add_argument("--split-manifest", default=str(Path(__file__).resolve().parents[1] / "configs/wed_split.json"))
    parser.add_argument("--output-root", default="data/WED_256")
    parser.add_argument("--crop-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    split_root = Path(args.split_root) if args.split_root else None
    output_root = Path(args.output_root)
    crop_size = int(args.crop_size)

    if split_root is not None:
        split_map = {name: load_split_stems(split_root, name) for name in ("train", "val")}
    else:
        split_map = json.loads(Path(args.split_manifest).read_text())["splits"]
    print(
        f"Generating {output_root} from {source_root} using split membership from {split_root or args.split_manifest} "
        f"at crop size {crop_size}x{crop_size}"
    )

    write_split(source_root, output_root, "train", split_map["train"], crop_size)
    write_split(source_root, output_root, "val", split_map["val"], crop_size)
    validate_output(source_root, output_root, split_map, crop_size)
    print("WED_256 generation complete.")


if __name__ == "__main__":
    main()
