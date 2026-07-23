import argparse
import random
import shutil
from pathlib import Path

def find_class_folder(root, class_name):
    matches = [p for p in root.rglob(class_name) if p.is_dir()]
    if not matches:
        raise RuntimeError(f"Could not find class folder: {class_name} under {root}")
    return matches[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_root = Path(args.input)
    output_root = Path(args.output)

    train_root = output_root / "Training"
    test_root = output_root / "Testing"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

    for cls in args.classes:
        cls_folder = find_class_folder(input_root, cls)

        files = []
        for ext in image_exts:
            files.extend(cls_folder.glob(ext))

        files = sorted(files)
        random.shuffle(files)

        if len(files) == 0:
            raise RuntimeError(f"No images found in {cls_folder}")

        split_idx = int(len(files) * args.train_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        train_out = train_root / cls
        test_out = test_root / cls
        train_out.mkdir(parents=True, exist_ok=True)
        test_out.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy2(f, train_out / f.name)

        for f in test_files:
            shutil.copy2(f, test_out / f.name)

        print(f"{cls}: total={len(files)}, train={len(train_files)}, test={len(test_files)}")

    print("\nDone.")
    print(f"Prepared dataset saved to: {output_root}")

if __name__ == "__main__":
    main()
