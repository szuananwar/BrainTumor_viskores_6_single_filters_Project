import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

COMBINATIONS = {
    "single_median": ["median"],
    "single_sobel": ["sobel"],
    "single_laplacian": ["laplacian"],
    "single_texture": ["texture"],
    "single_otsu": ["otsu"],
    "single_morphology": ["morphology"],
    "pair_median_sobel": ["median", "sobel"],
    "pair_sobel_laplacian": ["sobel", "laplacian"],
    "pair_texture_otsu": ["texture", "otsu"],
    "pair_otsu_morphology": ["otsu", "morphology"],
    "triple_median_sobel_laplacian": ["median", "sobel", "laplacian"],
    "triple_texture_otsu_morphology": ["texture", "otsu", "morphology"],
    "triple_sobel_texture_morphology": ["sobel", "texture", "morphology"],
    "all_features": ["median", "sobel", "laplacian", "texture", "otsu", "morphology"],
}

def normalize(x):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return (x * 255).astype(np.uint8)

def otsu_threshold(arr):
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0
    w_b = 0
    max_var = 0
    threshold = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold

def morphology(binary):
    padded = np.pad(binary, 1, mode="edge")
    out = np.zeros_like(binary)

    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            window = padded[i:i+3, j:j+3]
            out[i, j] = 255 if window.max() > 0 else 0

    return out.astype(np.uint8)

def texture_variance(arr):
    padded = np.pad(arr, 1, mode="edge").astype(np.float32)
    out = np.zeros_like(arr, dtype=np.float32)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i:i+3, j:j+3]
            out[i, j] = window.var()

    return normalize(out)

def extract_features(img_path):
    img = Image.open(img_path).convert("L").resize((224, 224))
    arr = np.array(img)

    median_img = img.filter(ImageFilter.MedianFilter(size=3))
    median = np.array(median_img)

    sobel_x = median_img.filter(ImageFilter.Kernel((3, 3), [-1,0,1,-2,0,2,-1,0,1], scale=1))
    sobel_y = median_img.filter(ImageFilter.Kernel((3, 3), [-1,-2,-1,0,0,0,1,2,1], scale=1))
    sobel = normalize(np.abs(np.array(sobel_x, dtype=np.float32)) + np.abs(np.array(sobel_y, dtype=np.float32)))

    laplacian_img = median_img.filter(ImageFilter.Kernel((3, 3), [0,1,0,1,-4,1,0,1,0], scale=1))
    laplacian = normalize(np.abs(np.array(laplacian_img, dtype=np.float32)))

    texture = texture_variance(median)

    threshold = otsu_threshold(median)
    otsu = np.where(median > threshold, 255, 0).astype(np.uint8)

    morph = morphology(otsu)

    return {
        "median": median,
        "sobel": sobel,
        "laplacian": laplacian,
        "texture": texture,
        "otsu": otsu,
        "morphology": morph,
    }

def fuse_features(feature_dict, selected):
    maps = [feature_dict[name] for name in selected]

    if len(maps) == 1:
        rgb = np.stack([maps[0], maps[0], maps[0]], axis=-1)

    elif len(maps) == 2:
        avg = normalize((maps[0].astype(np.float32) + maps[1].astype(np.float32)) / 2)
        rgb = np.stack([maps[0], maps[1], avg], axis=-1)

    elif len(maps) == 3:
        rgb = np.stack([maps[0], maps[1], maps[2]], axis=-1)

    else:
        edge_group = np.mean([feature_dict["sobel"], feature_dict["laplacian"]], axis=0)
        structure_group = np.mean([feature_dict["median"], feature_dict["texture"]], axis=0)
        mask_group = np.mean([feature_dict["otsu"], feature_dict["morphology"]], axis=0)
        rgb = np.stack([normalize(edge_group), normalize(structure_group), normalize(mask_group)], axis=-1)

    return Image.fromarray(rgb.astype(np.uint8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--combos", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["Training", "Testing"])
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    splits = args.splits
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

    selected_combos = COMBINATIONS
    if args.combos is not None:
        selected_combos = {k: COMBINATIONS[k] for k in args.combos}

    for combo_name, selected_features in selected_combos.items():
        print(f"\nCreating combination: {combo_name}")

        for split in splits:
            for cls in args.classes:
                src_dir = input_root / split / cls
                dst_dir = output_root / combo_name / split / cls
                dst_dir.mkdir(parents=True, exist_ok=True)

                files = []
                for ext in image_exts:
                    files.extend(src_dir.glob(ext))

                for idx, img_path in enumerate(files):
                    try:
                        out_name = img_path.stem + ".png"
                        out_path = dst_dir / out_name

                        # Resume support: skip images already generated
                        if out_path.exists():
                            continue

                        features = extract_features(img_path)
                        fused = fuse_features(features, selected_features)
                        fused.save(out_path)

                        if idx % 500 == 0:
                            print(f"{combo_name} | {split} | {cls}: {idx}/{len(files)}")

                    except Exception as e:
                        print(f"Skipped {img_path}: {e}")

    print("\nDone creating Viskores-style feature inputs.")
    print(f"Saved to: {output_root}")

if __name__ == "__main__":
    main()
