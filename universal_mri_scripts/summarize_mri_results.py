import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pretty_name(name):
    return name.replace("single_", "").replace("pair_", "").replace("triple_", "").replace("_", " + ").title()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline_accuracy", type=float, required=True, help="Baseline accuracy as percent, example 91.94")
    parser.add_argument("--output_name", default="mri_ablation_summary")
    args = parser.parse_args()

    root = Path(args.results)
    rows = []

    for metrics_path in root.glob("*/metrics.json"):
        combo = metrics_path.parent.name
        with open(metrics_path) as f:
            m = json.load(f)

        acc = m["accuracy_percent"]
        f1 = m["weighted_f1_percent"]
        auc = m.get("macro_auc", None)
        gain = acc - args.baseline_accuracy

        rows.append({
            "Combination": combo,
            "Feature Combination": pretty_name(combo),
            "Accuracy (%)": acc,
            "Gain vs Baseline (%)": gain,
            "Weighted F1 (%)": f1,
            "Macro AUC": auc,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Accuracy (%)", ascending=False)

    csv_path = root / f"{args.output_name}.csv"
    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False))

    plot_df = df.iloc[::-1]

    plt.figure(figsize=(16, 9))
    y = np.arange(len(plot_df))
    plt.barh(y, plot_df["Gain vs Baseline (%)"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y, plot_df["Feature Combination"])
    plt.xlabel("Accuracy Gain vs Baseline (%)")
    plt.ylabel("Viskores Feature Combination")
    plt.title("MRI Viskores Feature Ablation Study: Accuracy Gain vs Baseline")
    plt.grid(axis="x", alpha=0.25)

    for i, gain in enumerate(plot_df["Gain vs Baseline (%)"]):
        if gain >= 0:
            plt.text(gain + 0.05, i, f"+{gain:.2f}%", va="center", ha="left", fontsize=10)
        else:
            plt.text(gain - 0.05, i, f"{gain:.2f}%", va="center", ha="right", fontsize=10)

    plt.tight_layout()
    png_path = root / f"{args.output_name}_gain_chart.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved chart: {png_path}")

if __name__ == "__main__":
    main()
