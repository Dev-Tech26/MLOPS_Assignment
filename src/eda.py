import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from pathlib import Path

base_dir = Path.cwd()
if "__file__" in globals():
    base_dir = Path(__file__).resolve().parent

src_dir = base_dir if base_dir.name == "src" else base_dir / "src"
src_dir = src_dir.resolve()
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

project_root = src_dir.parent if src_dir.name == "src" else base_dir
DATA_PATH = project_root / "data" / "heart.csv"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

from preprocess import download_data, load_data, FEATURE_COLS, TARGET_COL
sns.set_theme(style="whitegrid", palette="muted")


def plot_histograms(df, save_dir):
    features = [c for c in df.columns if c != TARGET_COL]
    n = len(features)
    cols_g = 5
    rows_g = (n + cols_g - 1) // cols_g
    fig, axes = plt.subplots(rows_g, cols_g, figsize=(20, rows_g * 3.5))
    axes = axes.flatten()
    for i, col in enumerate(features):
        axes[i].hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions – Heart Disease UCI Dataset", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "01_histograms.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df, save_dir):
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(save_dir, "02_correlation_heatmap.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_class_distribution(df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df[TARGET_COL].value_counts()
    labels = ["No Disease", "Heart Disease"]
    colors = ["#4CAF50", "#F44336"]
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Class Distribution (Count)", fontsize=13, fontweight="bold")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Distribution (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "03_class_distribution.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_missing_values(df, save_dir):
    missing = df.isnull().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(missing.index, missing.values,
                  color=["#F44336" if v > 0 else "#4CAF50" for v in missing.values])
    ax.set_title("Missing Value Analysis", fontsize=13, fontweight="bold")
    ax.set_ylabel("Missing Count")
    for bar, v in zip(bars, missing.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(v), ha="center", fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "04_missing_values.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_feature_relationships(df, save_dir):
    key_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, feat in enumerate(key_feats):
        groups = [df[df[TARGET_COL] == 0][feat].dropna(),
                  df[df[TARGET_COL] == 1][feat].dropna()]
        bp = axes[i].boxplot(groups, patch_artist=True)
        bp["boxes"][0].set_facecolor("#4CAF5088")
        bp["boxes"][1].set_facecolor("#F4433688")
        axes[i].set_title(feat, fontsize=12, fontweight="bold")
        axes[i].set_xticklabels(["No Disease", "Heart Disease"])
    axes[-1].set_visible(False)
    fig.suptitle("Feature Distributions by Target Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "05_feature_relationships.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
    else:
        df = download_data()
        df.to_csv(DATA_PATH, index=False)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Class distribution:\n{df[TARGET_COL].value_counts()}")

    plot_histograms(df, REPORTS_DIR)
    plot_correlation_heatmap(df, REPORTS_DIR)
    plot_class_distribution(df, REPORTS_DIR)
    plot_missing_values(df, REPORTS_DIR)
    plot_feature_relationships(df, REPORTS_DIR)
    print("\n✅ EDA complete. All charts saved to reports/")


if __name__ == "__main__":
    main()