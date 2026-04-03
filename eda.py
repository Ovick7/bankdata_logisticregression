import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATASET_PATH = "../data/bank-full.csv"
PLOT_OUTPUT_DIR = "../reports/figures"


def load_data(path: str) -> pd.DataFrame:
    sep = ";" if path.endswith(".csv") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip().str.replace('"', "")
    df = df.applymap(lambda x: x.strip().replace('"', "") if isinstance(x, str) else x)
    return df


def plot_target_distribution(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    counts = df["y"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["steelblue", "tomato"])
    ax.set_xlabel("Subscribed to Term Deposit")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    n = len(numeric_cols)
    fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3, figsize=(15, 4 * ((n + 2) // 3)))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numeric Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "numeric_distributions.png"), dpi=150)
    plt.close()


def plot_categorical_vs_target(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    if "y" in categorical_cols:
        categorical_cols.remove("y")

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df["y"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_title(f"Subscription Rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Subscribed", labels=["No", "Yes"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cat_{col}_vs_target.png"), dpi=150)
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap - Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def main():
    print("Loading data for EDA...")
    df = load_data(DATASET_PATH)

    print("Plotting target distribution...")
    plot_target_distribution(df, PLOT_OUTPUT_DIR)

    print("Plotting numeric feature distributions...")
    plot_numeric_distributions(df, PLOT_OUTPUT_DIR)

    print("Plotting categorical features vs target...")
    plot_categorical_vs_target(df, PLOT_OUTPUT_DIR)

    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(df, PLOT_OUTPUT_DIR)

    print(f"\nAll EDA plots saved to: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
