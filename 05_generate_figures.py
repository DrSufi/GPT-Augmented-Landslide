#!/usr/bin/env python3
"""
05_generate_figures.py

Generates the three figures used in the proceeding paper:
1. Missingness profile
2. District by failure type matrix
3. Trigger taxonomy bubble plot by district
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORIGINAL_PATH = Path("data/Landslide_Inventory_of_CHA_Corected.csv")
AUGMENTED_PATH = Path("data/CHA_Landslide_GPT_style_augmented.csv")
FIG_DIR = Path("figures")


def is_missing(series: pd.Series) -> pd.Series:
    return (
        series.isna()
        | series.astype(str).str.strip().eq("")
        | series.astype(str).str.strip().str.lower().isin({"na", "n/a", "nan", "null", "none"})
    )


def numeric_value(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)


def plot_missingness(df: pd.DataFrame) -> None:
    n = len(df)
    missing = (
        pd.DataFrame(
            {
                "field": df.columns,
                "missing_percentage": [is_missing(df[col]).sum() / n * 100 for col in df.columns],
            }
        )
        .sort_values("missing_percentage", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(missing["field"], missing["missing_percentage"])
    ax.set_xlabel("Missing or blank records (%)")
    ax.set_ylabel("Inventory field")
    ax.set_title("Baseline Incompleteness Profile of the CHA Landslide Inventory")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plot1_missingness_profile.png", dpi=300)
    plt.close(fig)


def plot_district_failure_heatmap(df: pd.DataFrame) -> None:
    matrix = pd.crosstab(df["District"], df["Fail_Type"])
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    matrix = matrix[matrix.sum(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title("District Level Concentration Matrix of Landslide Failure Types")
    ax.set_xlabel("Failure type")
    ax.set_ylabel("District")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Record count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plot2_district_failure_heatmap.png", dpi=300)
    plt.close(fig)


def plot_trigger_bubble(df: pd.DataFrame) -> None:
    df = df.copy()
    df["Death_num"] = numeric_value(df["Death_"])

    grouped = (
        df.groupby(["District", "standardized_trigger"], dropna=False)
        .agg(records=("standardized_trigger", "size"), deaths=("Death_num", "sum"))
        .reset_index()
    )

    districts = grouped["District"].dropna().unique().tolist()
    triggers = grouped["standardized_trigger"].dropna().unique().tolist()

    x_map = {v: i for i, v in enumerate(triggers)}
    y_map = {v: i for i, v in enumerate(districts)}

    fig, ax = plt.subplots(figsize=(11, 6))
    x = grouped["standardized_trigger"].map(x_map)
    y = grouped["District"].map(y_map)
    sizes = grouped["records"] * 18

    ax.scatter(x, y, s=sizes, alpha=0.55)

    for _, row in grouped.iterrows():
        if row["records"] >= 5 or row["deaths"] > 0:
            ax.text(
                x_map[row["standardized_trigger"]],
                y_map[row["District"]],
                str(int(row["deaths"])),
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_xticks(range(len(triggers)))
    ax.set_xticklabels(triggers, rotation=45, ha="right")
    ax.set_yticks(range(len(districts)))
    ax.set_yticklabels(districts)
    ax.set_xlabel("Standardized trigger class")
    ax.set_ylabel("District")
    ax.set_title("Augmented Trigger Taxonomy by District")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plot3_trigger_district_bubble.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {ORIGINAL_PATH}")
    original = pd.read_csv(ORIGINAL_PATH)
    plot_missingness(original)
    plot_district_failure_heatmap(original)

    if not AUGMENTED_PATH.exists():
        raise FileNotFoundError(
            f"Augmented file not found: {AUGMENTED_PATH}. Run 03_gpt_augmentation_pipeline.py first."
        )
    augmented = pd.read_csv(AUGMENTED_PATH)
    plot_trigger_bubble(augmented)

    print(f"Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
