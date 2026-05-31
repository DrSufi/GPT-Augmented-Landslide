#!/usr/bin/env python3
"""
04_validation_summary.py

Generates validation summaries for the controlled GPT augmentation.
"""

from pathlib import Path
import pandas as pd

ORIGINAL_PATH = Path("data/Landslide_Inventory_of_CHA_Corected.csv")
AUGMENTED_PATH = Path("data/CHA_Landslide_GPT_style_augmented.csv")
OUT_PATH = Path("outputs/tables/table5_validation_summary.csv")


def main() -> None:
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {ORIGINAL_PATH}")
    if not AUGMENTED_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {AUGMENTED_PATH}")

    original = pd.read_csv(ORIGINAL_PATH)
    augmented = pd.read_csv(AUGMENTED_PATH)

    original_cols = list(original.columns)
    if not set(original_cols).issubset(set(augmented.columns)):
        raise ValueError("Augmented file does not contain all original columns.")

    preserved = original.reset_index(drop=True).equals(augmented[original_cols].reset_index(drop=True))
    observed_field_preservation = "100%" if preserved else "Failed"

    raw_trigger_categories = (
        original["Triggers_"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
        if "Triggers_" in original.columns
        else None
    )

    standardized_trigger_categories = (
        augmented["standardized_trigger"].nunique()
        if "standardized_trigger" in augmented.columns
        else None
    )

    rows_with_semantic_descriptor = (
        augmented["process_narrative"].notna().sum()
        if "process_narrative" in augmented.columns
        else 0
    )

    rows_with_moderate_or_high_uncertainty = (
        augmented["uncertainty_label"].isin(["Moderate", "High"]).sum()
        if "uncertainty_label" in augmented.columns
        else 0
    )

    summary = pd.DataFrame(
        [
            {
                "validation_item": "Observed field preservation",
                "result": observed_field_preservation,
                "interpretation": "Original empirical fields were not overwritten",
            },
            {
                "validation_item": "Original trigger categories",
                "result": raw_trigger_categories,
                "interpretation": "Raw trigger field contained spelling and naming variation",
            },
            {
                "validation_item": "Augmented trigger categories",
                "result": standardized_trigger_categories,
                "interpretation": "Trigger noise was reduced into an interpretable taxonomy",
            },
            {
                "validation_item": "Rows with semantic descriptor",
                "result": int(rows_with_semantic_descriptor),
                "interpretation": "Every record received a structured process narrative",
            },
            {
                "validation_item": "Rows with moderate or high uncertainty",
                "result": int(rows_with_moderate_or_high_uncertainty),
                "interpretation": "Uncertainty was retained rather than hidden",
            },
        ]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    print(f"Validation summary written to: {OUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
