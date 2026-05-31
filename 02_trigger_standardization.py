#!/usr/bin/env python3
"""
02_trigger_standardization.py

Normalizes raw trigger labels into a controlled taxonomy.
This script performs deterministic cleaning before any GPT annotation is used.
"""

from pathlib import Path
import re
import pandas as pd

DATA_PATH = Path("data/Landslide_Inventory_of_CHA_Corected.csv")
OUT_PATH = Path("outputs/tables/trigger_standardization_preview.csv")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def standardize_trigger(value) -> str:
    text = clean_text(value)

    if not text or text in {"na", "n/a", "nan", "null", "none"}:
        return "Unknown"

    # Correct common spelling variants
    text = text.replace("rianfall", "rainfall")
    text = text.replace("raifall", "rainfall")
    text = text.replace("axtivities", "activities")

    if "load" in text and "top" in text:
        return "Load induced instability"
    if "rainfall" in text and "hill" in text and "cut" in text:
        return "Rainfall and hill cutting"
    if "rainfall" in text and "road" in text and "cut" in text:
        return "Rainfall and road cutting"
    if "rainfall" in text and ("farming" in text or "jhum" in text or "agric" in text):
        return "Rainfall and slope agriculture"
    if "rainfall" in text and "construction" in text:
        return "Rainfall and construction activity"
    if "hill" in text and "cut" in text:
        return "Hill cutting"
    if "rainfall" in text:
        return "Rainfall"

    return "Unknown"


def trigger_confidence(raw_value, standardized_value: str) -> str:
    raw = clean_text(raw_value)
    if standardized_value == "Unknown":
        return "Low"
    if not raw:
        return "Low"
    if standardized_value in {"Hill cutting", "Load induced instability"}:
        return "Medium"
    return "High"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Triggers_" not in df.columns:
        raise KeyError("Expected column not found: Triggers_")

    df["standardized_trigger"] = df["Triggers_"].apply(standardize_trigger)
    df["trigger_confidence"] = [
        trigger_confidence(raw, std) for raw, std in zip(df["Triggers_"], df["standardized_trigger"])
    ]

    summary = (
        df.groupby(["standardized_trigger", "trigger_confidence"], dropna=False)
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    print(f"Trigger standardization summary written to: {OUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
