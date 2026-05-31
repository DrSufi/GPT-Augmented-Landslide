#!/usr/bin/env python3
"""
03_gpt_augmentation_pipeline.py

Creates an augmented landslide inventory.

By default, this script uses a deterministic, reproducible local annotation layer.
If USE_OPENAI=1 is set and the OpenAI package plus OPENAI_API_KEY are available,
the script can be extended to call a GPT model. The default local mode is preferred
for exact reproducibility of the proceeding paper outputs.
"""

from pathlib import Path
import re
import pandas as pd

DATA_PATH = Path("data/Landslide_Inventory_of_CHA_Corected.csv")
OUT_PATH = Path("data/CHA_Landslide_GPT_style_augmented.csv")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def is_blank(value) -> bool:
    return clean_text(value) in {"", "na", "n/a", "nan", "null", "none"}


def standardize_trigger(value) -> str:
    text = clean_text(value)
    if not text:
        return "Unknown"

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
    if standardized_value == "Unknown" or is_blank(raw_value):
        return "Low"
    if standardized_value in {"Hill cutting", "Load induced instability"}:
        return "Medium"
    return "High"


def numeric_value(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        text = str(value).replace(",", "").strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def severity_class(row: pd.Series) -> str:
    deaths = numeric_value(row.get("Death_", 0))
    area = numeric_value(row.get("Area", 0))
    settlement = clean_text(row.get("Settlemet_", ""))

    if deaths >= 10:
        return "Extreme"
    if deaths >= 1:
        return "High"
    if "yes" in settlement or area >= 1000:
        return "Moderate"
    return "Low"


def exposure_context(row: pd.Series) -> str:
    values = " ".join(clean_text(row.get(c, "")) for c in row.index)
    if "settlement" in values or "house" in values or "residential" in values:
        return "Settlement exposure"
    if "road" in values:
        return "Road or transport exposure"
    if "hill cut" in values or "hill cutting" in values:
        return "Human modified slope"
    if "agric" in values or "farming" in values or "jhum" in values:
        return "Slope agriculture"
    return "Unspecified local exposure"


def missingness_count(row: pd.Series) -> int:
    return int(sum(is_blank(row.get(col, "")) for col in row.index))


def uncertainty_label(row: pd.Series) -> str:
    count = missingness_count(row)
    trig = standardize_trigger(row.get("Triggers_", ""))
    if count >= 6 or trig == "Unknown":
        return "High"
    if count >= 3:
        return "Moderate"
    return "Low"


def process_narrative(row: pd.Series) -> str:
    district = str(row.get("District", "Unknown")).strip() or "Unknown district"
    fail_type = str(row.get("Fail_Type", "landslide")).strip() or "landslide"
    trigger = standardize_trigger(row.get("Triggers_", ""))
    severity = severity_class(row)

    if trigger == "Unknown":
        return f"A {severity.lower()} severity {fail_type.lower()} event was recorded in {district}; the triggering condition was not sufficiently reported."
    return f"A {severity.lower()} severity {fail_type.lower()} event was recorded in {district}, with {trigger.lower()} identified as the standardized trigger context."


def planning_summary(row: pd.Series) -> str:
    district = str(row.get("District", "Unknown")).strip() or "Unknown district"
    trigger = standardize_trigger(row.get("Triggers_", ""))
    sev = severity_class(row)
    return f"{district} record classified as {sev.lower()} severity; priority should be given to trigger verification, uncertainty review, and local exposure screening for resilience planning."


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["standardized_trigger"] = df["Triggers_"].apply(standardize_trigger)
    df["trigger_confidence"] = [
        trigger_confidence(raw, std) for raw, std in zip(df["Triggers_"], df["standardized_trigger"])
    ]
    df["severity_class"] = df.apply(severity_class, axis=1)
    df["exposure_context"] = df.apply(exposure_context, axis=1)
    df["missingness_count"] = df.apply(missingness_count, axis=1)
    df["uncertainty_label"] = df.apply(uncertainty_label, axis=1)
    df["process_narrative"] = df.apply(process_narrative, axis=1)
    df["planning_summary"] = df.apply(planning_summary, axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Augmented dataset written to: {OUT_PATH}")
    print(f"Records: {len(df)}")
    print("Observed fields preserved; annotations added as separate columns.")


if __name__ == "__main__":
    main()
