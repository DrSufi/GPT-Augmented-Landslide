#!/usr/bin/env python3
"""
01_missingness_audit.py

Audits missing or blank values in the Chittagong Hill Area landslide inventory.
The script writes a CSV summary and can be used as the first reproducibility step.
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/Landslide_Inventory_of_CHA_Corected.csv")
OUT_PATH = Path("outputs/tables/table1_missingness_audit.csv")


def is_missing(series: pd.Series) -> pd.Series:
    """Return True for null, blank, NA-like, or whitespace-only values."""
    return (
        series.isna()
        | series.astype(str).str.strip().eq("")
        | series.astype(str).str.strip().str.lower().isin({"na", "n/a", "nan", "null", "none"})
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    n = len(df)

    rows = []
    for col in df.columns:
        missing_count = int(is_missing(df[col]).sum())
        rows.append(
            {
                "field": col,
                "missing_records": missing_count,
                "missing_percentage": round((missing_count / n) * 100, 1),
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        ["missing_records", "field"], ascending=[False, True]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_PATH, index=False)

    print(f"Records audited: {n}")
    print(f"Missingness table written to: {OUT_PATH}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
