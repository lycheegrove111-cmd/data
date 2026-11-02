"""Audit and clean the glaucoma dataset metadata without changing raw files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


SPLITS = ("train", "valid", "test")


def find_image(path: Path, requested: str) -> Path | None:
    exact = path / requested
    if exact.is_file():
        return exact
    lower = requested.lower()
    matches = [p for p in path.iterdir() if p.is_file() and p.name.lower() == lower]
    return matches[0] if len(matches) == 1 else None


def audit_split(data_root: Path, split: str) -> tuple[pd.DataFrame, dict]:
    split_dir = data_root / split
    frame = pd.read_csv(data_root / f"{split}.csv")
    frame["Image_name"] = frame["Image_name"].astype("string")
    issues: list[dict] = []
    resolved: list[str | None] = []
    readable: list[bool] = []

    for name in frame["Image_name"]:
        image = find_image(split_dir, str(name)) if pd.notna(name) else None
        resolved.append(str(image) if image else None)
        if image is None:
            readable.append(False)
            issues.append({"split": split, "image_name": name, "issue": "missing_image"})
            continue
        try:
            with Image.open(image) as opened:
                opened.verify()
            readable.append(True)
        except Exception as exc:  # Pillow reports different exceptions per format.
            readable.append(False)
            issues.append({
                "split": split,
                "image_name": name,
                "issue": "unreadable_image",
                "detail": type(exc).__name__,
            })

    frame["resolved_path"] = resolved
    frame["readable"] = readable
    frame["duplicate_row"] = frame.duplicated(subset=["Image_name"], keep="first")
    frame["invalid_label"] = False
    if "True_Label" in frame:
        frame["invalid_label"] = ~frame["True_Label"].isin([0, 1])
    frame["age_missing"] = frame["Age"].isna()
    frame["age_anomaly"] = frame["Age"].notna() & ~frame["Age"].between(0, 120)

    for _, row in frame[frame["duplicate_row"] | frame["invalid_label"]].iterrows():
        issue = "duplicate_row" if row["duplicate_row"] else "invalid_label"
        issues.append({"split": split, "image_name": row["Image_name"], "issue": issue})

    keep = frame["readable"] & ~frame["duplicate_row"] & ~frame["invalid_label"]
    cleaned = frame.loc[keep].copy()
    median_age = cleaned["Age"].median()
    cleaned["Age_original"] = cleaned["Age"]
    cleaned["Age"] = cleaned["Age"].fillna(median_age)
    cleaned["Age_imputed"] = cleaned["Age_original"].isna()
    cleaned = cleaned.drop(columns=["resolved_path", "readable", "duplicate_row", "invalid_label"])
    cleaned.insert(0, "split", split)

    summary = {
        "rows_original": int(len(frame)),
        "rows_cleaned": int(len(cleaned)),
        "rows_removed": int(len(frame) - len(cleaned)),
        "missing_images": int((frame["resolved_path"].isna()).sum()),
        "unreadable_images": int((~frame["readable"] & frame["resolved_path"].notna()).sum()),
        "duplicate_rows": int(frame["duplicate_row"].sum()),
        "invalid_labels": int(frame["invalid_label"].sum()),
        "missing_age": int(frame["age_missing"].sum()),
        "age_anomaly": int(frame["age_anomaly"].sum()),
        "median_age_used_for_imputation": float(median_age) if pd.notna(median_age) else None,
    }
    if "True_Label" in cleaned:
        counts = cleaned["True_Label"].value_counts().to_dict()
        summary["label_0"] = int(counts.get(0, 0))
        summary["label_1"] = int(counts.get(1, 0))
    return cleaned, {"summary": summary, "issues": issues}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_issues: list[dict] = []
    summaries: dict[str, dict] = {}
    for split in SPLITS:
        cleaned, audit = audit_split(args.data_root, split)
        cleaned.to_csv(args.output_root / f"{split}_clean.csv", index=False)
        summaries[split] = audit["summary"]
        all_issues.extend(audit["issues"])

    pd.DataFrame(all_issues, columns=["split", "image_name", "issue", "detail"]).to_csv(
        args.output_root / "data_issues.csv", index=False
    )
    pd.DataFrame([{"split": split, **summary} for split, summary in summaries.items()]).to_csv(
        args.output_root / "class_counts.csv", index=False
    )
    report = {
        "data_root": str(args.data_root.resolve()),
        "splits": summaries,
        "total_issues": len(all_issues),
        "cleaning_policy": [
            "Keep the official train/valid/test split.",
            "Remove missing or unreadable images, duplicate rows, and labels outside {0, 1}.",
            "Keep age anomalies for traceability; impute missing ages with the cleaned split median.",
            "Never modify or delete raw images or source CSV files.",
        ],
    }
    (args.output_root / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
