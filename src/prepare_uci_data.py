import csv
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    ROOT_DIR / "data" / "raw" / "uci_sms_spam" / "SMSSpamCollection"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "uci_sms_spam.csv"


def prepare_uci_dataset():
    """Validate, clean, deduplicate, and save the UCI SMS dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"UCI dataset not found: {RAW_DATA_PATH}"
        )

    # Read the tab-separated dataset without interpreting message quotes.
    data = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        dtype=str,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="error",
    )

    original_count = len(data)

    # Remove incomplete records and normalize the fields.
    data = data.dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].str.strip().str.lower()
    data["text"] = data["text"].str.strip()

    data = data[
        data["label"].isin(["ham", "spam"])
        & data["text"].ne("")
    ].copy()

    if data.empty:
        raise ValueError("No valid messages were found.")

    # Prevent identical messages with conflicting labels.
    label_counts = data.groupby("text")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1]

    if not conflicting_texts.empty:
        raise ValueError(
            f"Found {len(conflicting_texts)} message texts "
            "with conflicting labels."
        )

    # Remove repeated messages that have the same label.
    before_deduplication = len(data)
    data = data.drop_duplicates(
        subset=["label", "text"]
    ).copy()

    duplicates_removed = before_deduplication - len(data)

    # Convert labels into numbers for machine learning.
    data["target"] = (
        data["label"].map({"ham": 0, "spam": 1}).astype(int)
    )

    if data["label"].nunique() != 2:
        raise ValueError(
            "The cleaned dataset must contain both ham and spam."
        )

    # Save the cleaned dataset.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print("===== UCI DATASET PREPARATION =====")
    print(f"Original records: {original_count}")
    print(f"Exact duplicates removed: {duplicates_removed}")
    print(f"Cleaned records: {len(data)}")

    print("\nClass counts:")
    print(data["label"].value_counts().to_string())

    print("\nNumeric target counts:")
    print(data["target"].value_counts().sort_index().to_string())

    print(f"\nSaved dataset to: {OUTPUT_PATH}")

    print("\nFirst five cleaned records:")
    print(data.head().to_string(index=False))


if __name__ == "__main__":
    prepare_uci_dataset()
