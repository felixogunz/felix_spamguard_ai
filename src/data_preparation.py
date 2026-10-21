
import pandas as pd

from src.config import DATA_PATH


def load_and_clean_data(data_path=DATA_PATH):
    """Load a dataset, validate it, and prepare its labels."""

    data = pd.read_csv(data_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    # Remove records without message text or a label.
    data = data.dropna(subset=["text", "label"]).copy()

    # Normalize whitespace and label formatting.
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = (
        data["label"].astype(str).str.strip().str.lower()
    )

    # Keep supported labels and non-empty messages.
    data = data[
        data["text"].ne("")
        & data["label"].isin(["ham", "spam"])
    ].copy()

    # Map categories to numerical values.
    data["target"] = data["label"].map({
        "ham": 0,
        "spam": 1,
    })

    if data.empty or data["target"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both ham and spam examples."
        )

    return data


if __name__ == "__main__":
    cleaned = load_and_clean_data()

    print(f"Cleaned records: {len(cleaned)}")
    print(cleaned["label"].value_counts().to_string())
