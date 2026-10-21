import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    UCI_DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    METRICS_PATH,
)
from src.data_preparation import load_and_clean_data


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "test_samples": int(len(y_test)),
    }


def main():
    # Explicitly train using the cleaned UCI dataset.
    data = load_and_clean_data(UCI_DATA_PATH)

    X = data["text"]
    y = data["target"]

    # Keep a stratified test set of unseen messages.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_model()

    # Learn vocabulary and classifier from training data only.
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the trained pipeline.
    joblib.dump(model, MODEL_PATH)

    report = {
        "model_name": "TF-IDF + Logistic Regression",
        "dataset": UCI_DATA_PATH.name,
        "demo_dataset_warning": UCI_DATA_PATH.name.startswith(
            "demo_"
        ),
        "dataset_rows": int(len(data)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "metrics": metrics,
    }

    METRICS_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nModel evaluation")
    print(json.dumps(report, indent=2))
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")

    if report["demo_dataset_warning"]:
        print(
            "\nWARNING: Demo-data metrics are only a pipeline "
            "check, not a real-world performance claim."
        )


if __name__ == "__main__":
    main()
