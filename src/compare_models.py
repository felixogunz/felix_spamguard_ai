from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import UCI_DATA_PATH
from src.data_preparation import load_and_clean_data


def build_logistic_regression():
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


def build_naive_bayes():
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
            MultinomialNB(alpha=0.5),
        ),
    ])


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():
    data = load_and_clean_data(UCI_DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["target"],
        test_size=0.25,
        random_state=42,
        stratify=data["target"],
    )

    models = {
        "Logistic Regression": build_logistic_regression(),
        "Multinomial Naive Bayes": build_naive_bayes(),
    }

    print("Dataset:", UCI_DATA_PATH.name)
    print("Total messages:", len(data))
    print("Training messages:", len(X_train))
    print("Test messages:", len(X_test))
    print("\nMODEL COMPARISON")
    print("-" * 76)

    for name, model in models.items():
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)

        print(f"\n{name}")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
