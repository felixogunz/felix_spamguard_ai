import json
import joblib

from src.config import MODEL_PATH, METRICS_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The model has not been trained. "
            "Run: python -m src.train"
        )

    return joblib.load(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    return json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )


def predict_message(text):
    model = load_model()

    probabilities = model.predict_proba([text])[0]
    spam_probability = float(probabilities[1])

    label = (
        "spam"
        if spam_probability >= 0.5
        else "ham"
    )

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]

    # Contribution of each active feature to the
    # linear spam score.
    contributions = vector.multiply(weights).toarray()[0]

    active_features = [
        (feature_names[i], float(contributions[i]))
        for i in vector.indices
    ]

    active_features.sort(
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    signals = [
        {
            "term": term,
            "direction": (
                "spam" if score > 0 else "legitimate"
            ),
            "weight": round(score, 4),
        }
        for term, score in active_features[:6]
    ]

    return {
        "label": label,
        "spam_probability": round(spam_probability, 4),
        "ham_probability": round(
            1.0 - spam_probability, 4
        ),
        "threshold": 0.5,
        "signals": signals,
    }
