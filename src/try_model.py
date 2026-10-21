import json

from src.inference import predict_message


examples = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Congratulations! Claim your free prize now!",
]

for message in examples:
    result = predict_message(message)

    print("\nMessage:")
    print(message)

    print("\nPrediction:")
    print(json.dumps(result, indent=2))
