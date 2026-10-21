# Felix SpamGuard

Felix SpamGuard is a Python application that checks SMS messages and predicts whether they are spam or normal messages.

The project includes data preparation, model training, model comparison, a REST API, and a web interface for testing messages.

## Features

- Classifies SMS messages as spam or ham.
- Uses TF-IDF to convert text into numerical features.
- Trains a logistic regression model.
- Compares logistic regression with Multinomial Naive Bayes.
- Evaluates models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Returns a prediction and estimated spam probability.
- Provides feature-contribution information for predictions.
- Uses FastAPI to serve predictions.
- Includes a browser-based dashboard.
- Includes automated API tests.

## Tools used

- Python for the application logic.
- pandas for loading and preparing data.
- scikit-learn for text processing, training, and evaluation.
- FastAPI for the API.
- Uvicorn for running the application.
- HTML and CSS for the web interface.
- JavaScript for dashboard interactions.
- pytest for automated tests.
- joblib for saving and loading the model.

## Dataset

The model was trained using the
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

After removing exact duplicates, the cleaned dataset contains:

- Original records: 5,574.
- Exact duplicates removed: 414.
- Cleaned messages: 5,160.
- Normal messages (ham): 4,518.
- Spam messages: 642.

Normal messages make up approximately 87.6% of the cleaned dataset, while spam messages make up about 12.4%.

Download and extract the dataset so that the message file is located at:

`data/raw/uci_sms_spam/SMSSpamCollection`

Keep the dataset's accompanying README and review its usage and attribution terms.

## Setup

### 1. Open the project

cd ~/projects/felix-spamguard-ai

### 2. Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies

python -m pip install -r requirements.txt

### 4. Prepare the dataset

After downloading and extracting the dataset, run:

python -m src.prepare_uci_data

The cleaned dataset is saved to:

`data/processed/uci_sms_spam.csv`

### 5. Train the model

python -m src.train

The training script evaluates the model and saves the trained pipeline and metrics in the `models/` directory.

### 6. Compare models

python -m src.compare_models

This compares logistic regression with Multinomial Naive Bayes using the same train-test split.

### 7. Start the application

uvicorn src.api:app --reload

Open the following addresses in your browser:

- Dashboard: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs

Press `Ctrl+C` to stop the server.

## Model results

Both models were evaluated on 1,290 messages held out from training.

### Logistic Regression

- Accuracy: 98.06%.
- Precision: 90.91%.
- Recall: 93.75%.
- F1-score: 92.31%.
- ROC-AUC: 99.60%.

### Multinomial Naive Bayes

- Accuracy: 96.36%.
- Precision: 100.00%.
- Recall: 70.63%.
- F1-score: 82.78%.
- ROC-AUC: 97.63%.

Logistic regression was selected as the main model because it provided a better balance between precision and recall on this test set.

### Logistic Regression confusion matrix

- 1,115 normal messages were correctly classified as normal.
- 15 normal messages were incorrectly flagged as spam.
- 150 spam messages were correctly identified.
- 10 spam messages were classified as normal.

These results come from one test split of the UCI dataset. Performance on new messages may differ.

## How it works

1. The dataset is loaded and cleaned.
2. TF-IDF converts message text into numerical features.
3. Logistic regression learns patterns associated with spam and normal messages.
4. The model is evaluated on messages excluded from training.
5. The trained pipeline and metrics are saved.
6. The API uses the saved model to classify submitted messages.
7. The dashboard displays the prediction and available results.

## Tests

Run the automated tests:

pytest -v

Check Python syntax:

python -m compileall -q src tests

Check installed dependencies:

python -m pip check

The current test suite covers the dashboard, API health, input validation, and evaluation-metric structure.

## Limitations

The model can make mistakes. It may flag a legitimate message as spam or fail to identify a spam message.

The dataset may not represent current spam techniques or every type of SMS message. The reported results come from a single test split and do not guarantee performance on real-world traffic.

Before using the application in a real service, I would test it on newer data, review the classification threshold, add suitable security controls, and monitor its performance.

## Future improvements

- Test the model on newer SMS datasets.
- Explore different classification thresholds.
- Improve the explanations provided with predictions.
- Add tests for more unusual inputs and API errors.
- Prepare the application for deployment.
- Track model versions and performance over time.
