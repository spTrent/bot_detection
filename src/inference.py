import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from features import build_features, read_cookies, read_events
from models import bot_probability

MODEL_PATH = 'models/catboost.joblib'


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)


def predict_bot_scores(
    model: Pipeline, events: pd.DataFrame, cookies: pd.DataFrame
) -> pd.DataFrame:
    features = build_features(events, cookies)
    scores = bot_probability(model, features)
    return pd.DataFrame({'cookie_id': features.index, 'score': scores})


def main() -> None:
    parser = argparse.ArgumentParser(description='Скоринг кук моделью')
    parser.add_argument('--cookies', required=True, help='csv с окнами кук')
    parser.add_argument('--events', required=True, help='csv с событиями')
    parser.add_argument('--output', required=True, help='куда записать скоры')
    parser.add_argument('--model', default=MODEL_PATH, help='путь к модели')
    arguments = parser.parse_args()
    model = load_model(arguments.model)
    cookies = read_cookies(arguments.cookies)
    events = read_events(arguments.events)
    scores = predict_bot_scores(model, events, cookies)
    scores.to_csv(arguments.output, index=False)


if __name__ == '__main__':
    main()
