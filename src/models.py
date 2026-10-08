import json

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from features import CATEGORICAL_FEATURES
from metric import precision_at_recall

RANDOM_STATE = 42
BOT_CLASS = 1
N_FOLDS = 5
RULE_TUNING_RECALL = 0.75
N_RULES = 6
CATBOOST_FIT_PARAMS = {'model__cat_features': CATEGORICAL_FEATURES}


def create_study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    return optuna.create_study(direction='maximize', sampler=sampler)


def out_of_fold_precision(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    fit_params: dict | None = None,
) -> float:
    folds = GroupKFold(n_splits=N_FOLDS)
    probabilities = cross_val_predict(
        model,
        features,
        target,
        groups=groups,
        cv=folds,
        method='predict_proba',
        params=fit_params,
    )
    return precision_at_recall(target, probabilities[:, BOT_CLASS])


def save_params(params: dict, params_path: str) -> None:
    with open(params_path, 'w') as params_file:
        json.dump(params, params_file, indent=4)
        params_file.write('\n')


def load_params(params_path: str) -> dict:
    with open(params_path) as params_file:
        return json.load(params_file)


def bot_probability(
    model: BaseEstimator, features: pd.DataFrame
) -> np.ndarray:
    return model.predict_proba(features)[:, BOT_CLASS]


def rule_score(features: pd.DataFrame, params: dict) -> pd.Series:
    fast_gaps = features.gap_median_seconds <= params['fast_gap_seconds']
    slow_gaps = features.gap_median_seconds > params['slow_gap_seconds']
    narrow_pointer = features.pointer_x_std < params['pointer_x_std']
    many_locations = (
        features.n_unique_locations >= params['n_unique_locations']
    )
    deep_search = features.max_search_page >= params['max_search_page']
    many_pages_per_query = (
        features.searches_per_query >= params['searches_per_query']
    )
    rules = [
        fast_gaps,
        slow_gaps,
        narrow_pointer,
        many_locations,
        deep_search,
        many_pages_per_query,
    ]
    rule_flags = pd.concat(rules, axis=1).astype(int)
    weights = np.array([params[f'weight_{index}'] for index in range(N_RULES)])
    return rule_flags @ weights / (weights.sum() or 1)


def tune_rules(
    features: pd.DataFrame, target: pd.Series, n_trials: int = 500
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_int('fast_gap_seconds', 5, 60)
        trial.suggest_int('slow_gap_seconds', 120, 1200)
        trial.suggest_int('pointer_x_std', 100, 500, step=10)
        trial.suggest_int('n_unique_locations', 2, 20)
        trial.suggest_int('max_search_page', 3, 30)
        trial.suggest_float('searches_per_query', 1, 6)
        for index in range(N_RULES):
            trial.suggest_float(f'weight_{index}', 0, 1)
        score = rule_score(features, trial.params)
        return precision_at_recall(target, score, recall=RULE_TUNING_RECALL)

    study = create_study()
    study.optimize(objective, n_trials=n_trials)
    return study


def build_logistic_regression(regularization: float) -> Pipeline:
    imputer = SimpleImputer(strategy='median', add_indicator=True)
    numeric_preprocessing = make_pipeline(imputer, StandardScaler())
    categorical_preprocessing = OneHotEncoder(
        handle_unknown='ignore', sparse_output=False
    )
    numeric_columns = make_column_selector(dtype_include='number')
    categorical_columns = make_column_selector(dtype_include='category')
    preprocessing = ColumnTransformer(
        [
            ('numeric', numeric_preprocessing, numeric_columns),
            ('categorical', categorical_preprocessing, categorical_columns),
        ]
    )
    model = LogisticRegression(
        C=regularization, max_iter=5000, random_state=RANDOM_STATE
    )
    return Pipeline([('preprocessing', preprocessing), ('model', model)])


def tune_logistic_regression(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_trials: int = 1000,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        regularization = trial.suggest_float('C', 1e-3, 1, log=True)
        model = build_logistic_regression(regularization)
        return out_of_fold_precision(model, features, target, groups)

    study = create_study()
    study.optimize(objective, n_trials=n_trials)
    return study


def prepare_for_catboost(features: pd.DataFrame) -> pd.DataFrame:
    as_object = dict.fromkeys(CATEGORICAL_FEATURES, object)
    missing_label = dict.fromkeys(CATEGORICAL_FEATURES, 'missing')
    return features.astype(as_object).fillna(missing_label)


def build_catboost(params: dict) -> Pipeline:
    preparation = FunctionTransformer(prepare_for_catboost)
    model = CatBoostClassifier(
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
        **params,
    )
    return Pipeline([('preparation', preparation), ('model', model)])


def fit_catboost(
    params: dict, features: pd.DataFrame, target: pd.Series
) -> Pipeline:
    model = build_catboost(params)
    return model.fit(features, target, **CATBOOST_FIT_PARAMS)


def tune_catboost(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_trials: int = 300,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_int('iterations', 100, 1500, log=True)
        trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        trial.suggest_int('depth', 3, 8)
        trial.suggest_float('l2_leaf_reg', 1, 30, log=True)
        trial.suggest_float('random_strength', 0.1, 10, log=True)
        trial.suggest_float('bagging_temperature', 0, 1)
        trial.suggest_categorical('auto_class_weights', [None, 'Balanced'])
        model = build_catboost(trial.params)
        return out_of_fold_precision(
            model, features, target, groups, CATBOOST_FIT_PARAMS
        )

    study = create_study()
    study.optimize(objective, n_trials=n_trials)
    return study
