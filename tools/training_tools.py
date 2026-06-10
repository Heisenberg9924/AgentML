
import time
from typing import Any, Dict, List, Tuple, Optional

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from lightgbm import (
    LGBMClassifier,
    LGBMRegressor
)

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor
)

from sklearn.model_selection import cross_val_score


# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_STATE = 42


# =====================================================
# MODEL REGISTRY
# =====================================================

def _with_defaults(
    defaults: Dict,
    params: Optional[Dict] = None
) -> Dict:
    """
    Merge model defaults with user hyperparameters.

    User-provided hyperparameters intentionally win.
    """

    merged = defaults.copy()
    merged.update(params or {})
    return merged


CLASSIFICATION_MODELS = {
    "logistic_regression":
        lambda params: LogisticRegression(
            **_with_defaults(
                {
                    "max_iter": 1000,
                    "random_state": RANDOM_STATE
                },
                params
            )
        ),

    "random_forest":
        lambda params: RandomForestClassifier(
            **_with_defaults(
                {
                    "n_estimators": 200,
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1
                },
                params
            )
        ),

    "xgboost":
        lambda params: XGBClassifier(
            **_with_defaults(
                {
                    "random_state": RANDOM_STATE,
                    "eval_metric": "logloss"
                },
                params
            )
        ),

    "lightgbm":
        lambda params: LGBMClassifier(
            **_with_defaults(
                {
                    "random_state": RANDOM_STATE,
                    "verbose": -1
                },
                params
            )
        ),

    "catboost":
        lambda params: CatBoostClassifier(
            **_with_defaults(
                {
                    "random_state": RANDOM_STATE,
                    "verbose": 0
                },
                params
            )
        )
}


REGRESSION_MODELS = {
    "linear_regression":
        lambda params: LinearRegression(
            **_with_defaults({}, params)
        ),

    "random_forest":
        lambda params: RandomForestRegressor(
            **_with_defaults(
                {
                    "n_estimators": 200,
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1
                },
                params
            )
        ),

    "xgboost":
        lambda params: XGBRegressor(
            **_with_defaults(
                {
                    "random_state": RANDOM_STATE
                },
                params
            )
        ),

    "lightgbm":
        lambda params: LGBMRegressor(
            **_with_defaults(
                {
                    "random_state": RANDOM_STATE,
                    "verbose": -1
                },
                params
            )
        ),

    "catboost":
        lambda params: CatBoostRegressor(
            **_with_defaults(
                {
                    "random_state": RANDOM_STATE,
                    "verbose": 0
                },
                params
            )
        )
}


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _get_model(
    task_type: str,
    model_name: str,
    hyperparameters: Optional[Dict] = None
):
    """
    Create and return a fresh model instance.
    """

    task_type = task_type.lower()
    model_name = model_name.lower()

    hyperparameters = hyperparameters or {}

    if task_type == "classification":

        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(
                f"Unsupported classification model: {model_name}"
            )

        return CLASSIFICATION_MODELS[
            model_name
        ](hyperparameters)

    elif task_type == "regression":

        if model_name not in REGRESSION_MODELS:
            raise ValueError(
                f"Unsupported regression model: {model_name}"
            )

        return REGRESSION_MODELS[
            model_name
        ](hyperparameters)

    else:
        raise ValueError(
            "task_type must be either "
            "'classification' or 'regression'"
        )


def _has_feature_importance(
    model: Any
) -> bool:
    """
    Check whether the model exposes
    feature importance information.
    """

    return (
        hasattr(model, "feature_importances_")
        or hasattr(model, "coef_")
    )


def _validate_inputs(
    X_train,
    y_train
) -> None:
    """
    Basic validation checks.
    """

    if X_train is None:
        raise ValueError("X_train cannot be None")

    if y_train is None:
        raise ValueError("y_train cannot be None")

    if len(X_train) == 0:
        raise ValueError("X_train is empty")

    if len(y_train) == 0:
        raise ValueError("y_train is empty")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must have "
            "the same number of rows"
        )


# =====================================================
# MAIN TRAINING TOOL
# =====================================================

def train_model(
    X_train,
    y_train,
    task_type: str,
    model_name: str,
    hyperparameters: Optional[Dict] = None
) -> Tuple[Any, Dict]:
    """
    Generic model training utility.

    Parameters
    ----------
    X_train : pd.DataFrame

    y_train : pd.Series

    task_type : str
        classification / regression

    model_name : str

        Classification:
            - logistic_regression
            - random_forest
            - xgboost
            - lightgbm
            - catboost

        Regression:
            - linear_regression
            - random_forest
            - xgboost
            - lightgbm
            - catboost

    hyperparameters : dict, optional
        Additional model-specific parameters.

    Returns
    -------
    model :
        Trained model object.

    metadata : dict
        Training metadata useful for
        evaluation, explainability,
        and critic-agent reasoning.
    """

    _validate_inputs(
        X_train,
        y_train
    )

    model = _get_model(
        task_type=task_type,
        model_name=model_name,
        hyperparameters=hyperparameters
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    train_time = round(
        time.time() - start_time,
        4
    )

    metadata = {
        "model_name":
            model.__class__.__name__,

        "task_type":
            task_type.lower(),

        "requested_model":
            model_name.lower(),

        "hyperparameters":
            hyperparameters or {},

        "train_time":
            train_time,

        "num_samples":
            len(X_train),

        "num_features":
            X_train.shape[1],

        "feature_importance_available":
            _has_feature_importance(model)
    }

    return model, metadata


# =====================================================
# MODEL SELECTION TOOL
# =====================================================

def compare_models(
    X_train,
    y_train,
    task_type: str,
    model_names: Optional[List[str]] = None,
    hyperparameters: Optional[Dict[str, Dict]] = None,
    cv: int = 5,
    scoring: Optional[str] = None
) -> Dict:
    """
    Compare candidate models with cross-validation.

    Returns agent-friendly results and keeps per-model failures
    instead of aborting the whole selection run.
    """

    _validate_inputs(
        X_train,
        y_train
    )

    task_type = task_type.lower()
    hyperparameters = hyperparameters or {}

    if task_type == "classification":
        registry = CLASSIFICATION_MODELS
        scoring = scoring or "f1_weighted"
    elif task_type == "regression":
        registry = REGRESSION_MODELS
        scoring = scoring or "r2"
    else:
        raise ValueError(
            "task_type must be either "
            "'classification' or 'regression'"
        )

    model_names = model_names or list(registry.keys())
    results = []
    failures = []

    for model_name in model_names:
        try:
            model = _get_model(
                task_type=task_type,
                model_name=model_name,
                hyperparameters=hyperparameters.get(model_name)
            )

            start_time = time.time()
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring
            )

            results.append(
                {
                    "model_name": model_name,
                    "score_mean": round(float(scores.mean()), 4),
                    "score_std": round(float(scores.std()), 4),
                    "scores": [
                        round(float(score), 4)
                        for score in scores
                    ],
                    "scoring": scoring,
                    "cv": cv,
                    "elapsed_time": round(
                        time.time() - start_time,
                        4
                    )
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "model_name": model_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)
                }
            )

    results = sorted(
        results,
        key=lambda item: item["score_mean"],
        reverse=True
    )

    return {
        "task_type": task_type,
        "best_model": (
            results[0]["model_name"]
            if results
            else None
        ),
        "results": results,
        "failures": failures
    }
