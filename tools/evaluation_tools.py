from typing import Literal, Optional, Dict, Any
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

AverageType = Literal[
    "micro",
    "macro",
    "samples",
    "weighted",
    "binary"
]


# ==========================================================
# Validation
# ==========================================================

def _validate_inputs(
    y_true,
    y_pred
):
    """
    Basic validation for evaluation inputs.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Input arrays cannot be empty."
        )


# ==========================================================
# Overfitting Detection
# ==========================================================

def _detect_overfitting(
    train_score=None,
    test_score=None,
    threshold=0.10
):
    """
    Detect potential overfitting based on
    train-test performance gap.
    """

    if train_score is None or test_score is None:
        return False, None

    gap = abs(train_score - test_score)

    return (
        gap > threshold,
        round(gap, 4)
    )


# ==========================================================
# Recommendations
# ==========================================================

def _generate_recommendations(
    metrics: dict,
    task_type: str
):
    """
    Generate recommendations for the
    Critic Agent.
    """

    recommendations = []

    if task_type == "classification":

        if metrics["accuracy"] < 0.70:
            recommendations.append(
                "Consider additional feature engineering."
            )

        if metrics["f1_score"] < 0.70:
            recommendations.append(
                "Class imbalance may exist. Consider resampling techniques."
            )

        if metrics["precision"] < 0.60:
            recommendations.append(
                "High false-positive rate detected. Consider model tuning."
            )

        if metrics["recall"] < 0.60:
            recommendations.append(
                "High false-negative rate detected. Consider improving recall."
            )

        if (
            metrics["accuracy"] -
            metrics["f1_score"]
        ) > 0.15:

            recommendations.append(
                "Accuracy is significantly higher than F1-score. Check for class imbalance."
            )

    elif task_type == "regression":

        if metrics["r2"] < 0.50:
            recommendations.append(
                "Model explains limited variance. Consider feature engineering."
            )

        if metrics["rmse"] > (
            metrics["mae"] * 2
        ):
            recommendations.append(
                "Large prediction errors detected. Investigate outliers."
            )

    if metrics.get("overfitting_risk"):

        recommendations.append(
            "Potential overfitting detected. Consider regularization, cross-validation, or reducing model complexity."
        )

    return recommendations


# ==========================================================
# Performance Summary
# ==========================================================

def _generate_summary(
    metrics: dict,
    task_type: str
):
    """
    Generate a lightweight performance summary.
    """

    if task_type == "classification":

        f1 = metrics["f1_score"]

        if f1 >= 0.90:
            return "Excellent classification performance."

        if f1 >= 0.80:
            return "Strong classification performance."

        if f1 >= 0.70:
            return "Moderate classification performance."

        return "Weak classification performance."

    r2 = metrics["r2"]

    if (
        r2 >= 0.90 and
        metrics["rmse"] < metrics["mae"] * 2
    ):
        return "Excellent predictive performance."

    if (
        r2 >= 0.80 and
        metrics["rmse"] < metrics["mae"] * 2
    ):
        return "Strong predictive performance."

    if r2 >= 0.60:
        return "Moderate predictive performance."

    return "Weak predictive performance."


# ==========================================================
# Classification Evaluation
# ==========================================================

def evaluate_classification(
    y_true,
    y_pred,
    y_proba=None,
    average: AverageType = "weighted",
    train_score=None,
    test_score=None
):
    """
    Evaluate classification predictions.
    """

    _validate_inputs(
        y_true,
        y_pred
    )
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:

        metrics = {

            "task_type": "classification",

            "n_samples": len(y_true),

            "accuracy": round(
                accuracy_score(
                    y_true,
                    y_pred
                ),
                4
            ),

            "precision": round(
                float(
                precision_score(
                    y_true,
                    y_pred,
                    average=average,
                    zero_division=0
                )),
                4
            ),

            "recall": round(
                float(
                recall_score(
                    y_true,
                    y_pred,
                    average=average,
                    zero_division=0
                )),
                4
            ),

            "f1_score": round(
                float(
                f1_score(
                    y_true,
                    y_pred,
                    average=average,
                    zero_division=0
                )),
                4
            )
        }

        if y_proba is not None:

            try:

                if len(np.unique(y_true)) == 2:

                    if hasattr(
                        y_proba,
                        "ndim"
                    ) and y_proba.ndim > 1:

                        y_proba = y_proba[:, 1]

                    metrics["roc_auc"] = round(
                        roc_auc_score(
                            y_true,
                            y_proba
                        ),
                        4
                    )

                else:

                    metrics["roc_auc"] = round(
                        roc_auc_score(
                            y_true,
                            y_proba,
                            multi_class="ovr"
                        ),
                        4
                    )

            except Exception:
                metrics["roc_auc"] = None

        else:
            metrics["roc_auc"] = None

        (
            metrics["overfitting_risk"],
            metrics["train_test_gap"]
        ) = _detect_overfitting(
            train_score,
            test_score
        )

        metrics[
            "performance_summary"
        ] = _generate_summary(
            metrics,
            "classification"
        )

        metrics[
            "recommendations"
        ] = _generate_recommendations(
            metrics,
            "classification"
        )

        return metrics

    except Exception as e:

        raise RuntimeError(
            f"Classification evaluation failed: {e}"
        )


# ==========================================================
# Regression Evaluation
# ==========================================================

def evaluate_regression(
    y_true,
    y_pred,
    train_score=None,
    test_score=None
):
    """
    Evaluate regression predictions.
    """

    _validate_inputs(
        y_true,
        y_pred
    )

    try:

        mse = mean_squared_error(
            y_true,
            y_pred
        )

        metrics = {

            "task_type": "regression",

            "n_samples": len(y_true),

            "mae": round(
                mean_absolute_error(
                    y_true,
                    y_pred
                ),
                4
            ),

            "mse": round(
                mse,
                4
            ),

            "rmse": round(
                np.sqrt(mse),
                4
            ),

            "r2": round(
                r2_score(
                    y_true,
                    y_pred
                ),
                4
            )
        }

        (
            metrics["overfitting_risk"],
            metrics["train_test_gap"]
        ) = _detect_overfitting(
            train_score,
            test_score
        )

        metrics[
            "performance_summary"
        ] = _generate_summary(
            metrics,
            "regression"
        )

        metrics[
            "recommendations"
        ] = _generate_recommendations(
            metrics,
            "regression"
        )

        return metrics

    except Exception as e:

        raise RuntimeError(
            f"Regression evaluation failed: {e}"
        )


# ==========================================================
# Unified Interface
# ==========================================================

def evaluate_model(
    y_true,
    y_pred,
    task_type,
    y_proba=None,
    train_score=None,
    test_score=None
):
    """
    Unified evaluation interface for all agents.
    """

    task_type = task_type.lower()

    if task_type == "classification":

        return evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            train_score=train_score,
            test_score=test_score
        )

    if task_type == "regression":

        return evaluate_regression(
            y_true=y_true,
            y_pred=y_pred,
            train_score=train_score,
            test_score=test_score
        )

    raise ValueError(
        f"Unsupported task type: {task_type}"
    )