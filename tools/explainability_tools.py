from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def get_feature_importance(
    model: Any,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Extract feature importances from tree-based models.
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{type(model).__name__} does not expose feature_importances_."
        )

    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature importance length does not match feature_names."
        )

    importance_dict = {
        feature: float(score)
        for feature, score in zip(feature_names, importances)
    }

    return dict(
        sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# ==========================================================
# COEFFICIENT IMPORTANCE
# ==========================================================

def get_coefficient_importance(
    model: Any,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Extract importance from linear/logistic models.

    Uses absolute coefficient magnitude for ranking.
    """

    if not hasattr(model, "coef_"):
        raise ValueError(
            f"{type(model).__name__} does not expose coef_."
        )

    coef = model.coef_

    if coef.ndim > 1:
        coef = np.mean(np.abs(coef), axis=0)
    else:
        coef = np.abs(coef)

    if len(coef) != len(feature_names):
        raise ValueError(
            "Coefficient length does not match feature_names."
        )

    importance_dict = {
        feature: float(score)
        for feature, score in zip(feature_names, coef)
    }

    return dict(
        sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# ==========================================================
# SHAP IMPORTANCE
# ==========================================================

def get_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    sample_size: int = 200
) -> Dict[str, float]:
    """
    Compute SHAP importance scores.

    Returns mean(|SHAP value|) per feature.
    """

    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is not installed. Install via: pip install shap"
        )

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "feature_names length must match X column count."
        )

    X_sample = (
        X.sample(
            n=min(sample_size, len(X)),
            random_state=42
        )
        if len(X) > sample_size
        else X
    )

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(
            model,
            X_sample
        )
    else:
        explainer = shap.Explainer(
            model,
            X_sample
        )

    shap_values = explainer(X_sample)

    if isinstance(shap_values, list):
        shap_values = np.mean(
            np.abs(np.asarray(shap_values)),
            axis=0
        )
    else:
        shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = np.mean(
            np.abs(shap_values),
            axis=2
        )

    importance = np.abs(shap_values).mean(axis=0)

    if importance.ndim > 1:
        importance = importance.mean(axis=0)

    if len(importance) != len(feature_names):
        raise ValueError(
            "SHAP importance length does not match feature_names."
        )

    importance_dict = {
        feature: float(score)
        for feature, score in zip(feature_names, importance)
    }

    return dict(
        sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# ==========================================================
# TOP FEATURES
# ==========================================================

def get_top_features(
    importance_dict: Dict[str, float],
    top_k: int = 10
) -> List[tuple]:
    """
    Return top-k features.
    """

    return list(importance_dict.items())[:top_k]


# ==========================================================
# INSIGHT GENERATION
# ==========================================================

def generate_feature_insights(
    importance_dict: Dict[str, float]
) -> List[str]:
    """
    Generate human-readable insights.
    """

    if not importance_dict:
        return []

    insights: List[str] = []

    items = list(importance_dict.items())

    top_feature, top_score = items[0]

    insights.append(
        f"{top_feature} is the most influential feature "
        f"(importance={top_score:.4f})."
    )

    for feature, score in items[:5]:

        if score >= 0.30:
            level = "high"

        elif score >= 0.10:
            level = "moderate"

        else:
            level = "low"

        insights.append(
            f"{feature} has {level} predictive influence."
        )

    return insights


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

def generate_feature_recommendations(
    importance_dict: Dict[str, float]
) -> List[str]:
    """
    Generate agent-consumable recommendations.
    """

    recommendations: List[str] = []

    low_features = [
        feature
        for feature, score in importance_dict.items()
        if score < 0.01
    ]

    if low_features:
        recommendations.append(
            f"Consider removing low-importance features: "
            f"{', '.join(low_features[:10])}."
        )

    top_features = list(importance_dict.keys())[:3]

    if len(top_features) >= 2:
        recommendations.append(
            "Consider interaction features involving: "
            f"{', '.join(top_features)}."
        )

    if len(low_features) > 5:
        recommendations.append(
            "Feature selection may reduce dimensionality "
            "without significant performance loss."
        )

    return recommendations


# ==========================================================
# MAIN EXPLAINABILITY INTERFACE
# ==========================================================

def explain_model(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
    use_shap: bool = True,
    shap_sample_size: int = 200
) -> Dict[str, Any]:
    """
    Unified explainability API.

    Priority:
    1. SHAP
    2. Coefficients
    3. Feature Importance

    Returns a consistent schema for agents.
    """

    if feature_names is None:
        feature_names = list(X.columns)

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "feature_names length must match X column count."
        )

    explanation_method = None
    importance_scores = None
    attempted_methods: List[Dict[str, str]] = []

    # ----------------------------------
    # SHAP (preferred)
    # ----------------------------------

    if use_shap:
        try:

            importance_scores = get_shap_explanations(
                model=model,
                X=X,
                feature_names=feature_names,
                sample_size=shap_sample_size
            )

            explanation_method = "shap"
            attempted_methods.append(
                {
                    "method": "shap",
                    "status": "success"
                }
            )

        except Exception as exc:
            attempted_methods.append(
                {
                    "method": "shap",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)
                }
            )

    # ----------------------------------
    # COEFFICIENTS
    # ----------------------------------

    if importance_scores is None:

        try:

            if hasattr(model, "coef_"):

                importance_scores = (
                    get_coefficient_importance(
                        model=model,
                        feature_names=feature_names
                    )
                )

                explanation_method = "coefficients"
                attempted_methods.append(
                    {
                        "method": "coefficients",
                        "status": "success"
                    }
                )
            else:
                attempted_methods.append(
                    {
                        "method": "coefficients",
                        "status": "skipped",
                        "error": "Model does not expose coef_."
                    }
                )

        except Exception as exc:
            attempted_methods.append(
                {
                    "method": "coefficients",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)
                }
            )

    # ----------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------

    if importance_scores is None:

        try:

            if hasattr(model, "feature_importances_"):

                importance_scores = (
                    get_feature_importance(
                        model=model,
                        feature_names=feature_names
                    )
                )

                explanation_method = "feature_importance"
                attempted_methods.append(
                    {
                        "method": "feature_importance",
                        "status": "success"
                    }
                )
            else:
                attempted_methods.append(
                    {
                        "method": "feature_importance",
                        "status": "skipped",
                        "error": "Model does not expose feature_importances_."
                    }
                )

        except Exception as exc:
            attempted_methods.append(
                {
                    "method": "feature_importance",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)
                }
            )

    # ----------------------------------
    # FAIL
    # ----------------------------------

    if importance_scores is None:

        raise ValueError(
            f"Explainability not supported for "
            f"{type(model).__name__}. Attempts: "
            f"{attempted_methods}"
        )

    top_features = get_top_features(
        importance_scores,
        top_k=top_k
    )

    insights = generate_feature_insights(
        importance_scores
    )

    recommendations = (
        generate_feature_recommendations(
            importance_scores
        )
    )

    return {
        "explanation_method": explanation_method,
        "importance_scores": importance_scores,
        "top_features": top_features,
        "insights": insights,
        "recommendations": recommendations,
        "attempted_methods": attempted_methods,
    }
