# graph/state.py

from typing import TypedDict, Any


class MLState(TypedDict, total=False):

    # Dataset

    dataset_path: str
    df: Any
    df_processed: Any

    # Target

    target_column: str
    problem_type: str

    # Split Data

    X_train: Any
    X_test: Any

    y_train: Any
    y_test: Any

    # Reports

    data_report: dict
    feature_report: dict

    experiment_results: dict

    explainability_report: dict

    critic_report: dict

    # Best Model

    best_model: Any
    best_model_name: str
    best_score: float

    # Artifacts

    imputers: dict
    encoders: dict
    frequency_maps: dict

    scaler: Any

    # Agent Logs

    agent_logs: list

    # Loop Control

    iteration_count: int