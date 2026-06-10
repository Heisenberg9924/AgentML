from agents.base_agent import BaseAgent

from tools.training_tools import train_model
from tools.evaluation_tools import evaluate_model


class ExperimentAgent(BaseAgent):

    def __init__(self):
        super().__init__("ExperimentAgent")

    def run(
        self,
        state: dict
    ) -> dict:

        self.log(
            "Starting model experimentation..."
        )

        X_train = state["X_train"]
        X_test = state["X_test"]

        y_train = state["y_train"]
        y_test = state["y_test"]

        problem_type = state["problem_type"]

        # -----------------------------------
        # Candidate Models
        # -----------------------------------

        if problem_type == "classification":

            candidate_models = [
                "logistic_regression",
                "random_forest",
                "xgboost",
                "lightgbm",
                "catboost"
            ]

            optimization_metric = "f1_score"

        else:

            candidate_models = [
                "linear_regression",
                "random_forest",
                "xgboost",
                "lightgbm",
                "catboost"
            ]

            optimization_metric = "r2"

        # -----------------------------------
        # Tracking
        # -----------------------------------

        experiment_results = {}

        all_trained_models = {}

        training_metadata = {}

        best_model = None
        best_model_name = None

        best_score = float("-inf")

        # -----------------------------------
        # Train + Evaluate
        # -----------------------------------

        for model_name in candidate_models:

            self.log(
                f"Training {model_name}"
            )

            try:

                model, metadata = train_model(
                    X_train=X_train,
                    y_train=y_train,
                    task_type=problem_type,
                    model_name=model_name
                )

                # -------------------------
                # Predictions
                # -------------------------

                y_pred = model.predict(
                    X_test
                )

                y_proba = None

                if (
                    problem_type == "classification"
                    and hasattr(
                        model,
                        "predict_proba"
                    )
                ):
                    try:

                        y_proba = (
                            model.predict_proba(
                                X_test
                            )
                        )

                    except Exception:

                        y_proba = None

                # -------------------------
                # Evaluation
                # -------------------------

                metrics = evaluate_model(
                    y_true=y_test,
                    y_pred=y_pred,
                    task_type=problem_type,
                    y_proba=y_proba
                )

                experiment_results[
                    model_name
                ] = metrics

                all_trained_models[
                    model_name
                ] = model

                training_metadata[
                    model_name
                ] = metadata

                current_score = metrics[
                    optimization_metric
                ]

                if current_score > best_score:

                    best_score = current_score

                    best_model = model

                    best_model_name = model_name

            except Exception as e:

                self.log(
                    f"{model_name} failed: {e}"
                )

                experiment_results[
                    model_name
                ] = {
                    "error": str(e)
                }
        
        
        print("\nMODEL RESULTS")

        for model_name, metrics in experiment_results.items():
            print(model_name)
            print(metrics)
            print("-" * 50)

        # -----------------------------------
        # Update State
        # -----------------------------------

        state[
            "experiment_results"
        ] = experiment_results

        state[
            "all_trained_models"
        ] = all_trained_models

        state[
            "training_metadata"
        ] = training_metadata

        state[
            "best_model"
        ] = best_model

        state[
            "best_model_name"
        ] = best_model_name

        state[
            "best_score"
        ] = best_score

        self.log(
            f"Best model: "
            f"{best_model_name} "
            f"({best_score:.4f})"
        )

        return state