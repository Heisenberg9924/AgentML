
from agents.base_agent import BaseAgent
import re

from tools.preprocessing_tools import (
    remove_duplicates,
    remove_constant_features,
    engineer_datetime_features,
    handle_missing_values,
    cap_outliers_iqr,
    encode_categorical_features,
    scale_features
)

from tools.dataset_tools import (
    split_features_target,
    train_test_split_dataset
)


class FeatureAgent(BaseAgent):

    def __init__(self, llm):
        super().__init__(
            name="FeatureAgent",
            llm=llm
        )

    def run(
        self,
        state: dict
    ) -> dict:

        self.log(
            "Starting FeatureAgent"
        )

        state.setdefault(
            "agent_logs",
            []
        )

        # =====================================
        # Load Data From State
        # =====================================

        df = state["df"]
        target = state["target_column"]

        self.log(
            f"Target column: {target}"
        )

        # =====================================
        # Duplicate Removal
        # =====================================

        self.log(
            "Removing duplicates"
        )

        df, duplicate_meta = (
            remove_duplicates(df)
        )

        # =====================================
        # Constant Feature Removal
        # =====================================

        self.log(
            "Removing constant features"
        )

        df, constant_meta = (
            remove_constant_features(
                df,
                target
            )
        )

        # =====================================
        # Datetime Feature Engineering
        # =====================================

        self.log(
            "Engineering datetime features"
        )

        df, datetime_meta = (
            engineer_datetime_features(
                df,
                target
            )
        )

        # =====================================
        # Missing Value Handling
        # =====================================

        self.log(
            "Handling missing values"
        )

        df, missing_meta = (
            handle_missing_values(
                df,
                target
            )
        )

        # =====================================
        # Outlier Treatment
        # =====================================

        self.log(
            "Capping outliers"
        )

        df, outlier_meta = (
            cap_outliers_iqr(
                df,
                target_column=target
            )
        )

        # =====================================
        # Split Features & Target
        # =====================================

        self.log(
            "Splitting features and target"
        )

        X, y = split_features_target(
            df,
            target
        )

        # =====================================
        # Train Test Split
        # =====================================

        self.log(
            "Creating train/test split"
        )

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = train_test_split_dataset(
            X,
            y
        )

        # =====================================
        # Encoding
        # =====================================

        self.log(
            "Encoding categorical features"
        )

        (
            X_train,
            X_test,
            encoding_meta
        ) = encode_categorical_features(
            X_train,
            X_test
        )

        encoding_artifacts = (
            encoding_meta["_artifacts"]
        )
        
        # =====================================
        # Sanitize Feature Names
        # =====================================

        self.log(
            "Sanitizing feature names"
)

        X_train.columns = [
            re.sub(r'[^A-Za-z0-9_]', '_', str(col))
             for col in X_train.columns
        ]

        X_test.columns = [
            re.sub(r'[^A-Za-z0-9_]', '_', str(col))
            for col in X_test.columns
]
        print("\nSanitized Feature Names:")
        for col in X_train.columns:
            print(f"  {col}")


        # =====================================
        # Scaling
        # =====================================

        self.log(
            "Scaling numerical features"
        )

        (
            X_train,
            X_test,
            scaler,
            scaling_meta
        ) = scale_features(
            X_train,
            X_test
        )

        # =====================================
        # Build Report
        # =====================================

        preprocessing_report = {

            "duplicates": duplicate_meta,

            "constant_features":
            constant_meta,

            "datetime_features":
            datetime_meta,

            "missing_values":
            missing_meta,

            "outliers":
            outlier_meta,

            "encoding":
            encoding_meta,

            "scaling":
            scaling_meta
        }

        # =====================================
        # Agent Logs
        # =====================================

        state["agent_logs"].append(
            {
                "agent": "FeatureAgent",
                "action": "preprocessing",
                "result": preprocessing_report
            }
        )

        # =====================================
        # Update State
        # =====================================

        state["df_processed"] = df

        state["X_train"] = X_train
        state["X_test"] = X_test

        state["y_train"] = y_train
        state["y_test"] = y_test

        state["preprocessing_report"] = (
            preprocessing_report
        )

        state["imputers"] = (
            missing_meta["imputers"]
        )

        state["encoders"] = (
            encoding_artifacts[
                "one_hot_encoders"
            ]
        )

        state["frequency_maps"] = (
            encoding_artifacts[
                "frequency_maps"
            ]
        )

        state["scaler"] = scaler

        self.log(
            "FeatureAgent completed successfully"
        )

        return state