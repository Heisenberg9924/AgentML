
from agents.base_agent import BaseAgent

from tools.dataset_tools import (
    load_dataset,
    dataset_summary,
    missing_value_report,
    detect_problem_type
)


class DataAgent(BaseAgent):

    def __init__(self, llm):
        super().__init__(
            name="DataAgent",
            llm=llm
        )

    def _detect_target_with_llm(
        self,
        columns
    ) -> str:

        prompt = f"""
        Dataset Columns:

        {columns}

        Predict the most likely target column.

        Return ONLY the column name.
        """

        response = self.invoke(prompt)

        return str(response).strip()

    def detect_obvious_target(
        self,
        columns
    ):

        common_targets = [
            "target",
            "label",
            "class",
            "y",
            "outcome",
            "survived",
            "species",
            "saleprice",
            "price",
            "diagnosis",
            "churn",
            "default"
        ]

        cols_lower = [
            col.lower()
            for col in columns
        ]

        for target in common_targets:

            if target in cols_lower:
                return columns[
                    cols_lower.index(target)
                ]

        return None

    def run(
        self,
        state: dict
    ) -> dict:

        self.log(
            "Starting DataAgent"
        )

        state.setdefault(
            "agent_logs",
            []
        )

        # -------------------------
        # Load Dataset
        # -------------------------

        self.log(
            f"Loading dataset from: {state['dataset_path']}"
        )

        df = load_dataset(
            state["dataset_path"]
        )

        self.log(
            f"Dataset loaded successfully. Shape={df.shape}"
        )

        # -------------------------
        # Target Detection
        # -------------------------

        target = state.get(
            "target_column"
        )

        if target is not None:

            self.log(
                f"Using user-provided target: {target}"
            )

        if target is None:

            target = self.detect_obvious_target(
                df.columns
            )

            if target is not None:

                self.log(
                    f"Heuristic target detection selected: {target}"
                )

                state["agent_logs"].append(
                    {
                        "agent": "DataAgent",
                        "action": "target_detection",
                        "method": "heuristic",
                        "result": target
                    }
                )

        if target is None:

            self.log(
                "Heuristic target detection failed. Using LLM."
            )

            target = self._detect_target_with_llm(
                list(df.columns)
            )

            self.log(
                f"LLM selected target column: {target}"
            )

            state["agent_logs"].append(
                {
                    "agent": "DataAgent",
                    "action": "target_detection",
                    "method": "llm",
                    "result": target
                }
            )

        # -------------------------
        # Dataset Analysis
        # -------------------------

        self.log(
            "Generating dataset summary"
        )

        summary = dataset_summary(
            df
        )

        self.log(
            "Generating missing value report"
        )

        missing_report = missing_value_report(
            df
        )

        problem_type = detect_problem_type(
            df,
            target
        )

        self.log(
            f"Detected problem type: {problem_type}"
        )

        state["agent_logs"].append(
            {
                "agent": "DataAgent",
                "action": "problem_type_detection",
                "result": problem_type
            }
        )

        # -------------------------
        # Update State
        # -------------------------

        self.log(
            "Updating shared state"
        )

        state["df"] = df
        state["target_column"] = target
        state["problem_type"] = problem_type
        state["dataset_summary"] = summary
        state["missing_report"] = missing_report

        self.log(
            "DataAgent completed successfully"
        )

        return state