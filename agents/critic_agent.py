from agents.base_agent import BaseAgent

import json


class CriticAgent(BaseAgent):

    def __init__(self, llm):
        super().__init__("CriticAgent")
        self.llm = llm

    def run(self, state):

        experiment_results = state["experiment_results"]

        best_model_name = state["best_model_name"]
        best_score = state["best_score"]

        problem_type = state["problem_type"]

        explainability_report = state.get(
            "explainability_report",
            {}
        )

        strengths = []
        weaknesses = []
        recommendations = []
        model_insights = []

        # =====================================================
        # PERFORMANCE ANALYSIS
        # =====================================================

        best_metrics = experiment_results.get(
            "best_metrics",
            {}
        )

        if problem_type == "classification":

            accuracy = best_metrics.get("accuracy")
            f1_score = best_metrics.get("f1_score")

            if (
                accuracy is not None
                and f1_score is not None
                and accuracy - f1_score > 0.15
            ):

                weaknesses.append(
                    "Accuracy is significantly higher than F1 score."
                )

                recommendations.append(
                    "Investigate class imbalance."
                )

            if best_score >= 0.90:

                strengths.append(
                    "Excellent predictive performance."
                )

            elif best_score >= 0.80:

                strengths.append(
                    "Strong predictive performance."
                )

            else:

                weaknesses.append(
                    "Model performance remains moderate."
                )

                recommendations.append(
                    "Additional feature engineering recommended."
                )

        else:

            if best_score >= 0.80:

                strengths.append(
                    "Model explains most target variance."
                )

            elif best_score >= 0.60:

                strengths.append(
                    "Model captures useful predictive patterns."
                )

            else:

                weaknesses.append(
                    "Model explains limited target variance."
                )

                recommendations.append(
                    "Create interaction features."
                )

                recommendations.append(
                    "Investigate nonlinear relationships."
                )

        # =====================================================
        # MODEL COMPARISON
        # =====================================================

        all_results = experiment_results.get(
            "all_results",
            {}
        )

        if len(all_results) >= 2:

            sorted_models = sorted(
                all_results.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )

            best_model = sorted_models[0][0]
            second_model = sorted_models[1][0]

            model_insights.append(
                f"{best_model} outperformed {second_model}."
            )

            boosting_models = [
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ]

            top_models = [
                model_name
                for model_name, _
                in sorted_models[:3]
            ]

            if any(
                model in boosting_models
                for model in top_models
            ):

                model_insights.append(
                    "Boosting models consistently outperform simpler models, indicating nonlinear relationships."
                )

        # =====================================================
        # EXPLAINABILITY ANALYSIS
        # =====================================================

        if explainability_report:

            top_features = explainability_report.get(
                "top_features",
                []
            )

            if top_features:

                top_feature_name = top_features[0][0]
                top_feature_score = top_features[0][1]

                strengths.append(
                    f"{top_feature_name} is a highly influential predictor."
                )

                if len(top_features) > 1:

                    second_score = top_features[1][1]

                    if (
                        second_score > 0
                        and top_feature_score / second_score > 5
                    ):

                        weaknesses.append(
                            "Predictions are heavily dependent on a single feature."
                        )

                        recommendations.append(
                            "Create interaction features to diversify predictive signals."
                        )

            importance_scores = explainability_report.get(
                "importance_scores",
                {}
            )

            low_importance_features = [
                feature
                for feature, score
                in importance_scores.items()
                if score < 0.01
            ]

            if len(low_importance_features) > 5:

                weaknesses.append(
                    "Several features contribute little predictive value."
                )

                recommendations.append(
                    "Apply feature selection."
                )

            recommendations.extend(
                explainability_report.get(
                    "recommendations",
                    []
                )
            )

        # =====================================================
        # RETRY DECISION
        # =====================================================

        if problem_type == "classification":

            retry_required = best_score < 0.85

        else:

            retry_required = best_score < 0.70

        next_action = (
            "FeatureAgent"
            if retry_required
            else "END"
        )

        # =====================================================
        # LLM REFLECTION
        # =====================================================

        llm_report = {}

        try:

            prompt = f"""
You are a Senior Machine Learning Scientist.

Analyze the following ML experiment.

Return ONLY valid JSON.

Required schema:

{{
    "observations": [],
    "weaknesses": [],
    "recommendations": [],
    "retry_recommended": true
}}

Problem Type:
{problem_type}

Best Model:
{best_model_name}

Best Score:
{best_score}

Experiment Results:
{experiment_results}

Explainability Report:
{explainability_report}

Rule Based Findings:

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Recommendations:
{recommendations}

Model Insights:
{model_insights}
"""

            response = self.llm.invoke(prompt)
            
            print("LLM Response:", response.content)

            llm_report = json.loads(
                response.content
            )

        except Exception as e:

            llm_report = {
                "observations": [],
                "weaknesses": [],
                "recommendations": [],
                "retry_recommended": retry_required,
                "error": str(e)
            }

        # =====================================================
        # MERGE RULES + LLM
        # =====================================================

        recommendations.extend(
            llm_report.get(
                "recommendations",
                []
            )
        )

        weaknesses.extend(
            llm_report.get(
                "weaknesses",
                []
            )
        )

        # =====================================================
        # FINAL REPORT
        # =====================================================

        critic_report = {

            "best_model": best_model_name,

            "best_score": best_score,

            "strengths": list(
                set(strengths)
            ),

            "weaknesses": list(
                set(weaknesses)
            ),

            "recommendations": list(
                set(recommendations)
            ),

            "model_insights": model_insights,

            "llm_observations": llm_report.get(
                "observations",
                []
            ),

            "llm_report": llm_report,

            "retry_required": retry_required,

            "next_action": next_action
        }

        state["critic_report"] = critic_report
        
        #----------------------------------------------
        # Iteration Control
        #----------------------------------------------
        
        state['iteration_count'] = (
            state.get('iteration_count', 0) + 1
        )

        return state