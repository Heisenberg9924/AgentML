from graph.workflow import app
from graph.state import MLState

initial_state : MLState = {
    "dataset_path": "data/Titanic-Dataset.csv",
    "iteration_count": 0
}
print("Starting Workflow...")
result = app.invoke(initial_state)
print("Workflow Completed.")
print(result["critic_report"])
