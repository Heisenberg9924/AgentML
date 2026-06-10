from langgraph.graph import (
    StateGraph,
    END
)

from graph.state import MLState

from graph.nodes import (
    data_node,
    feature_node,
    experiment_node,
    critic_node
)

from graph.router import (
    critic_router
)

# =====================================
# Create Graph
# =====================================

workflow = StateGraph(
    MLState
)

# =====================================
# Add Nodes
# =====================================

workflow.add_node(
    "data",
    data_node
)

workflow.add_node(
    "feature",
    feature_node
)

workflow.add_node(
    "experiment",
    experiment_node
)

workflow.add_node(
    "critic",
    critic_node
)

# =====================================
# Entry Point
# =====================================

workflow.set_entry_point(
    "data"
)

# =====================================
# Normal Flow
# =====================================

workflow.add_edge(
    "data",
    "feature"
)

workflow.add_edge(
    "feature",
    "experiment"
)

workflow.add_edge(
    "experiment",
    "critic"
)

# =====================================
# Conditional Flow
# =====================================

workflow.add_conditional_edges(
    "critic",
    critic_router
)

# =====================================
# Compile
# =====================================

app = workflow.compile()