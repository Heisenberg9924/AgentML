from langgraph.graph import END


MAX_ITERATIONS = 3


def critic_router(state):

    retry_required = (
        state["critic_report"]
        ["retry_required"]
    )

    iteration_count = (
        state.get(
            "iteration_count",
            0
        )
    )

    if (
        retry_required
        and iteration_count < MAX_ITERATIONS
    ):
        return "feature"

    return END