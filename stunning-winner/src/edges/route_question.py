from src.llm.models import question_router


def route_question(state):
    """Route the question to either web search or RAG pipeline."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        next_step = "web_search"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        next_step = "retrieve"
    else:
        raise ValueError(f"Unknown datasource: {source.datasource}")

    return {**state, "next": next_step}
