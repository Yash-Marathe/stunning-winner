from src.llm.models import question_rewriter

def transform_query(state):
    """Transform the query to improve retrieval results."""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {
        "documents": documents,
        "question": better_question,
        "history": state["history"]
    }