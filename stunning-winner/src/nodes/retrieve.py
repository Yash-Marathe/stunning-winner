from src.retriever.vectorstore import get_retriever

def retrieve(state):
    """Retrieve documents relevant to the question."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = get_retriever().invoke(question)
    return {"documents": documents, "question": question, "history": state["history"]}