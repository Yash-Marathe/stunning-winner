from src.llm.models import rag_chain

def generate(state):
    """Generate an answer based on retrieved documents and question."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "history": state["history"]
    }