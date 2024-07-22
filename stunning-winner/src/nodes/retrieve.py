from src.retriever.vectorstore import get_retriever
from langchain.schema import Document


def retrieve(state):
    """Retrieve documents relevant to the question."""
    print("---RETRIEVE---")
    question = state["question"]
    retrieved_docs = get_retriever().invoke(question)

    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in retrieved_docs
    ]

    # Directly return the dictionary, Langgraph will handle the transition
    return {
        "documents": documents,
        "question": question,
        "history": state["history"]
    }
