from langgraph.graph import END, StateGraph
from src.utils.state import GraphState
from src.nodes.retrieve import retrieve
from src.nodes.generate import generate
from src.nodes.grade_documents import grade_documents
from src.nodes.transform_query import transform_query
from src.nodes.web_search import web_search
from src.edges.route_question import route_question
from src.edges.decide_to_generate import decide_to_generate
from src.edges.grade_generation import grade_generation_v_documents_and_question
from langgraph.graph import END, StateGraph


def create_workflow():
    """Create and return the question answering workflow graph."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("route_question", route_question)  # Add this line
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Set the entry point
    workflow.set_entry_point("route_question")

    # Add conditional edges from route_question
    workflow.add_conditional_edges(
        "route_question",
        lambda x: x["next"],  # Use a lambda function to extract the 'next' key
        {
            "web_search": "web_search",
            "retrieve": "retrieve",
        })

    # Rest of the edges
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()
