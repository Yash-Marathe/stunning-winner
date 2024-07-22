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

def create_workflow():
    """Create and return the question answering workflow graph."""
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
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