from langchain_openai import ChatOpenAI
from langchain import hub
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

# Data models
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# LLM and Prompts
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_router = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to health insurance, its terms and offers from specific providers.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router),
        ("human", "{question}"),
    ]
)

system_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallucination),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_answer),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

# Structured LLMs
structured_llm_router = llm.with_structured_output(RouteQuery)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

# Pipelines
question_router = route_prompt | structured_llm_router
retrieval_grader = grade_prompt | structured_llm_grader
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader
answer_grader = answer_prompt | structured_llm_answer_grader
question_rewriter = re_write_prompt | llm | StrOutputParser()

# RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# RAG generation chain
rag_chain = rag_prompt | llm | StrOutputParser()