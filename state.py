"""
State definitions for the multi-intent query processing graph
"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


class QueryAnalysis(TypedDict):
    """Single query analysis"""

    sub_question: str
    intent: str  # "SQL_QUERY" or "SUMMARY_SEARCH"
    required_files: List[str]
    join_key: str
    depends_on_index: int


class QueryResult(TypedDict):
    """Result of a single query execution"""

    sub_question: str
    intent: str
    status: str  # "success", "error", "placeholder"
    sql_query: Optional[str]
    sql_explanation: Optional[str]
    results: str
    execution_duration: float
    error: Optional[str]


class FinalSummary(TypedDict):
    """The structure for the final narrative summary."""

    summary_text: str
    tables: List[Dict[str, Any]]
    has_tables: bool
    error: Optional[str]  # Added for completeness/error handling


class GraphState(TypedDict):
    """Main state for the LangGraph workflow"""

    # Input
    user_question: str
    config: Any
    global_catalog_dict: Dict[str, Any]
    catalog_schema: str
    enable_debug: bool

    # Analysis
    analyses: List[QueryAnalysis]
    total_questions: int
    independent_count: int
    dependent_count: int

    # Execution tracking
    executed_results: Dict[int, QueryResult]
    remaining_indices: List[int]
    current_batch: List[int]

    # Results
    final_results: List[QueryResult]
    total_duration: float
    status: str  # "success", "partial_success", "error"
    error: Optional[str]
    final_summary: Optional[FinalSummary]

    # Messages for debugging
    messages: Annotated[List[str], add_messages]
