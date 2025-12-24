"""
State definitions for the multi-intent query processing graph
"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated

from .enums import GraphStatus, IntentType, QueryStatus


class QueryAnalysis(TypedDict):
    """
    Single query analysis result.

    Attributes:
        sub_question: The decomposed sub-question to be executed
        intent: Type of query (SQL_QUERY or SUMMARY_SEARCH)
        required_files: List of file names required for this query
        join_key: Column name for joining multiple files (if applicable)
        depends_on_index: Index of the query this depends on (-1 if independent)
    """

    sub_question: str
    intent: IntentType
    required_files: List[str]
    join_key: str
    depends_on_index: int  # -1 indicates no dependency, >= 0 indicates dependency index


class QueryResult(TypedDict):
    """
    Result of a single query execution.

    Attributes:
        sub_question: The sub-question that was executed
        intent: Type of query (SQL_QUERY or SUMMARY_SEARCH)
        status: Execution status (success, error, or placeholder)
        sql_query: Generated SQL query (None for non-SQL queries)
        sql_explanation: Explanation of the SQL query
        results: JSON string of query results
        execution_duration: Time taken to execute in seconds
        error: Error message if execution failed
    """

    sub_question: str
    intent: IntentType
    status: QueryStatus
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


class GraphState(TypedDict, total=False):
    """
    Main state for the LangGraph workflow.

    This state is passed through all nodes in the graph and accumulates
    information about query analysis, execution, and results.
    """

    # Input fields
    user_question: str  # Original user question
    config: Any  # Application configuration object
    global_catalog_dict: Dict[str, Any]  # Catalog of available data files
    catalog_schema: str  # JSON schema of the catalog
    enable_debug: bool  # Whether to enable debug logging

    # Analysis results
    analyses: List[QueryAnalysis]  # List of decomposed sub-questions
    total_questions: int  # Total number of sub-questions
    independent_count: int  # Number of independent queries
    dependent_count: int  # Number of dependent queries

    # Execution tracking
    executed_results: Dict[int, QueryResult]  # Results indexed by query number
    remaining_indices: List[int]  # Indices of queries not yet executed
    current_batch: List[int]  # Indices of queries in current execution batch

    # Final results
    final_results: List[QueryResult]  # All results in original order
    total_duration: float  # Total execution time in seconds
    status: GraphStatus  # Overall execution status
    error: Optional[str]  # Error message if execution failed
    final_summary: Optional[FinalSummary]  # Generated summary of all results

    # Debugging and logging
    messages: Annotated[List[str], add_messages]  # Log messages

    # Monitoring
    metrics_collector: Optional[Any]  # MetricsCollector instance
