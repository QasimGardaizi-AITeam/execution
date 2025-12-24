"""
LangGraph nodes for query processing workflow
"""

import json
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from openai import AzureOpenAI

from .chains import (
    analyze_query_chain,
    generate_final_summary_chain,
    generate_sql_chain,
)
from .enums import GraphStatus, IntentType, QueryStatus
from .logging_config import get_logger, log_query_execution
from .state import GraphState, QueryAnalysis, QueryResult
from .tools import (
    build_path_mapping,
    df_to_json_result,
    execute_duckdb_query,
    extract_schema_from_catalog,
    read_top_rows_duckdb,
    validate_state,
)
from .validation import ValidationError, validate_file_names

logger = get_logger()


class SqlExecutionError(Exception):
    """Custom exception raised when DuckDB returns an error DataFrame indicating a failed execution."""

    pass


# HELPER FUNCTIONS FOR execute_sql_query_node
def _handle_client_init_error(state: GraphState, error_msg: str) -> None:
    """Handles the state update when the LLM client fails to initialize."""
    logger.error(error_msg)
    state["messages"].append(f"[ERROR] {error_msg}")

    # Mark all queries in batch as failed
    for idx in state["current_batch"]:
        analysis = state["analyses"][idx]
        state["executed_results"][idx] = QueryResult(
            sub_question=analysis["sub_question"],
            intent=analysis["intent"],
            status=QueryStatus.ERROR.value,
            sql_query=None,
            sql_explanation=None,
            results=json.dumps({"Error": error_msg}),
            execution_duration=0.0,
            error=error_msg,
        )

        # Record in metrics if available
        metrics = state.get("metrics_collector")
        if metrics:
            metrics.record_query_execution(
                query_index=idx,
                sub_question=analysis["sub_question"],
                intent=analysis["intent"],
                status="error",
                execution_duration=0.0,
                error=error_msg,
            )

    # Remove from remaining indices
    for idx in state["current_batch"]:
        if idx in state["remaining_indices"]:
            state["remaining_indices"].remove(idx)

    # Clear batch
    state["current_batch"] = []


def _get_dependency_result(
    state: GraphState, analysis: QueryAnalysis, attempt: int
) -> str:
    """Gets the execution result of a dependent query if available."""
    dependency_result = ""
    dep_idx = analysis["depends_on_index"]

    if dep_idx >= 0:
        dep_result = state["executed_results"].get(dep_idx)
        if dep_result and dep_result["status"] == QueryStatus.SUCCESS.value:
            dependency_result = dep_result["results"]
            if attempt == 0 or state["enable_debug"]:
                logger.info(f"Using result from Q{dep_idx + 1}")
    return dependency_result


def _get_dynamic_sample_data(
    state: GraphState, analysis: QueryAnalysis, path_map: Dict[str, str], attempt: int
) -> str:
    """Reads and formats sample data from the first required file."""
    df_sample = "No sample data available or required."

    if not path_map:
        return df_sample

    # SECURITY FIX: Validate required_files list before accessing
    if not analysis["required_files"]:
        logger.warning("No files specified in required_files")
        return "No files specified for analysis."

    # Identify the first file and its URI
    first_file_name = analysis["required_files"][0]
    first_uri = path_map.get(first_file_name) or list(path_map.values())[0]

    if attempt == 0 or state["enable_debug"]:
        logger.debug(f"Reading sample data from: {first_uri}")

    # Read the data using the utility function
    df_sample_markdown = read_top_rows_duckdb(first_uri, state["config"])

    df_sample = (
        f"--- Actual Sample Rows from '{first_file_name}' ---\n" + df_sample_markdown
    )
    if attempt == 0 or state["enable_debug"]:
        logger.debug("Successfully read sample rows")

    return df_sample


def _get_query_context(
    state: GraphState, analysis: QueryAnalysis, attempt: int
) -> Tuple[str, Dict[str, str], str, str]:
    """Prepares all necessary context for SQL generation."""

    # 1. Get dependency result
    dependency_result = _get_dependency_result(state, analysis, attempt)

    # 2. Extract schema
    parquet_schema, _ = extract_schema_from_catalog(
        analysis["required_files"], state["global_catalog_dict"]
    )

    # 3. Build path map
    path_map = build_path_mapping(
        analysis["required_files"], state["global_catalog_dict"]
    )

    # 4. Dynamic sample data logic
    df_sample = _get_dynamic_sample_data(state, analysis, path_map, attempt)

    return dependency_result, path_map, df_sample, parquet_schema


def _run_query_with_self_healing(
    llm_client: AzureOpenAI, state: GraphState, analysis: QueryAnalysis
) -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame], Optional[str]]:
    """Executes the SQL generation and retry loop."""

    sql_query = None
    explanation = None
    result_df = None
    previous_error_msg = None
    error_msg = None
    max_attempts = 2

    for attempt in range(max_attempts):

        current_sql_query = None
        current_explanation = None

        try:
            if attempt > 0:
                logger.warning(
                    f"FAILED. ATTEMPTING SELF-HEALING FALLBACK (Try {attempt + 1}/{max_attempts})"
                )

            # 1. Prepare context for LLM
            dependency_result, path_map, df_sample, parquet_schema = _get_query_context(
                state, analysis, attempt
            )

            # 2. Generate SQL (passes error for self-healing)
            current_sql_query, current_explanation = generate_sql_chain(
                llm_client=llm_client,
                deployment_name=state["config"].azure_openai.llm_deployment_name,
                user_query=analysis["sub_question"],
                parquet_schema=parquet_schema,
                df_sample=df_sample,
                path_map=path_map,
                semantic_context=dependency_result,
                error_message=previous_error_msg,
            )

            logger.info(f"Generated SQL: {current_sql_query[:100]}...")
            logger.debug(f"Explanation: {current_explanation}")

            # 3. Execute query
            result_df = execute_duckdb_query(current_sql_query, state["config"])

            # 4. Check for execution errors (DuckDB returns a DataFrame with an 'Error' column)
            if "Error" in result_df.columns:
                current_error_msg = result_df["Error"].iloc[0]
                raise SqlExecutionError(f"DuckDB Execution Error: {current_error_msg}")

            # If successful, assign final values and break the retry loop
            sql_query = current_sql_query
            explanation = current_explanation
            error_msg = None
            break

        except Exception as e:
            previous_error_msg = str(e)
            sql_query = current_sql_query
            explanation = current_explanation
            error_msg = previous_error_msg

            if attempt == max_attempts - 1:
                logger.error(
                    f"FINAL ERROR: Failed after {max_attempts} attempts: {error_msg}"
                )
            else:
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

    return sql_query, explanation, result_df, error_msg


def _log_and_record_result(
    state: GraphState,
    idx: int,
    analysis: QueryAnalysis,
    sql_query: Optional[str],
    explanation: Optional[str],
    result_df: Optional[pd.DataFrame],
    error_msg: Optional[str],
    execution_duration: float,
) -> None:
    """Logs the final result and records it in the state."""

    # Determine status and row_count first
    if error_msg:
        status = QueryStatus.ERROR.value
        row_count = None

        # Record error result
        state["executed_results"][idx] = QueryResult(
            sub_question=analysis["sub_question"],
            intent=analysis["intent"],
            status=status,
            sql_query=sql_query,
            sql_explanation=explanation,
            results=json.dumps({"Error": error_msg}),
            execution_duration=execution_duration,
            error=error_msg,
        )

    elif result_df is not None:
        status = QueryStatus.SUCCESS.value
        row_count = result_df.shape[0]

        # Record success result
        state["executed_results"][idx] = QueryResult(
            sub_question=analysis["sub_question"],
            intent=analysis["intent"],
            status=status,
            sql_query=sql_query,
            sql_explanation=explanation,
            results=df_to_json_result(result_df),
            execution_duration=execution_duration,
            error=None,
        )

        # Log success
        logger.info(f"Executed in {execution_duration:.2f}s")
        logger.info(f"Results: {row_count} rows x {result_df.shape[1]} cols")

        if state["enable_debug"] and not result_df.empty:
            logger.debug("\n" + result_df.head(5).to_markdown(index=False))

    else:
        # Shouldn't happen, but handle it
        status = QueryStatus.ERROR.value
        row_count = None

    # Record metrics if collector available
    metrics = state.get("metrics_collector")
    if metrics:
        metrics.record_query_execution(
            query_index=idx,
            sub_question=analysis["sub_question"],
            intent=analysis["intent"],
            status=status,
            execution_duration=execution_duration,
            row_count=row_count,
            error=error_msg,
        )


def validate_input_node(state: GraphState) -> GraphState:
    """
    Validate input state and initialize fields.
    """

    logger.info("NODE: Validate Input")

    # Validate required fields
    is_valid, error_msg = validate_state(state)
    if not is_valid:
        state["error"] = error_msg
        state["status"] = GraphStatus.ERROR.value
        state["messages"] = [f"[ERROR] Validation failed: {error_msg}"]
        return state

    # Initialize fields
    state.setdefault("enable_debug", False)
    state.setdefault("executed_results", {})
    state.setdefault("remaining_indices", [])
    state.setdefault("current_batch", [])
    state.setdefault("final_results", [])
    state.setdefault("messages", [])
    state.setdefault("error", None)
    state.setdefault("final_summary", None)

    state["messages"].append("[SUCCESS] Input validation passed")
    logger.info("[SUCCESS] Input validated")

    return state


def analyze_query_node(state: GraphState) -> GraphState:
    """
    Analyze user query and decompose into sub-questions.
    """

    logger.info("NODE: Analyze Query")

    try:
        llm_client = AzureOpenAI(
            api_key=state["config"].azure_openai.llm_api_key,
            azure_endpoint=state["config"].azure_openai.llm_endpoint,
            api_version=state["config"].azure_openai.llm_api_version,
        )

        analyses_list, usage = analyze_query_chain(
            llm_client=llm_client,
            user_question=state["user_question"],
            deployment_name=state["config"].azure_openai.llm_deployment_name,
            catalog_schema=state["catalog_schema"],
        )

        # Convert to QueryAnalysis objects
        state["analyses"] = [
            QueryAnalysis(
                sub_question=item["sub_question"],
                intent=item["intent"],
                required_files=item["required_files"],
                join_key=item.get("join_key", ""),
                depends_on_index=item.get("depends_on_index", -1),
            )
            for item in analyses_list
        ]

        # Validate file names
        for idx, analysis in enumerate(state["analyses"]):
            try:
                analysis["required_files"] = validate_file_names(
                    analysis["required_files"], state["global_catalog_dict"]
                )
            except ValidationError as e:
                logger.warning(f"Q{idx + 1} file validation failed: {e}")

        state["total_questions"] = len(state["analyses"])
        state["independent_count"] = sum(
            1 for a in state["analyses"] if a["depends_on_index"] == -1
        )
        state["dependent_count"] = state["total_questions"] - state["independent_count"]

        # Initialize remaining indices
        state["remaining_indices"] = list(range(state["total_questions"]))

        # Log analysis
        msg = f"[SUCCESS] Analyzed query into {state['total_questions']} sub-questions"
        state["messages"].append(msg)
        logger.info(msg)

        if usage:
            logger.info(f"Tokens used: {usage['total_tokens']}")

        for idx, analysis in enumerate(state["analyses"]):
            logger.info(f"--- Question {idx + 1} ---")
            logger.info(f"Sub-Question: {analysis['sub_question']}")
            logger.info(f"Intent: {analysis['intent']}")
            logger.info(f"Files: {', '.join(analysis['required_files'])}")
            if analysis["depends_on_index"] >= 0:
                logger.info(f"⚠️  Depends on Q{analysis['depends_on_index'] + 1}")

        return state

    except Exception as e:
        error_msg = f"Query analysis failed: {str(e)}"
        state["error"] = error_msg
        state["status"] = GraphStatus.ERROR.value
        state["messages"].append(f"[ERROR] {error_msg}")
        logger.error(f"[ERROR] {error_msg}")

        # Fallback: treat as single query
        state["analyses"] = [
            QueryAnalysis(
                sub_question=state["user_question"],
                intent="SQL_QUERY",
                required_files=["*"],
                join_key="",
                depends_on_index=-1,
            )
        ]
        state["total_questions"] = 1
        state["independent_count"] = 1
        state["dependent_count"] = 0
        state["remaining_indices"] = [0]

        return state


def identify_ready_queries_node(state: GraphState) -> GraphState:
    """
    Identify queries that are ready to execute (dependencies satisfied).
    """

    logger.info("NODE: Identify Ready Queries")

    ready = []
    for idx in state["remaining_indices"]:
        analysis = state["analyses"][idx]
        dep_idx = analysis["depends_on_index"]

        # Ready if independent or dependency already executed
        if dep_idx == -1 or dep_idx in state["executed_results"]:
            ready.append(idx)

    if not ready and state["remaining_indices"]:
        # Circular dependency detected
        error_msg = "Circular dependency detected"
        state["error"] = error_msg
        state["messages"].append(f"[ERROR] {error_msg}")
        logger.error(f"[ERROR] {error_msg}")

        # Mark remaining as errors
        for idx in state["remaining_indices"]:
            analysis = state["analyses"][idx]
            state["executed_results"][idx] = QueryResult(
                sub_question=analysis["sub_question"],
                intent=analysis["intent"],
                status=QueryStatus.ERROR.value,
                sql_query=None,
                sql_explanation=None,
                results=json.dumps({"Error": "Circular dependency"}),
                execution_duration=0.0,
                error="Circular dependency",
            )

        state["remaining_indices"] = []

    state["current_batch"] = ready

    msg = f"[SUCCESS] Identified {len(ready)} ready queries"
    state["messages"].append(msg)
    logger.info(msg)

    return state


def execute_sql_query_node(state: GraphState) -> GraphState:
    """
    Execute SQL queries in current batch with LLM-based self-healing fallback.
    """

    logger.info(f"NODE: Execute SQL Queries (Batch: {len(state['current_batch'])})")

    # Track which indices we process
    processed_indices = []

    try:
        llm_client = AzureOpenAI(
            api_key=state["config"].azure_openai.llm_api_key,
            azure_endpoint=state["config"].azure_openai.llm_endpoint,
            api_version=state["config"].azure_openai.llm_api_version,
        )
    except Exception as e:
        error_msg = f"Failed to initialize LLM client: {str(e)}"
        _handle_client_init_error(state, error_msg)
        return state

    for idx in state["current_batch"]:
        analysis = state["analyses"][idx]

        # Skip non-SQL queries
        if analysis["intent"] != "SQL_QUERY":
            continue

        processed_indices.append(idx)
        logger.info(f"--- Processing Q{idx + 1}: {analysis['sub_question']} ---")

        start_time = time.time()

        # Execute query with self-healing attempts
        sql_query, explanation, result_df, error_msg = _run_query_with_self_healing(
            llm_client, state, analysis
        )

        execution_duration = time.time() - start_time

        # Log and record the final result (success or error)
        _log_and_record_result(
            state,
            idx,
            analysis,
            sql_query,
            explanation,
            result_df,
            error_msg,
            execution_duration,
        )

    # Remove processed indices from remaining
    for idx in processed_indices:
        if idx in state["remaining_indices"]:
            state["remaining_indices"].remove(idx)

    return state


def execute_summary_search_node(state: GraphState) -> GraphState:
    """
    Execute summary search queries in current batch (placeholder).
    """

    logger.info(
        f"NODE: Execute Summary Searches (Batch: {len(state['current_batch'])})"
    )

    # Track which indices we process
    processed_indices = []

    for idx in state["current_batch"]:
        analysis = state["analyses"][idx]

        # Skip non-summary queries
        if analysis["intent"] != "SUMMARY_SEARCH":
            continue

        processed_indices.append(idx)
        logger.info(f"--- Processing Q{idx + 1}: {analysis['sub_question']} ---")
        logger.info("[PLACEHOLDER] Summary search not implemented")

        # Get dependency result if needed
        dependency_result = ""
        if analysis["depends_on_index"] >= 0:
            dep_result = state["executed_results"].get(analysis["depends_on_index"])
            if dep_result and dep_result["status"] == QueryStatus.SUCCESS.value:
                dependency_result = dep_result["results"]
                logger.info(f"→ Using result from Q{analysis['depends_on_index'] + 1}")

        # Create placeholder result
        state["executed_results"][idx] = QueryResult(
            sub_question=analysis["sub_question"],
            intent=analysis["intent"],
            status=QueryStatus.PLACEHOLDER.value,
            sql_query=None,
            sql_explanation=None,
            results=json.dumps(
                {
                    "message": "Summary search functionality coming soon",
                    "query": analysis["sub_question"],
                    "files": analysis["required_files"],
                    "dependency_context": (
                        dependency_result[:200] if dependency_result else None
                    ),
                }
            ),
            execution_duration=0.0,
            error=None,
        )

        logger.info("[PLACEHOLDER] Marked as complete")

    # Remove processed indices from remaining
    for idx in processed_indices:
        if idx in state["remaining_indices"]:
            state["remaining_indices"].remove(idx)

    # Clear current batch after processing
    state["current_batch"] = []

    return state


def generate_final_summary_node(
    state: GraphState,
) -> Dict[str, Any]:
    """
    Generate final summary combining all query results.
    Uses LLM to create narrative summary with optional tables.
    """

    logger.info("NODE: Generate Final Summary")

    try:
        llm_client = AzureOpenAI(
            api_key=state["config"].azure_openai.llm_api_key,
            azure_endpoint=state["config"].azure_openai.llm_endpoint,
            api_version=state["config"].azure_openai.llm_api_version,
        )
        successful_results = [
            r
            for r in state["final_results"]
            if r["status"] in ["success", "placeholder"]
        ]

        if not successful_results:
            logger.info("[SKIP] No successful results to summarize")
            # --- CHANGE 1: Return explicit update for no results ---
            return {
                "final_summary": {
                    "summary_text": "No results were successfully generated to summarize.",
                    "tables": [],
                    "has_tables": False,
                    "error": None,
                }
            }

        # Generate summary
        summary_result = generate_final_summary_chain(
            llm_client=llm_client,
            deployment_name=state["config"].azure_openai.llm_deployment_name,
            user_question=state["user_question"],
            query_results=successful_results,
        )

        logger.info("[SUCCESS] Generated final summary")
        logger.info(f"Has tables: {summary_result['has_tables']}")
        logger.info(f"Number of tables: {len(summary_result['tables'])}")

        if state["enable_debug"]:
            logger.debug("\n--- Summary Preview ---")
            logger.debug(summary_result["summary_text"][:500] + "...")

        # --- CHANGE 2: Return explicit update on success ---
        return {
            "final_summary": summary_result,
        }

    except Exception as e:
        error_msg = f"Failed to generate summary: {str(e)}"
        state["messages"].append(f"[ERROR] {error_msg}")
        logger.error(f"[ERROR] {error_msg}")
        logger.error(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback

        logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")

        # --- CHANGE 3: Return explicit update on error ---
        return {
            "final_summary": {
                "summary_text": f"Error generating summary: {error_msg}",
                "tables": [],
                "has_tables": False,
                "error": error_msg,
            }
        }


def finalize_results_node(state: GraphState) -> GraphState:
    """
    Finalize and organize all results.
    """

    logger.info("NODE: Finalize Results")

    # Build final results in original order
    state["final_results"] = []
    for idx in range(state["total_questions"]):
        if idx in state["executed_results"]:
            state["final_results"].append(state["executed_results"][idx])
        else:
            # Shouldn't happen, but handle missing results
            analysis = state["analyses"][idx]
            state["final_results"].append(
                QueryResult(
                    sub_question=analysis["sub_question"],
                    intent=analysis["intent"],
                    status=QueryStatus.ERROR.value,
                    sql_query=None,
                    sql_explanation=None,
                    results=json.dumps({"Error": "Query was not executed"}),
                    execution_duration=0.0,
                    error="Query was not executed",
                )
            )

    # Determine overall status
    error_count = sum(1 for r in state["final_results"] if r["status"] == "error")
    success_count = sum(1 for r in state["final_results"] if r["status"] == "success")
    placeholder_count = sum(
        1 for r in state["final_results"] if r["status"] == "placeholder"
    )

    if error_count == 0 and placeholder_count == 0:
        state["status"] = GraphStatus.SUCCESS.value
    elif success_count > 0 or placeholder_count > 0:
        state["status"] = GraphStatus.PARTIAL_SUCCESS.value
    else:
        state["status"] = GraphStatus.ERROR.value

    msg = f"[SUCCESS] Finalized {len(state['final_results'])} results"
    state["messages"].append(msg)
    logger.info(msg)
    logger.info(f"Status: {state['status']}")
    logger.info(
        f"Success: {success_count}, Placeholders: {placeholder_count}, Errors: {error_count}"
    )

    return state


def should_continue_execution(state: GraphState) -> str:
    """
    Routing function: determine if more queries need execution.
    """
    if state.get("error") and state["status"] == "error":
        return "finalize"

    if state["remaining_indices"]:
        return "continue"
    else:
        return "finalize"
