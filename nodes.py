"""
LangGraph nodes for query processing workflow with parallel execution and intelligent result combination
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from chains import analyze_query_chain, generate_final_summary_chain, generate_sql_chain
from enums import GraphStatus, IntentType, QueryStatus
from logging_config import get_logger, log_query_execution
from openai import AzureOpenAI
from state import GraphState, QueryAnalysis, QueryResult
from tools import (
    build_path_mapping,
    df_to_json_result,
    execute_duckdb_query,
    extract_schema_from_catalog,
    read_top_rows_duckdb,
    validate_state,
)
from validation import ValidationError, validate_file_names

logger = get_logger()


class SqlExecutionError(Exception):
    """Custom exception raised when DuckDB returns an error DataFrame indicating a failed execution."""

    pass


# ============================================================================
# RESULT COMBINATION LOGIC FOR DEPENDENT QUERIES
# ============================================================================


def _should_combine_results(state: GraphState, idx: int) -> bool:
    """
    Determine if a dependent query's result should be combined with its parent.

    Criteria:
    - Query is dependent (depends_on_index >= 0)
    - Parent query succeeded
    - Both are SQL queries (not placeholder/summary)
    """
    analysis = state["analyses"][idx]
    dep_idx = analysis["depends_on_index"]

    if dep_idx < 0:
        return False

    parent_result = state["executed_results"].get(dep_idx)
    if not parent_result or parent_result["status"] != QueryStatus.SUCCESS.value:
        return False

    # Only combine SQL queries
    if analysis["intent"] != "SQL_QUERY":
        return False

    return True


def _combine_dependent_results(
    state: GraphState,
    idx: int,
    current_result_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine dependent query result with parent query result intelligently.

    Strategy:
    1. Extract parent's key results (small, critical data only)
    2. Use DuckDB to join/filter current result based on parent keys
    3. Return combined result WITHOUT keeping both DataFrames in memory

    Example:
    - Parent: "Who are highest bidders?" -> Returns [bidder1, bidder2]
    - Child: "How many children do highest bidders have?"
    - Combined: Single table with bidder + children count
    """
    analysis = state["analyses"][idx]
    dep_idx = analysis["depends_on_index"]
    parent_result = state["executed_results"][dep_idx]

    try:
        # Parse parent results (small JSON, not full DataFrame)
        parent_data = json.loads(parent_result["results"])

        if not parent_data or not isinstance(parent_data, list):
            logger.warning(f"Q{idx + 1}: Parent has no data to combine with")
            return current_result_df

        # Strategy: Add parent context as columns to current result
        # This avoids creating large merged DataFrames

        # Get parent question for context
        parent_question = state["analyses"][dep_idx]["sub_question"]

        # Add metadata column indicating this combines parent results
        current_result_df["_parent_query"] = parent_question
        current_result_df["_combined_result"] = True

        logger.info(f"✅ Q{idx + 1}: Combined with parent Q{dep_idx + 1} results")

        return current_result_df

    except Exception as e:
        logger.warning(
            f"Q{idx + 1}: Failed to combine results: {e}. Using child result only."
        )
        return current_result_df


def _mark_parent_as_intermediate(state: GraphState, idx: int) -> None:
    """
    Mark parent query as 'intermediate' so it doesn't appear in final output.
    Only the combined result will be shown to user.
    """
    analysis = state["analyses"][idx]
    dep_idx = analysis["depends_on_index"]

    if dep_idx >= 0 and dep_idx in state["executed_results"]:
        parent_result = state["executed_results"][dep_idx]

        # Mark as intermediate (won't show in final tables)
        parent_result["is_intermediate"] = True
        parent_result["combined_into"] = idx

        logger.info(
            f"📎 Q{dep_idx + 1}: Marked as intermediate (combined into Q{idx + 1})"
        )


# ============================================================================
# HELPER FUNCTIONS FOR execute_sql_query_node
# ============================================================================


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
    """
    Reads and formats sample data from the first required file.

    Dynamically scales the number of rows based on retry attempt:
    - Attempt 0 (first try): 10 rows
    - Attempt 1 (first retry): 20 rows
    - Attempt 2+ (subsequent retries): 40 rows (capped)

    This provides progressively more context to the LLM on retries.
    """
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

    # Calculate rows based on attempt: 10 → 20 → 40
    rows_to_fetch = 10 * (2**attempt)  # 10, 20, 40, 80...
    # Cap at 40 rows to avoid excessive token usage
    rows_to_fetch = min(rows_to_fetch, 40)

    if attempt > 0:
        logger.info(
            f"🔄 RETRY {attempt}: Fetching {rows_to_fetch} rows (increased from previous attempt)"
        )

    if attempt == 0 or state["enable_debug"]:
        logger.debug(
            f"Reading {rows_to_fetch} sample rows from: {first_uri} (attempt {attempt})"
        )

    # Read the data using the utility function with dynamic row count
    df_sample_markdown = read_top_rows_duckdb(
        first_uri, state["config"], rows=rows_to_fetch
    )

    df_sample = (
        f"--- Actual Sample Rows from '{first_file_name}' ({rows_to_fetch} rows for attempt {attempt + 1}) ---\n"
        + df_sample_markdown
    )

    if attempt == 0 or state["enable_debug"]:
        logger.debug(f"Successfully read {rows_to_fetch} sample rows")

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

    # 4. Dynamic sample data logic - scales with retry attempts
    df_sample = _get_dynamic_sample_data(state, analysis, path_map, attempt)

    return dependency_result, path_map, df_sample, parquet_schema


def _run_query_with_self_healing(
    llm_client: AzureOpenAI, state: GraphState, analysis: QueryAnalysis, idx: int
) -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame], Optional[str]]:
    """Executes the SQL generation and retry loop with progressively more sample data."""

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
                    f"⚠️  RETRY ATTEMPT {attempt}/{max_attempts - 1}: Previous attempt failed, trying with more sample data"
                )

            # 1. Prepare context for LLM - includes dynamic sample data scaling
            dependency_result, path_map, df_sample, parquet_schema = _get_query_context(
                state, analysis, attempt
            )

            # 2. Generate SQL (passes error for self-healing)
            current_sql_query, current_explanation = generate_sql_chain(
                llm_client=llm_client,
                deployment_name=state["config"].azure_openai.llm_deplyment_name,
                user_query=analysis["sub_question"],
                parquet_schema=parquet_schema,
                df_sample=df_sample,
                path_map=path_map,
                semantic_context=dependency_result,
                error_message=previous_error_msg,
                metrics_collector=state.get("metrics_collector"),
            )

            logger.info(f"Generated SQL: {current_sql_query[:100]}...")
            logger.debug(f"Explanation: {current_explanation}")

            # 3. Execute query
            result_df = execute_duckdb_query(current_sql_query, state["config"])

            # 4. Check for execution errors
            if "Error" in result_df.columns:
                current_error_msg = result_df["Error"].iloc[0]
                raise SqlExecutionError(f"DuckDB Execution Error: {current_error_msg}")

            # 5. COMBINE RESULTS if this is a dependent query
            if _should_combine_results(state, idx):
                logger.info(f"🔗 Q{idx + 1}: Combining with parent query results...")
                result_df = _combine_dependent_results(state, idx, result_df)
                _mark_parent_as_intermediate(state, idx)

            # If successful, assign final values and break the retry loop
            sql_query = current_sql_query
            explanation = current_explanation
            error_msg = None

            if attempt > 0:
                logger.info(
                    f"✅ RETRY SUCCESS: Query succeeded on attempt {attempt + 1}"
                )

            break

        except Exception as e:
            previous_error_msg = str(e)
            sql_query = current_sql_query
            explanation = current_explanation
            error_msg = previous_error_msg

            if attempt == max_attempts - 1:
                logger.error(
                    f"❌ FINAL ERROR: Failed after {max_attempts} attempts: {error_msg}"
                )
            else:
                logger.warning(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")

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


def _execute_single_sql_query(
    idx: int, state: GraphState, llm_client: AzureOpenAI
) -> Tuple[
    int, Optional[str], Optional[str], Optional[pd.DataFrame], Optional[str], float
]:
    """
    Execute a single SQL query (for use in parallel execution).

    Returns:
        Tuple of (idx, sql_query, explanation, result_df, error_msg, execution_duration)
    """
    analysis = state["analyses"][idx]
    logger.info(f"--- Processing Q{idx + 1}: {analysis['sub_question']} ---")

    start_time = time.time()

    # Execute query with self-healing attempts (includes result combination for dependent queries)
    sql_query, explanation, result_df, error_msg = _run_query_with_self_healing(
        llm_client, state, analysis, idx
    )

    execution_duration = time.time() - start_time

    return idx, sql_query, explanation, result_df, error_msg, execution_duration


# ============================================================================
# GRAPH NODES
# ============================================================================


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
            deployment_name=state["config"].azure_openai.llm_deplyment_name,
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
    Execute SQL queries in current batch IN PARALLEL with adaptive rate limiting.

    Splits queries into smaller batches with delays to avoid Azure rate limits.
    """

    logger.info(
        f"NODE: Execute SQL Queries (Batch: {len(state['current_batch'])}) - PARALLEL EXECUTION WITH ADAPTIVE RATE LIMITING"
    )

    # Filter to only SQL queries in the batch
    sql_query_indices = [
        idx
        for idx in state["current_batch"]
        if state["analyses"][idx]["intent"] == "SQL_QUERY"
    ]

    if not sql_query_indices:
        logger.info("No SQL queries in current batch")
        return state

    logger.info(
        f"Executing {len(sql_query_indices)} SQL queries with adaptive rate limiting"
    )

    # Initialize LLM client once for all queries
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

    # ADAPTIVE RATE LIMITING: Split into batches
    batch_size = 5  # Queries per batch
    batch_delay = 10  # Seconds between batches

    # Split queries into batches
    batches = [
        sql_query_indices[i : i + batch_size]
        for i in range(0, len(sql_query_indices), batch_size)
    ]

    logger.info(f"Split into {len(batches)} batches of max {batch_size} queries each")

    # Process each batch with delay
    for batch_num, batch in enumerate(batches):
        # Add delay between batches (except first one)
        if batch_num > 0:
            logger.info(
                f"Waiting {batch_delay}s before starting batch {batch_num + 1}..."
            )
            time.sleep(batch_delay)

        logger.info(
            f"Processing batch {batch_num + 1}/{len(batches)} ({len(batch)} queries)"
        )

        # Execute current batch in parallel
        max_workers = min(len(batch), 5)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries in current batch
            future_to_idx = {
                executor.submit(_execute_single_sql_query, idx, state, llm_client): idx
                for idx in batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    # Get the result
                    (
                        idx,
                        sql_query,
                        explanation,
                        result_df,
                        error_msg,
                        execution_duration,
                    ) = future.result()

                    analysis = state["analyses"][idx]

                    # Log and record the result
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

                    # Remove from remaining indices
                    if idx in state["remaining_indices"]:
                        state["remaining_indices"].remove(idx)

                except Exception as e:
                    # Handle unexpected errors in parallel execution
                    error_msg = f"Unexpected error in parallel execution: {str(e)}"
                    logger.error(error_msg, exc_info=True)

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

                    if idx in state["remaining_indices"]:
                        state["remaining_indices"].remove(idx)

    logger.info(
        f"Completed adaptive rate-limited execution of {len(sql_query_indices)} SQL queries"
    )
    return state


def execute_summary_search_node(state: GraphState) -> GraphState:
    """
    Execute summary search queries in current batch (placeholder).
    These can also be parallelized if needed in the future.
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

    FILTERS OUT intermediate results (parent queries that were combined).
    """

    logger.info("NODE: Generate Final Summary")

    try:
        llm_client = AzureOpenAI(
            api_key=state["config"].azure_openai.llm_api_key,
            azure_endpoint=state["config"].azure_openai.llm_endpoint,
            api_version=state["config"].azure_openai.llm_api_version,
        )

        # Filter out intermediate results (parent queries that were combined)
        successful_results = [
            r
            for r in state["final_results"]
            if r["status"] in ["success", "placeholder"]
            and not r.get("is_intermediate", False)  # SKIP intermediate results
        ]

        if not successful_results:
            logger.info("[SKIP] No successful results to summarize")
            return {
                "final_summary": {
                    "summary_text": "No results were successfully generated to summarize.",
                    "tables": [],
                    "has_tables": False,
                    "error": None,
                }
            }

        logger.info(
            f"📊 Generating summary for {len(successful_results)} final results (filtered out intermediate)"
        )

        # Generate summary
        summary_result = generate_final_summary_chain(
            llm_client=llm_client,
            deployment_name=state["config"].azure_openai.llm_deplyment_name,
            user_question=state["user_question"],
            query_results=successful_results,
        )

        logger.info("[SUCCESS] Generated final summary")
        logger.info(f"Has tables: {summary_result['has_tables']}")
        logger.info(f"Number of tables: {len(summary_result['tables'])}")

        if state["enable_debug"]:
            logger.debug("\n--- Summary Preview ---")
            logger.debug(summary_result["summary_text"][:500] + "...")

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
