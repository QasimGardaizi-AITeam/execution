"""
LangGraph nodes for query processing workflow
"""

import json
import time
from typing import Any, Dict

from openai import AzureOpenAI

from .chains import (
    analyze_query_chain,
    generate_final_summary_chain,
    generate_sql_chain,
)
from .state import GraphState, QueryAnalysis, QueryResult
from .tools import (
    build_path_mapping,
    df_to_json_result,
    execute_duckdb_query,
    extract_schema_from_catalog,
    read_top_rows_duckdb,
    validate_state,
)


def validate_input_node(state: GraphState) -> GraphState:
    """
    Validate input state and initialize fields.
    """
    print("\n" + "=" * 80)
    print("NODE: Validate Input")
    print("=" * 80)

    # Validate required fields
    is_valid, error_msg = validate_state(state)
    if not is_valid:
        state["error"] = error_msg
        state["status"] = "error"
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
    print("[SUCCESS] Input validated")

    return state


def analyze_query_node(state: GraphState) -> GraphState:
    """
    Analyze user query and decompose into sub-questions.
    """
    print("\n" + "=" * 80)
    print("NODE: Analyze Query")
    print("=" * 80)

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
        print(msg)

        if usage:
            print(f"Tokens used: {usage['total_tokens']}")

        for idx, analysis in enumerate(state["analyses"]):
            print(f"\n--- Question {idx + 1} ---")
            print(f"Sub-Question: {analysis['sub_question']}")
            print(f"Intent: {analysis['intent']}")
            print(f"Files: {', '.join(analysis['required_files'])}")
            if analysis["depends_on_index"] >= 0:
                print(f"⚠️  Depends on Q{analysis['depends_on_index'] + 1}")

        return state

    except Exception as e:
        error_msg = f"Query analysis failed: {str(e)}"
        state["error"] = error_msg
        state["status"] = "error"
        state["messages"].append(f"[ERROR] {error_msg}")
        print(f"[ERROR] {error_msg}")

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
    print("\n" + "=" * 80)
    print("NODE: Identify Ready Queries")
    print("=" * 80)

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
        print(f"[ERROR] {error_msg}")

        # Mark remaining as errors
        for idx in state["remaining_indices"]:
            analysis = state["analyses"][idx]
            state["executed_results"][idx] = QueryResult(
                sub_question=analysis["sub_question"],
                intent=analysis["intent"],
                status="error",
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
    print(msg)

    return state


def execute_sql_query_node(state: GraphState) -> GraphState:
    """
    Execute SQL queries in current batch with LLM-based self-healing fallback.
    """
    print("\n" + "=" * 80)
    print(f"NODE: Execute SQL Queries (Batch: {len(state['current_batch'])})")
    print("=" * 80)

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
        state["messages"].append(f"[ERROR] {error_msg}")
        print(f"[ERROR] {error_msg}")
        return state

    for idx in state["current_batch"]:
        analysis = state["analyses"][idx]

        # Skip non-SQL queries
        if analysis["intent"] != "SQL_QUERY":
            continue

        processed_indices.append(idx)
        print(f"\n--- Processing Q{idx + 1}: {analysis['sub_question']} ---")

        start_time = time.time()

        # --- SELF-HEALING FALLBACK MECHANISM START ---

        sql_query = None
        explanation = None
        result_df = None
        error_msg = None

        # Track the error from the previous attempt for self-healing
        previous_error_msg = None

        # Max 2 attempts: 1 original attempt + 1 self-healing fallback
        max_attempts = 2

        for attempt in range(max_attempts):

            current_sql_query = None
            current_explanation = None
            current_error_msg = None

            try:
                if attempt > 0:
                    print(
                        f"--- FAILED. ATTEMPTING SELF-HEALING FALLBACK (Try {attempt + 1}/{max_attempts}) ---"
                    )
                    # previous_error_msg will be passed to generate_sql_chain below

                # 1. Get dependency result if needed
                dependency_result = ""
                if analysis["depends_on_index"] >= 0:
                    dep_result = state["executed_results"].get(
                        analysis["depends_on_index"]
                    )
                    if dep_result and dep_result["status"] == "success":
                        dependency_result = dep_result["results"]
                        if attempt == 0 or state["enable_debug"]:
                            print(
                                f"→ Using result from Q{analysis['depends_on_index'] + 1}"
                            )

                # 2. Extract schema
                parquet_schema, _ = extract_schema_from_catalog(
                    analysis["required_files"], state["global_catalog_dict"]
                )

                # 3. Build path map
                path_map = build_path_mapping(
                    analysis["required_files"], state["global_catalog_dict"]
                )

                # 4. Dynamic sample data logic
                df_sample = "No sample data available or required."
                if path_map:
                    first_file_name = analysis["required_files"][0]
                    first_uri = (
                        path_map.get(first_file_name) or list(path_map.values())[0]
                    )

                    if attempt == 0 or state["enable_debug"]:
                        print(f"Reading sample data from: {first_uri}")

                    df_sample_markdown = read_top_rows_duckdb(
                        first_uri, state["config"]
                    )

                    df_sample = (
                        f"--- Actual Sample Rows from '{first_file_name}' ---\n"
                        + df_sample_markdown
                    )
                    if attempt == 0 or state["enable_debug"]:
                        print(f"[SAMPLE] Successfully read sample rows.")

                # 5. Generate SQL
                current_sql_query, current_explanation = generate_sql_chain(
                    llm_client=llm_client,
                    deployment_name=state["config"].azure_openai.llm_deployment_name,
                    user_query=analysis["sub_question"],
                    parquet_schema=parquet_schema,
                    df_sample=df_sample,
                    path_map=path_map,
                    semantic_context=dependency_result,
                    error_message=previous_error_msg,  # <--- PASS ERROR FOR SELF-HEALING
                )

                print(f"[SQL] {current_sql_query}")
                print(f"[EXPLANATION] {current_explanation}")

                # 6. Execute query
                result_df = execute_duckdb_query(current_sql_query, state["config"])

                # Check for execution errors (DuckDB returns a DataFrame with an 'Error' column)
                if "Error" in result_df.columns:
                    current_error_msg = result_df["Error"].iloc[0]
                    # This raises an exception to trigger the next attempt or final failure
                    raise Exception(f"DuckDB Execution Error: {current_error_msg}")

                # If successful, assign final values and break the retry loop
                sql_query = current_sql_query
                explanation = current_explanation
                error_msg = None  # Clear any previous error
                break

            except Exception as e:
                # Store the error from this attempt to feed back to the LLM on the next attempt
                previous_error_msg = str(e)
                # Store the SQL/Explanation that failed for final logging if max_attempts is reached
                sql_query = current_sql_query
                explanation = current_explanation
                error_msg = previous_error_msg

                if attempt == max_attempts - 1:
                    print(
                        f"[FINAL ERROR] Failed after {max_attempts} attempts: {error_msg}"
                    )
                else:
                    print(f"[ERROR] Attempt {attempt + 1} failed: {error_msg}")

        # --- SELF-HEALING FALLBACK MECHANISM END ---

        execution_duration = time.time() - start_time

        # Process result after retry loop
        if error_msg:
            # Final failure case
            state["executed_results"][idx] = QueryResult(
                sub_question=analysis["sub_question"],
                intent=analysis["intent"],
                status="error",
                sql_query=sql_query,
                sql_explanation=explanation,
                results=json.dumps({"Error": error_msg}),
                execution_duration=execution_duration,
                error=error_msg,
            )
        elif result_df is not None:
            # Final success case
            state["executed_results"][idx] = QueryResult(
                sub_question=analysis["sub_question"],
                intent=analysis["intent"],
                status="success",
                sql_query=sql_query,
                sql_explanation=explanation,
                results=df_to_json_result(result_df),
                execution_duration=execution_duration,
                error=None,
            )
            print(f"[SUCCESS] Executed in {execution_duration:.2f}s")
            print(f"[RESULTS] {result_df.shape[0]} rows x {result_df.shape[1]} cols")

            if state["enable_debug"] and not result_df.empty:
                print("\n" + result_df.head(5).to_markdown(index=False))

    # Remove processed indices from remaining
    for idx in processed_indices:
        if idx in state["remaining_indices"]:
            state["remaining_indices"].remove(idx)

    return state


def execute_summary_search_node(state: GraphState) -> GraphState:
    """
    Execute summary search queries in current batch (placeholder).
    """
    print("\n" + "=" * 80)
    print(f"NODE: Execute Summary Searches (Batch: {len(state['current_batch'])})")
    print("=" * 80)

    # Track which indices we process
    processed_indices = []

    for idx in state["current_batch"]:
        analysis = state["analyses"][idx]

        # Skip non-summary queries
        if analysis["intent"] != "SUMMARY_SEARCH":
            continue

        processed_indices.append(idx)
        print(f"\n--- Processing Q{idx + 1}: {analysis['sub_question']} ---")
        print("[PLACEHOLDER] Summary search not implemented")

        # Get dependency result if needed
        dependency_result = ""
        if analysis["depends_on_index"] >= 0:
            dep_result = state["executed_results"].get(analysis["depends_on_index"])
            if dep_result and dep_result["status"] == "success":
                dependency_result = dep_result["results"]
                print(f"→ Using result from Q{analysis['depends_on_index'] + 1}")

        # Create placeholder result
        state["executed_results"][idx] = QueryResult(
            sub_question=analysis["sub_question"],
            intent=analysis["intent"],
            status="placeholder",
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

        print(f"[PLACEHOLDER] Marked as complete")

    # Remove processed indices from remaining
    for idx in processed_indices:
        if idx in state["remaining_indices"]:
            state["remaining_indices"].remove(idx)

    # Clear current batch after processing
    state["current_batch"] = []

    return state


def generate_final_summary_node(
    state: GraphState,
) -> Dict[str, Any]:  # Note: Function return type is changed
    """
    Generate final summary combining all query results.
    Uses LLM to create narrative summary with optional tables.
    """
    print("\n" + "=" * 80)
    print("NODE: Generate Final Summary")
    print("=" * 80)

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
            print("[SKIP] No successful results to summarize")
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

        # state["final_summary"] = summary_result # OLD: Mutating state directly

        print(f"[SUCCESS] Generated final summary")
        print(f"Has tables: {summary_result['has_tables']}")
        print(f"Number of tables: {len(summary_result['tables'])}")

        if state["enable_debug"]:
            print("\n--- Summary Preview ---")
            print(summary_result["summary_text"][:500] + "...")

        # --- CHANGE 2: Return explicit update on success ---
        return {
            "final_summary": summary_result,
        }

    except Exception as e:
        error_msg = f"Failed to generate summary: {str(e)}"
        state["messages"].append(f"[ERROR] {error_msg}")
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback

        print(f"[ERROR] Traceback: {traceback.format_exc()}")

        # Provide fallback summary
        # state["final_summary"] = { ... } # OLD: Mutating state directly

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
    print("\n" + "=" * 80)
    print("NODE: Finalize Results")
    print("=" * 80)

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
                    status="error",
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
        state["status"] = "success"
    elif success_count > 0 or placeholder_count > 0:
        state["status"] = "partial_success"
    else:
        state["status"] = "error"

    msg = f"[SUCCESS] Finalized {len(state['final_results'])} results"
    state["messages"].append(msg)
    print(msg)
    print(f"Status: {state['status']}")
    print(
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
