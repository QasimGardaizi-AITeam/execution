"""
LLM chains for query analysis and SQL generation
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger, log_llm_call, track_execution_time
from openai import APIError, AzureOpenAI, RateLimitError, Timeout
from retry_utils import retry_with_exponential_backoff

logger = get_logger()


def analyze_query_chain(
    llm_client: AzureOpenAI,
    user_question: str,
    deployment_name: str,
    catalog_schema: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, int]]]:
    """
    Analyze user query and decompose into sub-questions.

    Args:
        llm_client: Azure OpenAI client
        user_question: User's question
        deployment_name: Model deployment name
        catalog_schema: JSON string of catalog schema

    Returns:
        Tuple of (analyses_list, usage_dict)
    """
    UNIFIED_TOOL_SPEC = {
        "type": "function",
        "function": {
            "name": "analyze_query",
            "description": "Analyze user query and decompose into structured sub-questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "analyses": {
                        "type": "array",
                        "description": "List of analyzed sub-questions",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sub_question": {
                                    "type": "string",
                                    "description": "A single, atomic question",
                                },
                                "intent": {
                                    "type": "string",
                                    "enum": ["SQL_QUERY", "SUMMARY_SEARCH"],
                                    "description": "Intent classification",
                                },
                                "required_files": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of EXACT file names from catalog",
                                },
                                "join_key": {
                                    "type": "string",
                                    "description": "Common column name for joins",
                                },
                                "depends_on_index": {
                                    "type": "integer",
                                    "description": "Index of dependency (-1 if independent)",
                                },
                            },
                            "required": ["sub_question", "intent", "required_files"],
                        },
                    }
                },
                "required": ["analyses"],
            },
        },
    }

    SYSTEM_PROMPT = f"""
You are an expert query analyzer. Analyze the user's question and provide structured analysis.

--- AVAILABLE DATA CATALOG ---
{catalog_schema}
--- END CATALOG ---

**DECOMPOSITION RULES:**
1. Break into MULTIPLE sub-questions ONLY if atomically independent OR dependent
2. Keep as SINGLE question if parts share same filter/aggregation
3. If handled by single SQL query, do NOT decompose
4. Make Minimum number of sub-questions that can be handled by single SQL query either by join or by cte or simple query.

**INTENT CLASSIFICATION:**
- SQL_QUERY: Precise filtering, aggregation, dates, numerical comparisons
- SUMMARY_SEARCH: Fuzzy logic, conceptual search, RAG, GraphDB
(FOR NOW RETURN ONLY SQL_QUERY)

**FILE IDENTIFICATION:**
1. Use EXACT file names from catalog
2. Include ONLY files with required columns
3. Never use wildcards - use actual file names
4. Use ['*'] ONLY if ALL files needed

**DEPENDENCY DETECTION:**
- Set depends_on_index to dependency index
- -1 means independent
- Only set depends_on_index if the query MUST use specific values from a previous result
- If queries can access the same source data independently, keep them independent (depends_on_index = -1)
- Don't create dependencies just because queries are related conceptually
"""

    try:

        @retry_with_exponential_backoff(
            max_attempts=3,
            initial_wait=2.0,
            exceptions=(RateLimitError, APIError, Timeout),
        )
        def make_llm_call():
            return llm_client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze: {user_question}"},
                ],
                tools=[UNIFIED_TOOL_SPEC],
                tool_choice={"type": "function", "function": {"name": "analyze_query"}},
                temperature=0.0,
                timeout=30,
            )

        with track_execution_time("Query Analysis LLM Call", logger) as timing:
            response = make_llm_call()

        # SECURITY FIX: Validate tool_calls exists before accessing
        if not response.choices or not response.choices[0].message.tool_calls:
            logger.error("LLM response missing tool_calls")
            raise ValueError(
                "No tool calls in LLM response. The model may not have followed instructions."
            )

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            log_llm_call(
                logger,
                deployment_name,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                timing["duration"],
            )

        logger.info(
            f"Analyzed query into {len(args.get('analyses', []))} sub-questions"
        )
        return args["analyses"], usage

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from LLM: {e}")
        raise ValueError(f"Invalid JSON response from LLM: {e}")
    except KeyError as e:
        logger.error(f"Missing required field in LLM response: {e}")
        raise ValueError(f"Missing required field in LLM response: {e}")
    except Exception as e:
        logger.error(f"Query analysis failed: {e}", exc_info=True)
        raise RuntimeError(f"Query analysis failed: {e}")


def generate_sql_chain(
    llm_client: AzureOpenAI,
    deployment_name: str,
    user_query: str,
    parquet_schema: str,
    df_sample: str,
    path_map: Dict[str, str],
    semantic_context: str = "",
    error_message: Optional[str] = None,
    metrics_collector: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    Generate SQL query using LLM.

    Args:
        llm_client: Azure OpenAI client
        deployment_name: Model deployment name
        user_query: User's question
        parquet_schema: Schema information
        df_sample: Sample data info
        path_map: Mapping of files to URIs
        semantic_context: Optional semantic context
        error_message: Optional error message from a previous failed attempt (for self-healing)

    Returns:
        Tuple of (sql_query, explanation)
    """
    # Build augmentation hint
    augmentation_hint = ""
    if semantic_context:
        augmentation_hint = f"""
--- SEMANTIC CONTEXT ---
{semantic_context}

Use exact values from semantic context in WHERE clauses.
"""

    # Build error correction hint for self-healing
    error_correction_hint = ""
    if error_message:
        error_correction_hint = f"""
--- PREVIOUS FAILURE (CRITICAL) ---
The LAST ATTEMPT to generate and execute SQL FAILED with the following error:
{error_message}

You MUST REVISE the SQL QUERY to fix this error. Your new SQL must resolve the issue described above.
"""

    # Build path hint
    PATH_HINT = "\n--- FILE PATH MAPPING (CRITICAL) ---\n"
    if path_map:
        PATH_HINT += "Use EXACT full Azure URI with read_parquet().\n"
        PATH_HINT += "NO placeholders or wildcards.\n\n"
        for file_name, uri in path_map.items():
            PATH_HINT += f"'{file_name}' → '{uri}'\n"

        if len(path_map) == 1:
            uri = list(path_map.values())[0]
            PATH_HINT += f"\nExample:\nSELECT * FROM read_parquet('{uri}') WHERE ...\n"
        else:
            uris = list(path_map.values())
            PATH_HINT += f"\nExample JOIN:\nSELECT * FROM read_parquet('{uris[0]}') t1 JOIN read_parquet('{uris[1]}') t2 ON ...\n"
    else:
        PATH_HINT = "No specific files identified.\n"

    sql_prompt = f"""
Generate DuckDB SQL query for Parquet files on Azure Blob Storage.

{augmentation_hint}
{error_correction_hint} # <--- INCLUDED ERROR HINT
{PATH_HINT}

--- SCHEMA ---
{parquet_schema}

--- SAMPLE DATA ---
{df_sample}

--- QUERY ---
{user_query}

**CRITICAL INSTRUCTIONS FOR DUCKDB SQL GENERATION:**
1. **URI MANDATE:** Use the EXACT full Azure URI provided in FILE PATH MAPPING with `read_parquet()` (e.g., `read_parquet('azure://...')`). NEVER use placeholders or wildcards.
2. **COLUMN NAMES (CRITICAL):** 
   - Always use EXACT column names from the schema with proper quotes
   - Column names with spaces MUST be quoted: "Full name", "Position Title", "EBS Cost Center"
   - Column names are case-sensitive: "Full name" ≠ "full_name"
   - NEVER use snake_case if schema shows spaces: Use "Full name" NOT "full_name"
   - Check the SCHEMA section for exact column names before writing SQL
3. **COLUMN ALIASES:** Use `AS` to give clear, user-friendly names to calculated fields (e.g., `SUM(...) AS total_sales`).
4. **MANDATORY GROUPING:** If the `SELECT` clause contains any aggregate function (like `SUM`, `AVG`, `COUNT`), you **MUST** include a `GROUP BY` clause listing all non-aggregated columns (`region`, `product_category`, etc.). This is required to prevent "Binder Error."
5. **RANKING/TOP-N:** For "highest X per Y" or "top N" questions, you **MUST** use the `QUALIFY` clause with a Window Function (`ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY sum_col DESC) = 1`) to filter the results.
6. **CLAUSE ORDER (CRITICAL):** The sequence of clauses is strictly enforced: `... from .. WHERE ... **GROUP BY** ... **QUALIFY** ...  . The **GROUP BY** clause must immediately precede the **QUALIFY** clause.
7. **AGGREGATION CHOICE:** When calculating totals (e.g., "annual sales"), use `SUM()`.
8. **NULL HANDLING:** Include `WHERE column IS NOT NULL` for all columns used in critical calculations (aggregations, filters) to ensure accuracy.
9. ** TEXT MATCHING (CRITICAL — STRICT ENFORCEMENT):**
   - NEVER use = for text comparison
   - ALWAYS use case-insensitive fuzzy matching
   - ALWAYS use LIKE with wildcards (%)
   - ALWAYS use UPPER() or LOWER()
   - ALWAYS use REPLACE() to normalize punctuation
   - For name or title lookups:
     * Split words on spaces
     * Search each word with OR
     * ALSO include a full-string fuzzy match
   - Example pattern (MANDATORY):

     WHERE (
       UPPER(REPLACE(col, '.', '')) LIKE '%WORD1%'
       OR UPPER(REPLACE(col, '.', '')) LIKE '%WORD2%'
       OR UPPER(REPLACE(col, '.', '')) LIKE '%WORD1 WORD2%'
     )

Return valid JSON:
{{
    "sql_query": "SELECT ...",
    "explanation": "Brief explanation"
}}
"""

    try:

        @retry_with_exponential_backoff(
            max_attempts=3,
            initial_wait=2.0,
            exceptions=(RateLimitError, APIError, Timeout),
        )
        def make_llm_call():
            return llm_client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": sql_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=30,
            )

        with track_execution_time("SQL Generation LLM Call", logger) as timing:
            response = make_llm_call()

        result = json.loads(response.choices[0].message.content.strip())

        sql_query = result.get("sql_query", "")
        explanation = result.get("explanation", "")

        if not sql_query:
            logger.error("LLM generated empty SQL query")
            raise ValueError("Empty SQL query generated")

        # Log token usage if available
        if response.usage:
            log_llm_call(
                logger,
                deployment_name,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                timing["duration"],
            )
            if metrics_collector:
                metrics_collector.record_llm_call(
                    model=deployment_name,
                    operation="sql_generation",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    duration=timing["duration"],
                )

        logger.debug(f"Generated SQL query: {sql_query[:100]}...")
        return sql_query, explanation

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response: {e}")
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        logger.error(f"SQL generation failed: {e}", exc_info=True)
        raise RuntimeError(f"SQL generation failed: {e}")


def create_tables_from_results(
    query_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Create complete tables directly from query results without LLM truncation.

    This bypasses token limits by creating tables directly from JSON data
    instead of passing all data through the LLM.

    Args:
        query_results: List of query results

    Returns:
        List of table dictionaries with complete data
    """
    tables = []

    for idx, result in enumerate(query_results):
        if result["status"] != "success":
            continue

        # Parse the JSON results
        try:
            data = json.loads(result["results"])

            if not data or not isinstance(data, list):
                logger.warning(f"Result {idx + 1} has no data or invalid format")
                continue

            # Extract headers from first row
            if not data[0]:
                continue

            headers = list(data[0].keys())

            # Extract all rows (NO TRUNCATION)
            rows = []
            for row_data in data:
                row = [str(row_data.get(h, "")) for h in headers]
                rows.append(row)

            # Create table with ALL data
            tables.append(
                {
                    "title": result["sub_question"],
                    "description": f"Complete results: {len(rows)} rows",
                    "headers": headers,
                    "rows": rows,
                }
            )

            logger.info(f"Created table with {len(rows)} rows for result {idx + 1}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse result {idx + 1}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error creating table for result {idx + 1}: {e}")
            continue

    return tables


def generate_final_summary_chain(
    llm_client: AzureOpenAI,
    deployment_name: str,
    user_question: str,
    query_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate final summary combining all query results.

    Uses a hybrid approach:
    1. Creates complete tables directly from data (bypasses token limits)
    2. Uses LLM only for narrative summary (with truncated context)

    Args:
        llm_client: Azure OpenAI client
        deployment_name: Model deployment name
        user_question: Original user question
        query_results: List of query results

    Returns:
        Dictionary with summary_text, tables, and has_tables
    """
    # STEP 1: Create complete tables directly from data (no LLM, no truncation)
    complete_tables = create_tables_from_results(query_results)

    logger.info(f"Created {len(complete_tables)} complete tables from results")
    for idx, table in enumerate(complete_tables):
        logger.info(f"  Table {idx + 1}: {len(table['rows'])} rows")

    # STEP 2: Prepare lightweight context for LLM (summary only, not tables)
    results_context = ""
    for idx, result in enumerate(query_results, 1):
        results_context += f"\n--- Result {idx} ---\n"
        results_context += f"Question: {result['sub_question']}\n"
        results_context += f"Status: {result['status']}\n"

        if result["status"] == "success":
            results_context += f"SQL: {result.get('sql_query', 'N/A')}\n"

            # Only include metadata, not full data (to save tokens)
            try:
                data = json.loads(result["results"])
                results_context += f"Rows returned: {len(data)}\n"

                # Include just a small sample for context
                if len(data) > 0:
                    sample_size = min(3, len(data))
                    sample_data = data[:sample_size]
                    results_context += f"Sample (first {sample_size} rows): {json.dumps(sample_data)}\n"
            except:
                results_context += f"Data: {result['results'][:500]}...\n"

        elif result["status"] == "placeholder":
            results_context += f"Placeholder: {result['results']}\n"

        results_context += "\n"

    # STEP 3: Use LLM to generate narrative summary ONLY (not tables)
    summary_prompt = f"""
You are an expert data analyst. Generate a BRIEF narrative summary of the query results.

**ORIGINAL QUESTION:**
{user_question}

**QUERY RESULTS SUMMARY:**
{results_context}

**YOUR TASK:**
Generate a concise narrative summary (2-4 paragraphs) that:
1. Directly answers the original question
2. Highlights key findings and insights
3. References specific numbers and patterns from the data
4. Is written for a business audience (not developers)

**IMPORTANT:**
- DO NOT create tables - they are handled separately
- Focus on narrative insights and interpretation
- Be specific with numbers and findings
- Keep it concise but comprehensive

**OUTPUT FORMAT (JSON):**
{{
    "summary_text": "Your narrative summary here (2-4 paragraphs)"
}}
"""

    try:

        @retry_with_exponential_backoff(
            max_attempts=3,
            initial_wait=2.0,
            exceptions=(RateLimitError, APIError, Timeout),
        )
        def make_llm_call():
            return llm_client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
                timeout=60,
            )

        with track_execution_time("Summary Generation LLM Call", logger) as timing:
            response = make_llm_call()

        result = json.loads(response.choices[0].message.content.strip())

        # Combine LLM summary with complete tables (created in Step 1)
        summary_result = {
            "summary_text": result.get("summary_text", "No summary generated"),
            "has_tables": len(complete_tables) > 0,
            "tables": complete_tables,  # Use complete tables, not LLM-generated ones
        }

        # Log token usage if available
        if response.usage:
            log_llm_call(
                logger,
                deployment_name,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                timing["duration"],
            )

        logger.info(
            f"Generated summary with {len(summary_result['tables'])} complete tables"
        )
        return summary_result

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response: {e}")

        # Fallback: return tables without LLM summary
        return {
            "summary_text": f"Error generating narrative summary: {e}. See tables below for complete results.",
            "has_tables": len(complete_tables) > 0,
            "tables": complete_tables,
        }

    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)

        # Fallback: return tables without LLM summary
        return {
            "summary_text": f"Error generating summary: {e}. See tables below for complete results.",
            "has_tables": len(complete_tables) > 0,
            "tables": complete_tables,
        }
