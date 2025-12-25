"""
LLM chains for query analysis and SQL generation - FINAL COMPREHENSIVE FIX
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

**CRITICAL DECOMPOSITION RULES (READ CAREFULLY):**

1. **SINGLE QUERY PATTERNS (DO NOT DECOMPOSE):**
   - Hierarchical/parent-child relationships (e.g., "group tasks by parent", "assign subtasks to main tasks")
   - Aggregations with grouping (e.g., "sum sales by region")
   - Sorting/ordering operations (e.g., "sort by date")
   - Window functions needed (e.g., "top N per category", "running totals", "next/previous row")
   - Join operations (e.g., "combine data from files")
   - Single analytical question with multiple display fields
   
   **Examples of queries to keep as SINGLE:**
   - "Find main tasks and their subtasks sorted by ID" → SINGLE QUERY
   - "Group employees by department and show their salaries" → SINGLE QUERY
   - "Assign children to parents based on sequence" → SINGLE QUERY
   - "Top 5 products per region by sales" → SINGLE QUERY

2. **MULTIPLE QUERY PATTERNS (CAN DECOMPOSE):**
   - Truly independent questions (e.g., "What's average salary? What's average age?")
   - Sequential analysis where later query needs ACTUAL RESULTS (not just structure) from first
   - Different analytical perspectives on unrelated metrics
   
   **Examples of queries to decompose:**
   - "Who are the top bidders? How many children do they have?" → 2 QUERIES (different files, sequential)
   - "What's total revenue? What's total cost? What's profit margin?" → 3 QUERIES (independent metrics)

3. **DEPENDENCY DETECTION (STRICT RULES):**
   - Set depends_on_index ONLY when query needs SPECIFIC VALUES from previous result
   - Do NOT create dependency just because queries are related conceptually
   - Do NOT create dependency for structural/pattern understanding
   
   **Dependency Examples:**
   - Q1: "Find employees with salary > 100k" → Q2: "What are their manager names?" ✅ DEPENDENT (needs specific employee IDs)
   - Q1: "Find all tasks" → Q2: "Group tasks by type" ❌ NOT DEPENDENT (Q2 can access same data independently)

4. **SQL CAPABILITY CHECK:**
   - Can this be done with window functions (LEAD, LAG, ROW_NUMBER, PARTITION BY)? → SINGLE QUERY
   - Can this be done with CTEs (WITH clause)? → SINGLE QUERY
   - Can this be done with GROUP BY + HAVING? → SINGLE QUERY
   - Can this be done with self-joins? → SINGLE QUERY

**INTENT CLASSIFICATION:**
- SQL_QUERY: Precise filtering, aggregation, dates, numerical comparisons, hierarchical grouping
- SUMMARY_SEARCH: Fuzzy logic, conceptual search, RAG, GraphDB
(FOR NOW RETURN ONLY SQL_QUERY)

**FILE IDENTIFICATION:**
1. Use EXACT file names from catalog
2. Include ONLY files with required columns
3. Never use wildcards - use actual file names
4. Use ['*'] ONLY if ALL files needed

**EXAMPLES:**

Example 1 (SINGLE QUERY - Hierarchical):
User: "If Summary=Yes, treat as main task. Assign following Summary=No rows as subtasks. Sort by ID."
Analysis: [{{
  "sub_question": "Group tasks hierarchically by Summary field, assign subtasks to main tasks, and sort by ID",
  "intent": "SQL_QUERY",
  "required_files": ["tasks_file"],
  "depends_on_index": -1
}}]
Reasoning: Window functions can handle parent-child assignment. Single query.

Example 2 (MULTIPLE QUERIES - Sequential):
User: "Who are the highest bidders? How many children do highest bidders have?"
Analysis: [
  {{
    "sub_question": "Who are the highest bidders?",
    "intent": "SQL_QUERY",
    "required_files": ["bidders"],
    "depends_on_index": -1
  }},
  {{
    "sub_question": "How many children do the highest bidders have?",
    "intent": "SQL_QUERY", 
    "required_files": ["children"],
    "depends_on_index": 0
  }}
]
Reasoning: Different files, Q2 needs specific bidder names from Q1 results.

Example 3 (SINGLE QUERY - Aggregation):
User: "Show total sales by region and top 3 products per region"
Analysis: [{{
  "sub_question": "Calculate total sales by region and identify top 3 products per region",
  "intent": "SQL_QUERY",
  "required_files": ["sales"],
  "depends_on_index": -1
}}]
Reasoning: Window functions with PARTITION BY can do this. Single query.
"""

    try:

        @retry_with_exponential_backoff(max_attempts=3)
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
--- SEMANTIC CONTEXT (REFERENCE DATA) ---
{semantic_context}

**HOW TO USE SEMANTIC CONTEXT:**
- This shows SAMPLE/EXAMPLE data from a related query
- Use it to UNDERSTAND the data structure and relationships
- DO NOT filter your query to match these specific values
- DO NOT use WHERE clauses that limit results to context values
- Think: "This is what the data LOOKS LIKE" not "This is what to FILTER FOR"

**CORRECT vs INCORRECT Usage:**

INCORRECT (Too Restrictive):
- WHERE id IN (1, 2, 3)  -- Don't limit to context IDs
- WHERE name = 'John Doe'  -- Don't match specific context values
- WHERE unique_id = 0  -- Don't hardcode single values from context

CORRECT (Use Pattern):
- Apply the SAME LOGIC to ALL relevant rows
- Use context to understand column names, data types, patterns
- Filter based on CONDITIONS (e.g., summary='Yes') not VALUES (e.g., id=123)
"""

    # Build error correction hint for self-healing
    error_correction_hint = ""
    if error_message:
        error_correction_hint = f"""
--- PREVIOUS FAILURE (CRITICAL) ---
The LAST ATTEMPT to generate and execute SQL FAILED with the following error:
{error_message}

**COMMON ERRORS AND FIXES:**
1. "Only SELECT queries are allowed" → Use WITH (CTE) for complex queries, not temporary tables
2. "Column not found" → Check exact column names in schema (quotes, case-sensitivity)
3. "Cartesian product" → Add proper JOIN conditions with upper/lower bounds
4. "Binder error" → Add GROUP BY for all non-aggregated columns

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
{error_correction_hint}
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

3. **HIERARCHICAL/GROUPING QUERIES (CRITICAL):**
   - For parent-child relationships, use LEAD/LAG window functions
   - For "assign X to Y based on sequence", use window functions with PARTITION BY
   - For "group by parent", use self-joins or CTEs with window functions
   
   **Example - Assign subtasks to main tasks:**
   ```sql
   WITH main_tasks AS (
     SELECT id, task_name,
            LEAD(id) OVER (ORDER BY id) as next_main_id
     FROM read_parquet('...') 
     WHERE summary = 'Yes'
   )
   SELECT m.id as main_task_id, m.task_name as main_task,
          t.id as subtask_id, t.task_name as subtask_name
   FROM read_parquet('...') t
   LEFT JOIN main_tasks m 
     ON t.id >= m.id 
     AND (t.id < m.next_main_id OR m.next_main_id IS NULL)
   WHERE t.summary = 'No'
   ORDER BY m.id, t.id
   ```

4. **COLUMN ALIASES:** Use `AS` to give clear, user-friendly names to calculated fields (e.g., `SUM(...) AS total_sales`).

5. **MANDATORY GROUPING:** If the `SELECT` clause contains any aggregate function (like `SUM`, `AVG`, `COUNT`), you **MUST** include a `GROUP BY` clause listing all non-aggregated columns. This is required to prevent "Binder Error."

6. **RANKING/TOP-N:** For "highest X per Y" or "top N" questions, you **MUST** use the `QUALIFY` clause with a Window Function (`ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY sum_col DESC) = 1`) to filter the results.

7. **LIMIT CLAUSE:** 
   - For "top N" queries, use LIMIT N
   - For "highest value" queries, get ALL rows matching that value (no LIMIT)
   - Example: "top 5" → LIMIT 5, but "highest bidder" → no LIMIT (might be multiple tied)

8. **CLAUSE ORDER (CRITICAL):** The sequence of clauses is strictly enforced: `SELECT ... FROM ... WHERE ... GROUP BY ... HAVING ... QUALIFY ... ORDER BY ... LIMIT`. The **GROUP BY** clause must precede **QUALIFY**.

9. **AGGREGATION CHOICE:** When calculating totals (e.g., "annual sales"), use `SUM()`.

10. **NULL HANDLING:** Include `WHERE column IS NOT NULL` for all columns used in critical calculations (aggregations, filters) to ensure accuracy.

11. **TEXT MATCHING (CRITICAL — STRICT ENFORCEMENT):**
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
     ```sql
     WHERE (
       UPPER(REPLACE(col, '.', '')) LIKE '%WORD1%'
       OR UPPER(REPLACE(col, '.', '')) LIKE '%WORD2%'
       OR UPPER(REPLACE(col, '.', '')) LIKE '%WORD1 WORD2%'
     )
     ```

12. **JOIN BOUNDARIES (CRITICAL FOR HIERARCHICAL QUERIES):**
   - When assigning items to groups based on sequence/order, ALWAYS define BOTH lower AND upper boundaries
   - Use LEAD() to get the "next group start" as upper boundary
   - Example: Assigning subtasks between main tasks
     ```sql
     -- WRONG: No upper bound, creates cartesian product
     WHERE subtask.id > main_task.id
     
     -- RIGHT: Both lower and upper bounds
     WHERE subtask.id > main_task.id 
       AND (subtask.id < next_main_task.id OR next_main_task.id IS NULL)
     ```

13. **CTEs ARE ALLOWED:** Common Table Expressions (WITH clause) are safe and encouraged for complex queries.

Return valid JSON:
{{{{
    "sql_query": "SELECT ...",
    "explanation": "Brief explanation"
}}}}
"""

    try:

        @retry_with_exponential_backoff(max_attempts=3)
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
        # Skip intermediate results
        if result.get("is_intermediate", False):
            logger.info(
                f"Skipping intermediate result {idx + 1} (combined into another query)"
            )
            continue

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
        # Skip intermediate results
        if result.get("is_intermediate", False):
            continue

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
4. Is written for general users
5. Don't write like 'query results show that' or similar, don't mention query, treat QUERY RESULTS SUMMARY as knowledge base and answer for original question

**IMPORTANT:**
- DO NOT create tables - they are handled separately
- Focus on narrative insights and interpretation
- Be specific with numbers and findings
- Keep it concise but comprehensive

**OUTPUT FORMAT (JSON):**
{{{{
    "summary_text": "Your narrative summary here (2-4 paragraphs)"
}}}}
"""

    try:

        @retry_with_exponential_backoff(max_attempts=3)
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
