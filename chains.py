"""
LLM chains for query analysis and SQL generation
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI


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

**INTENT CLASSIFICATION:**
- SQL_QUERY: Precise filtering, aggregation, dates, numerical comparisons
- SUMMARY_SEARCH: Fuzzy logic, conceptual search, RAG, GraphDB

**FILE IDENTIFICATION:**
1. Use EXACT file names from catalog
2. Include ONLY files with required columns
3. Never use wildcards - use actual file names
4. Use ['*'] ONLY if ALL files needed

**DEPENDENCY DETECTION:**
- Set depends_on_index to dependency index
- -1 means independent
"""

    try:
        response = llm_client.chat.completions.create(
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

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return args["analyses"], usage

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required field in LLM response: {e}")
    except Exception as e:
        raise RuntimeError(f"Query analysis failed: {e}")


def generate_sql_chain(
    llm_client: AzureOpenAI,
    deployment_name: str,
    user_query: str,
    parquet_schema: str,
    df_sample: str,
    path_map: Dict[str, str],
    semantic_context: str = "",
    error_message: str = None,  # <--- ADDED: Error message for self-healing
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
2. **COLUMN ALIASES:** Use `AS` to give clear, user-friendly names to calculated fields (e.g., `SUM(...) AS total_sales`).
3. **MANDATORY GROUPING:** If the `SELECT` clause contains any aggregate function (like `SUM`, `AVG`, `COUNT`), you **MUST** include a `GROUP BY` clause listing all non-aggregated columns (`region`, `product_category`, etc.). This is required to prevent "Binder Error."
4. **RANKING/TOP-N:** For "highest X per Y" or "top N" questions, you **MUST** use the `QUALIFY` clause with a Window Function (`ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY sum_col DESC) = 1`) to filter the results.
5. **CLAUSE ORDER (CRITICAL):** The sequence of clauses is strictly enforced: `... FROM ... WHERE ... **GROUP BY** ... **QUALIFY** ...  . The **GROUP BY** clause must immediately precede the **QUALIFY** clause.
6. **AGGREGATION CHOICE:** When calculating totals (e.g., "annual sales"), use `SUM()`.
7. **NULL HANDLING:** Include `WHERE column IS NOT NULL` for all columns used in critical calculations (aggregations, filters) to ensure accuracy.
8. **JSON FORMAT:** Your final output MUST be a valid JSON object.

Return valid JSON:
{{
    "sql_query": "SELECT ...",
    "explanation": "Brief explanation"
}}
"""

    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": sql_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=30,
        )

        result = json.loads(response.choices[0].message.content.strip())

        sql_query = result.get("sql_query", "")
        explanation = result.get("explanation", "")

        if not sql_query:
            raise ValueError("Empty SQL query generated")

        return sql_query, explanation

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        raise RuntimeError(f"SQL generation failed: {e}")


def generate_final_summary_chain(
    llm_client: AzureOpenAI,
    deployment_name: str,
    user_question: str,
    query_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate final summary combining all query results.

    Args:
        llm_client: Azure OpenAI client
        deployment_name: Model deployment name
        user_question: Original user question
        query_results: List of query results

    Returns:
        Dictionary with summary_text, tables, and has_tables
    """
    # Prepare results context
    results_context = ""
    for idx, result in enumerate(query_results, 1):
        results_context += f"\n--- Result {idx} ---\n"
        results_context += f"Question: {result['sub_question']}\n"
        results_context += f"Status: {result['status']}\n"

        if result["status"] == "success":
            results_context += f"SQL: {result.get('sql_query', 'N/A')}\n"
            results_context += (
                f"Data: {result['results'][:1000]}...\n"  # Truncate for context
            )
        elif result["status"] == "placeholder":
            results_context += f"Placeholder: {result['results']}\n"

        results_context += "\n"

    summary_prompt = f"""
You are an expert data analyst. Generate a comprehensive summary of query results.

**ORIGINAL QUESTION:**
{user_question}

**QUERY RESULTS:**
{results_context}

**YOUR TASK:**
1. Analyze all results and synthesize insights
2. Determine if tabular data is appropriate:
   - Use tables for: comparisons, rankings, statistical data, trends over time
   - Use text only for: conceptual explanations, qualitative insights, recommendations
3. Create a narrative summary with key findings
4. If tables are appropriate, format them clearly with headers and data

**OUTPUT FORMAT (JSON):**
{{
    "summary_text": "Comprehensive narrative summary with key insights, trends, and patterns. Be specific and reference actual data points. Organize with clear sections if needed.",
    "has_tables": true/false,
    "tables": [
        {{
            "title": "Table title",
            "description": "Brief description",
            "headers": ["Column1", "Column2", "Column3"],
            "rows": [
                ["Value1", "Value2", "Value3"],
                ["Value4", "Value5", "Value6"]
            ]
        }}
    ]
}}

**GUIDELINES:**
- Be specific with numbers, percentages, and data points
- Highlight key trends and patterns
- Compare across different dimensions when relevant
- Keep summary concise but comprehensive (500-1500 words)
- Tables should have clear headers and meaningful data
- If no table needed, return empty tables array
"""

    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
            timeout=60,
        )

        result = json.loads(response.choices[0].message.content.strip())

        # Validate structure
        summary_result = {
            "summary_text": result.get("summary_text", "No summary generated"),
            "has_tables": result.get("has_tables", False),
            "tables": result.get("tables", []),
        }

        return summary_result

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        raise RuntimeError(f"Summary generation failed: {e}")
