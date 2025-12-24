"""
Utility functions for query processing
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from .logging_config import get_logger
from .retry_utils import retry_with_exponential_backoff

logger = get_logger()


def validate_sql_query(query: str) -> None:
    """
    Validate SQL query for safety - only allow SELECT operations.

    This prevents SQL injection attacks by ensuring only read-only
    SELECT queries are executed. Any attempt to modify data or
    database structure will be rejected.

    Args:
        query: SQL query string to validate

    Raises:
        ValueError: If query contains dangerous operations
    """
    query_upper = query.upper().strip()

    # Only allow SELECT queries
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous keywords
    dangerous_keywords = [
        "DROP",
        "DELETE",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "INSERT",
        "UPDATE",
        "GRANT",
        "REVOKE",
        "EXEC",
        "EXECUTE",
        "PRAGMA",
    ]

    for keyword in dangerous_keywords:
        if re.search(rf"\b{keyword}\b", query_upper):
            raise ValueError(f"Dangerous SQL operation detected: {keyword}")

    logger.debug("SQL query validation passed")


def execute_duckdb_query(query: str, config: Any) -> pd.DataFrame:
    """
    Execute DuckDB query against Azure Blob Storage with proper error handling.

    Args:
        query: SQL query to execute (must be SELECT only)
        config: Configuration object with Azure credentials

    Returns:
        DataFrame with query results or error

    Security:
        - Validates query to allow only SELECT operations
        - Uses proper escaping for connection string

    Raises:
        ValueError: If query validation fails
    """
    # SECURITY: Validate query before execution
    validate_sql_query(query)

    try:

        @retry_with_exponential_backoff(
            max_attempts=2,
            initial_wait=2.0,
            exceptions=(duckdb.IOException,),
        )
        def execute_with_retry():
            # Use context manager for automatic resource cleanup
            with duckdb.connect() as conn:
                # Install and load Azure extension
                conn.execute("INSTALL azure;")
                conn.execute("LOAD azure;")

                # SECURITY FIX: DuckDB doesn't support parameterized SET statements
                # Use proper escaping instead (single quotes must be doubled)
                escaped_conn_str = config.azure_storage.connection_string.replace(
                    "'", "''"
                )
                conn.execute(
                    f"SET azure_storage_connection_string='{escaped_conn_str}';"
                )

                logger.debug(f"Executing query: {query[:100]}...")

                # Execute the query
                result_df = conn.execute(query).fetchdf()

                if result_df is None or result_df.empty:
                    logger.info("Query returned no results")
                    return pd.DataFrame({"message": ["No results returned"]})

                logger.info(f"Query returned {len(result_df)} rows")
                return result_df

        return execute_with_retry()

    except duckdb.CatalogException as e:
        error_msg = f"Catalog error: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame({"Error": [error_msg]})
    except duckdb.ParserException as e:
        error_msg = f"SQL syntax error: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame({"Error": [error_msg]})
    except duckdb.IOException as e:
        error_msg = f"IO error accessing files: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame({"Error": [error_msg]})
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return pd.DataFrame({"Error": [error_msg]})


def read_top_rows_duckdb(uri: str, config: Any) -> str:
    """
    Reads the top 5 rows from a Parquet file at a given URI using DuckDB
    and formats the result as a Markdown table string.

    Args:
        uri: Azure URI of the parquet file
        config: Configuration object with Azure credentials

    Returns:
        Markdown formatted table string or error message
    """
    try:
        logger.debug(f"Reading sample rows from: {uri}")

        # Construct the DuckDB query to read the file and limit rows
        sql_query = f"SELECT * FROM read_parquet('{uri}') LIMIT 5"

        result_df: pd.DataFrame = execute_duckdb_query(sql_query, config)

        # Check for errors in the result
        if "Error" in result_df.columns:
            error_msg = result_df["Error"].iloc[0]
            logger.warning(f"Error reading sample data: {error_msg}")
            return f"Error: {error_msg}"

        if result_df.empty:
            logger.info("No sample rows found")
            return "No sample rows found."

        # Format the DataFrame as a clean Markdown table for the LLM
        return result_df.to_markdown(index=False)

    except Exception as e:
        error_msg = f"Error fetching sample data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def df_to_json_result(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to JSON string with error handling.

    Args:
        df: DataFrame to convert

    Returns:
        JSON string representation
    """
    try:
        if df is None or df.empty:
            return json.dumps([])
        return df.to_json(orient="records", date_format="iso", default_handler=str)
    except Exception as e:
        return json.dumps({"Error": f"Failed to serialize results: {str(e)}"})


# ===========================================================================
# HELPER FUNCTIONS FOR SCHEMA EXTRACTION
# ===========================================================================


def _find_file_info(
    file_name: str, global_catalog_dict: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Finds file metadata by logical name or file_name."""
    for logical_name, info in global_catalog_dict.items():
        if info.get("file_name") == file_name or logical_name == file_name:
            return info
    return None


def _format_file_schema(file_name: str, file_info: Dict[str, Any]) -> str:
    """Builds the schema string for a single file."""
    schema_str = ""
    if "columns" in file_info:
        schema_str += f"\n--- {file_name} Schema ---\n"
        for col in file_info["columns"]:
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "unknown")
            col_desc = col.get("description", "")

            # Format: - column_name (TYPE): description
            line = f"  - {col_name} ({col_type})"
            if col_desc:
                line += f": {col_desc}"
            schema_str += line + "\n"
    return schema_str


def _format_file_sample_data(file_name: str, file_info: Dict[str, Any]) -> str:
    """Builds the sample data string for a single file, including metadata and rows."""
    df_sample_str = ""

    # 1. Add file description metadata
    df_sample_str += f"\n--- {file_name} File Metadata ---\n"
    if "row_count" in file_info:
        df_sample_str += f"Total Rows: {file_info['row_count']}"
        if "summary" in file_info:
            df_sample_str += f" | Summary: {file_info['summary']}"
        df_sample_str += "\n"

    # 2. Extract actual sample rows
    sample_key = None
    if "sample_data_markdown" in file_info and file_info["sample_data_markdown"]:
        sample_key = "sample_data_markdown"
    elif "sample_data" in file_info and file_info["sample_data"]:
        sample_key = "sample_data"

    if sample_key:
        df_sample_str += f"\n--- {file_name} Sample Rows (Top 5) ---\n"
        df_sample_str += file_info[sample_key]
        df_sample_str += "\n"

    return df_sample_str


# ===========================================================================
# REFACTORED MAIN FUNCTION
# ===========================================================================


def extract_schema_from_catalog(
    required_files: List[str], global_catalog_dict: Dict[str, Any]
) -> tuple[str, str]:
    """
    Extract schema and sample data info (including actual top rows) from catalog.

    Args:
        required_files: List of required file names (e.g., ['TDQP_parquet'])
        global_catalog_dict: Global catalog dictionary containing all file metadata

    Returns:
        Tuple of (schema_str, sample_data_str)
    """
    all_parquet_schema = []
    all_df_sample = []

    # 1. Handle edge cases (no files or wildcard)
    if not required_files or required_files == ["*"]:
        return "", ""

    for file_name in required_files:
        file_info = _find_file_info(file_name, global_catalog_dict)

        if not file_info:
            continue

        # A. SCHEMA EXTRACTION
        all_parquet_schema.append(_format_file_schema(file_name, file_info))

        # B. SAMPLE DATA EXTRACTION
        all_df_sample.append(_format_file_sample_data(file_name, file_info))

    return "".join(all_parquet_schema), "".join(all_df_sample)


def build_path_mapping(
    required_files: List[str], global_catalog_dict: Dict[str, Any]
) -> Dict[str, str]:
    """
    Build mapping of file names to Azure URIs.

    Args:
        required_files: List of required file names
        global_catalog_dict: Global catalog dictionary

    Returns:
        Dictionary mapping file names to URIs
    """
    path_map = {}

    if not required_files or required_files == ["*"]:
        return path_map

    for file_name in required_files:
        for logical_name, file_info in global_catalog_dict.items():
            if file_info.get("file_name") == file_name or logical_name == file_name:
                uri = file_info.get("azure_uri") or file_info.get("parquet_path")
                if uri:
                    path_map[file_name] = uri
                break

    return path_map


def validate_state(state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate state has required fields.

    Args:
        state: State dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        "user_question",
        "config",
        "global_catalog_dict",
        "catalog_schema",
    ]

    for field in required_fields:
        if field not in state or state[field] is None:
            logger.error(f"State validation failed: missing field '{field}'")
            return False, f"Missing required field: {field}"

    logger.debug("State validation passed")
    return True, None
