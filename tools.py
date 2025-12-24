"""
Utility functions for query processing
"""

import json
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd


def execute_duckdb_query(query: str, config: Any) -> pd.DataFrame:
    """
    Execute DuckDB query against Azure Blob Storage with proper error handling.

    Args:
        query: SQL query to execute
        config: Configuration object with Azure credentials

    Returns:
        DataFrame with query results or error
    """
    conn = None
    try:
        conn = duckdb.connect()

        # Install and load Azure extension
        conn.execute("INSTALL azure;")
        conn.execute("LOAD azure;")

        # Set Azure connection string
        escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
        conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")

        # Execute the query with timeout
        result_df = conn.execute(query).fetchdf()

        if result_df is None or result_df.empty:
            return pd.DataFrame({"message": ["No results returned"]})

        return result_df

    except duckdb.CatalogException as e:
        error_msg = f"Catalog error: {str(e)}"
        return pd.DataFrame({"Error": [error_msg]})
    except duckdb.ParserException as e:
        error_msg = f"SQL syntax error: {str(e)}"
        return pd.DataFrame({"Error": [error_msg]})
    except duckdb.IOException as e:
        error_msg = f"IO error accessing files: {str(e)}"
        return pd.DataFrame({"Error": [error_msg]})
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return pd.DataFrame({"Error": [error_msg]})
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def read_top_rows_duckdb(uri: str, config: Any) -> str:
    """
    Reads the top 5 rows from a Parquet file at a given URI using DuckDB
    and formats the result as a Markdown table string.
    """
    try:
        # Construct the DuckDB query to read the file and limit rows
        sql_query = f"SELECT * FROM read_parquet('{uri}') LIMIT 5"

        # NOTE: This assumes 'execute_duckdb_query' is available and handles
        # connection and configuration (like Azure credentials)
        result_df: pd.DataFrame = execute_duckdb_query(sql_query, config)

        if result_df.empty:
            return "No sample rows found."

        # Format the DataFrame as a clean Markdown table for the LLM
        return result_df.to_markdown(index=False)

    except Exception as e:
        return f"Error fetching sample data: {str(e)}"


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


def validate_state(state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
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
            return False, f"Missing required field: {field}"

    return True, None
