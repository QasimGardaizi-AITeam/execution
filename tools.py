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
    parquet_schema = ""
    df_sample = ""

    # 1. Handle edge cases (no files or wildcard)
    if not required_files or required_files == ["*"]:
        return parquet_schema, df_sample

    for file_name in required_files:
        # Find the metadata for the required file, using either the logical name or file_name
        file_info = None
        for logical_name, info in global_catalog_dict.items():
            if info.get("file_name") == file_name or logical_name == file_name:
                file_info = info
                break

        if not file_info:
            continue

        # --- A. SCHEMA EXTRACTION ---
        if "columns" in file_info:
            parquet_schema += f"\n--- {file_name} Schema ---\n"
            for col in file_info["columns"]:
                col_name = col.get("name", "unknown")
                col_type = col.get("type", "unknown")
                col_desc = col.get("description", "")

                # Format: - column_name (TYPE): description
                parquet_schema += f"  - {col_name} ({col_type})"
                if col_desc:
                    parquet_schema += f": {col_desc}"
                parquet_schema += "\n"

        # --- B. SAMPLE DATA EXTRACTION (FIXED) ---

        # 1. Add file description metadata (what the old code was doing)
        df_sample += f"\n--- {file_name} File Metadata ---\n"
        if "row_count" in file_info:
            df_sample += f"Total Rows: {file_info['row_count']}"
            if "summary" in file_info:
                df_sample += f" | Summary: {file_info['summary']}"
            df_sample += "\n"

        # 2. **CRITICAL FIX:** Extract the actual sample rows (assuming key is 'sample_data_markdown')
        # This provides the LLM with actual values for semantic context.
        if "sample_data_markdown" in file_info and file_info["sample_data_markdown"]:
            df_sample += f"\n--- {file_name} Sample Rows (Top 5) ---\n"
            # Assuming this field contains a pre-formatted string (e.g., Markdown table from Pandas)
            df_sample += file_info["sample_data_markdown"]
            df_sample += "\n"
        elif "sample_data" in file_info and file_info["sample_data"]:
            # Fallback/alternative key check
            df_sample += f"\n--- {file_name} Sample Rows (Top 5) ---\n"
            df_sample += file_info["sample_data"]
            df_sample += "\n"

    return parquet_schema, df_sample


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
