"""
Enums for type-safe status and intent values.

This module defines enumerations used throughout the pipeline execution
to ensure type safety and consistency when representing:
- Query execution status (success, error, placeholder)
- Query intent types (SQL query vs summary search)
- Overall graph execution status

Using enums instead of magic strings provides:
- Type checking at development time
- IDE autocomplete support
- Prevention of typos
- Self-documenting code

Usage:
    from .enums import QueryStatus, IntentType, GraphStatus

    # Use .value to get the string representation
    result = {
        "status": QueryStatus.SUCCESS.value,  # "success"
        "intent": IntentType.SQL_QUERY.value   # "SQL_QUERY"
    }

    # Compare with enum members
    if result["status"] == QueryStatus.SUCCESS.value:
        print("Query succeeded!")
"""

from enum import Enum


class QueryStatus(str, Enum):
    """Status of individual query execution."""

    SUCCESS = "success"
    ERROR = "error"
    PLACEHOLDER = "placeholder"


class IntentType(str, Enum):
    """Type of query intent."""

    SQL_QUERY = "SQL_QUERY"
    SUMMARY_SEARCH = "SUMMARY_SEARCH"


class GraphStatus(str, Enum):
    """Overall graph execution status."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    PENDING = "pending"
