"""
Enums for type-safe status and intent values
"""

from enum import Enum


class QueryStatus(str, Enum):
    """Status of a query execution"""

    SUCCESS = "success"
    ERROR = "error"
    PLACEHOLDER = "placeholder"


class IntentType(str, Enum):
    """Type of query intent"""

    SQL_QUERY = "SQL_QUERY"
    SUMMARY_SEARCH = "SUMMARY_SEARCH"


class GraphStatus(str, Enum):
    """Overall status of the graph execution"""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    PENDING = "pending"
