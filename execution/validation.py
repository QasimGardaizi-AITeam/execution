"""
Input validation and sanitization utilities for pipeline execution
"""

import re
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger()

# Constants for regex patterns
CONTROL_CHARS_PATTERN = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
REDACTED_PLACEHOLDER = r"\1***REDACTED***"

# Sensitive data patterns for redaction
SENSITIVE_PATTERNS = [
    (r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'\}\s]+)', REDACTED_PLACEHOLDER),
    (r'(password["\']?\s*[:=]\s*["\']?)([^"\'\}\s]+)', REDACTED_PLACEHOLDER),
    (r'(secret["\']?\s*[:=]\s*["\']?)([^"\'\}\s]+)', REDACTED_PLACEHOLDER),
    (r'(token["\']?\s*[:=]\s*["\']?)([^"\'\}\s]+)', REDACTED_PLACEHOLDER),
]


class ValidationError(Exception):
    """Raised when input validation fails"""

    pass


def sanitize_user_question(question: str, max_length: int = 1000) -> str:
    """
    Sanitize user question input.

    Args:
        question: User's question string
        max_length: Maximum allowed length

    Returns:
        Sanitized question string

    Raises:
        ValidationError: If validation fails
    """
    if not question or not isinstance(question, str):
        raise ValidationError("Question must be a non-empty string")

    # Remove leading/trailing whitespace
    question = question.strip()

    if not question:
        raise ValidationError("Question cannot be empty or only whitespace")

    # Check length
    if len(question) > max_length:
        logger.warning(
            f"Question truncated from {len(question)} to {max_length} characters"
        )
        question = question[:max_length]

    # Remove control characters except newlines and tabs
    question = re.sub(CONTROL_CHARS_PATTERN, "", question)

    # Normalize whitespace
    question = " ".join(question.split())

    logger.debug(f"Sanitized question: {question[:100]}...")
    return question


def _is_safe_file_name(file_name: str) -> bool:
    """Check if file name is safe (no path traversal attempts)."""
    return ".." not in file_name and "/" not in file_name and "\\" not in file_name


def _find_file_in_catalog(sanitized_name: str, catalog: Dict[str, Any]) -> bool:
    """Check if file exists in catalog."""
    for logical_name, info in catalog.items():
        if info.get("file_name") == sanitized_name or logical_name == sanitized_name:
            return True
    return False


def validate_file_names(file_names: List[str], catalog: Dict[str, Any]) -> List[str]:
    """
    Validate and sanitize file names against catalog.

    Args:
        file_names: List of file names to validate
        catalog: Global catalog dictionary

    Returns:
        List of validated file names

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(file_names, list):
        raise ValidationError("File names must be a list")

    if not file_names:
        raise ValidationError("File names list cannot be empty")

    # Handle wildcard
    if file_names == ["*"]:
        logger.debug("Wildcard file selection - all files allowed")
        return file_names

    validated = []
    invalid = []

    for file_name in file_names:
        if not isinstance(file_name, str):
            invalid.append(str(file_name))
            continue

        # Sanitize file name - remove path traversal attempts
        sanitized = file_name.strip()
        if not _is_safe_file_name(sanitized):
            logger.warning(f"Suspicious file name rejected: {file_name}")
            invalid.append(file_name)
            continue

        # Check if file exists in catalog
        if _find_file_in_catalog(sanitized, catalog):
            validated.append(sanitized)
        else:
            invalid.append(file_name)

    if invalid:
        logger.warning(f"Invalid file names: {invalid}")

    if not validated:
        raise ValidationError(f"No valid files found. Invalid: {invalid}")

    logger.debug(f"Validated {len(validated)} file names")
    return validated


def sanitize_sql_query(query: str, max_length: int = 10000) -> str:
    """
    Sanitize SQL query for logging and display.

    Note: This does NOT make SQL safe for execution - use parameterized
    queries for that. This is only for sanitizing output.

    Args:
        query: SQL query string
        max_length: Maximum allowed length

    Returns:
        Sanitized query string
    """
    if not query or not isinstance(query, str):
        return ""

    # Remove control characters
    query = re.sub(CONTROL_CHARS_PATTERN, "", query)

    # Truncate if too long
    if len(query) > max_length:
        logger.warning(
            f"SQL query truncated from {len(query)} to {max_length} characters"
        )
        query = query[:max_length] + "... [truncated]"

    return query


def _validate_attributes(obj: Any, attrs: List[str], context: str) -> None:
    """Validate that object has required attributes with non-empty values."""
    for attr in attrs:
        if not hasattr(obj, attr):
            raise ValidationError(f"{context} missing: {attr}")

        value = getattr(obj, attr)
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValidationError(f"{context} {attr} is empty")


def validate_config_object(config: Any) -> bool:
    """
    Validate configuration object has required attributes.

    Args:
        config: Configuration object to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    # Validate top-level attributes
    required_attrs = ["azure_openai", "azure_storage"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValidationError(f"Config missing required attribute: {attr}")

    # Validate Azure OpenAI config
    openai_attrs = ["llm_api_key", "llm_endpoint", "llm_deployment_name"]
    _validate_attributes(config.azure_openai, openai_attrs, "Azure OpenAI config")

    # Validate Azure Storage config
    storage_attrs = ["connection_string", "account_name", "container_name"]
    _validate_attributes(config.azure_storage, storage_attrs, "Azure Storage config")

    logger.debug("Configuration validation passed")
    return True


def sanitize_error_message(error: Exception, max_length: int = 500) -> str:
    """
    Sanitize error message for safe logging and display.

    Args:
        error: Exception object
        max_length: Maximum message length

    Returns:
        Sanitized error message
    """
    message = str(error)

    # Remove control characters
    message = re.sub(CONTROL_CHARS_PATTERN, "", message)

    # Truncate if too long
    if len(message) > max_length:
        message = message[:max_length] + "... [truncated]"

    # Remove potential sensitive information patterns using predefined patterns

    for pattern, replacement in SENSITIVE_PATTERNS:
        message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)

    return message
