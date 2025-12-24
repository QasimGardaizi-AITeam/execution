"""
Centralized logging configuration for pipeline execution
"""

import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


def redact_sensitive_info(message: str) -> str:
    """
    Redact sensitive information from log messages.

    Automatically removes:
    - API keys
    - Azure connection strings (AccountKey, SAS tokens)
    - Passwords
    - Tokens
    - Email addresses

    Args:
        message: Log message to redact

    Returns:
        Redacted message with sensitive info replaced
    """
    if not isinstance(message, str):
        return str(message)
    REDACTION_REPLACEMENT = r"\1***REDACTED***"

    patterns = [
        # API keys (various formats)
        (r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([^\s"\']{20,})', REDACTION_REPLACEMENT),
        # Azure connection strings
        (r"(AccountKey=)([^;]+)", REDACTION_REPLACEMENT),
        (r"(SharedAccessSignature=)([^;]+)", REDACTION_REPLACEMENT),
        # Passwords
        (r'(password["\']?\s*[:=]\s*["\']?)([^\s"\']+)', REDACTION_REPLACEMENT),
        # Tokens
        (r'(token["\']?\s*[:=]\s*["\']?)([^\s"\']+)', REDACTION_REPLACEMENT),
        # Email addresses
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", r"***EMAIL***"),
    ]

    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)

    return message


class RedactingFormatter(logging.Formatter):
    """Custom formatter that redacts sensitive information from log messages."""

    def format(self, record):
        """Format the log record and redact sensitive information."""
        original = super().format(record)
        return redact_sensitive_info(original)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
) -> logging.Logger:
    """
    Configure logging for the pipeline execution module with PII redaction.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        enable_console: Whether to enable console output

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("pipeline.execution")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter with PII redaction
    formatter = RedactingFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    Get the global logger instance, creating it if necessary.

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


@contextmanager
def track_execution_time(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to track and log execution time of operations.

    Args:
        operation_name: Name of the operation being tracked
        logger: Optional logger instance (uses global logger if not provided)

    Yields:
        Dictionary that will contain the duration after completion

    Example:
        with track_execution_time("SQL Query Execution") as timing:
            result = execute_query(...)
        print(f"Took {timing['duration']:.2f}s")
    """
    if logger is None:
        logger = get_logger()

    timing = {"duration": 0.0}
    start_time = time.time()

    try:
        logger.debug(f"Starting: {operation_name}")
        yield timing
    finally:
        duration = time.time() - start_time
        timing["duration"] = duration
        logger.info(f"{operation_name} completed in {duration:.2f}s")


def log_llm_call(
    logger: logging.Logger,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    duration: Optional[float] = None,
):
    """
    Log LLM API call details in a structured format.

    Args:
        logger: Logger instance
        model: Model name/deployment
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        total_tokens: Total tokens used
        duration: Call duration in seconds
    """
    parts = [f"LLM Call - Model: {model}"]

    if total_tokens is not None:
        parts.append(f"Tokens: {total_tokens}")
        if prompt_tokens is not None and completion_tokens is not None:
            parts.append(f"(prompt: {prompt_tokens}, completion: {completion_tokens})")

    if duration is not None:
        parts.append(f"Duration: {duration:.2f}s")

    logger.info(" | ".join(parts))


def log_query_execution(
    logger: logging.Logger,
    query_index: int,
    sub_question: str,
    status: str,
    duration: float,
    row_count: Optional[int] = None,
    error: Optional[str] = None,
):
    """
    Log query execution results in a structured format.

    Args:
        logger: Logger instance
        query_index: Index of the query
        sub_question: The sub-question being executed
        status: Execution status (success, error, placeholder)
        duration: Execution duration in seconds
        row_count: Number of rows returned (if successful)
        error: Error message (if failed)
    """
    parts = [
        f"Query {query_index + 1}",
        f"Status: {status}",
        f"Duration: {duration:.2f}s",
    ]

    if row_count is not None:
        parts.append(f"Rows: {row_count}")

    if error:
        parts.append(f"Error: {error}")

    log_level = logging.INFO if status == "success" else logging.ERROR
    logger.log(log_level, " | ".join(parts))
    logger.debug(f"Question: {sub_question}")
