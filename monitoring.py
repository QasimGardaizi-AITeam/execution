"""
Telemetry and monitoring utilities for pipeline execution
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger()


@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""

    query_index: int
    sub_question: str
    intent: str
    status: str
    execution_duration: float
    sql_generation_duration: Optional[float] = None
    sql_execution_duration: Optional[float] = None
    retry_count: int = 0
    row_count: Optional[int] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMMetrics:
    """Metrics for LLM API calls"""

    model: str
    operation: str  # "query_analysis", "sql_generation", "summary_generation"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineMetrics:
    """Aggregated metrics for entire pipeline execution"""

    total_duration: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    placeholder_queries: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_rows_returned: int = 0
    query_metrics: List[QueryMetrics] = field(default_factory=list)
    llm_metrics: List[LLMMetrics] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MetricsCollector:
    """
    Centralized metrics collection for pipeline execution.

    Usage:
        collector = MetricsCollector()
        collector.record_query_execution(...)
        collector.record_llm_call(...)
        metrics = collector.get_metrics()
    """

    def __init__(self):
        self.pipeline_metrics = PipelineMetrics()
        self.start_time = time.time()
        logger.debug("Metrics collector initialized")

    def record_query_execution(
        self,
        query_index: int,
        sub_question: str,
        intent: str,
        status: str,
        execution_duration: float,
        row_count: Optional[int] = None,
        error: Optional[str] = None,
        retry_count: int = 0,
    ):
        """Record metrics for a query execution"""
        metrics = QueryMetrics(
            query_index=query_index,
            sub_question=sub_question,
            intent=intent,
            status=status,
            execution_duration=execution_duration,
            row_count=row_count,
            error=error,
            retry_count=retry_count,
        )

        self.pipeline_metrics.query_metrics.append(metrics)
        self.pipeline_metrics.total_queries += 1

        if status == "success":
            self.pipeline_metrics.successful_queries += 1
            if row_count:
                self.pipeline_metrics.total_rows_returned += row_count
        elif status == "error":
            self.pipeline_metrics.failed_queries += 1
            if error:
                self.pipeline_metrics.errors.append(error)
        elif status == "placeholder":
            self.pipeline_metrics.placeholder_queries += 1

        logger.debug(
            f"Recorded query metrics: Q{query_index + 1} - {status} - {execution_duration:.2f}s"
        )

    def record_llm_call(
        self,
        model: str,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        duration: float,
    ):
        """Record metrics for an LLM API call"""
        metrics = LLMMetrics(
            model=model,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration=duration,
        )

        self.pipeline_metrics.llm_metrics.append(metrics)
        self.pipeline_metrics.total_llm_calls += 1
        self.pipeline_metrics.total_tokens_used += total_tokens
        self.pipeline_metrics.total_prompt_tokens += prompt_tokens
        self.pipeline_metrics.total_completion_tokens += completion_tokens

        logger.debug(
            f"Recorded LLM metrics: {operation} - {total_tokens} tokens - {duration:.2f}s"
        )

    def finalize(self):
        """Finalize metrics collection"""
        self.pipeline_metrics.total_duration = time.time() - self.start_time
        logger.info(
            f"Pipeline execution completed in {self.pipeline_metrics.total_duration:.2f}s"
        )

    def get_metrics(self) -> PipelineMetrics:
        """Get collected metrics"""
        return self.pipeline_metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dictionary"""
        return {
            "total_duration": self.pipeline_metrics.total_duration,
            "total_queries": self.pipeline_metrics.total_queries,
            "successful_queries": self.pipeline_metrics.successful_queries,
            "failed_queries": self.pipeline_metrics.failed_queries,
            "placeholder_queries": self.pipeline_metrics.placeholder_queries,
            "success_rate": (
                self.pipeline_metrics.successful_queries
                / self.pipeline_metrics.total_queries
                if self.pipeline_metrics.total_queries > 0
                else 0.0
            ),
            "total_llm_calls": self.pipeline_metrics.total_llm_calls,
            "total_tokens_used": self.pipeline_metrics.total_tokens_used,
            "total_prompt_tokens": self.pipeline_metrics.total_prompt_tokens,
            "total_completion_tokens": self.pipeline_metrics.total_completion_tokens,
            "total_rows_returned": self.pipeline_metrics.total_rows_returned,
            "average_query_duration": (
                sum(m.execution_duration for m in self.pipeline_metrics.query_metrics)
                / len(self.pipeline_metrics.query_metrics)
                if self.pipeline_metrics.query_metrics
                else 0.0
            ),
            "average_llm_duration": (
                sum(m.duration for m in self.pipeline_metrics.llm_metrics)
                / len(self.pipeline_metrics.llm_metrics)
                if self.pipeline_metrics.llm_metrics
                else 0.0
            ),
            "error_count": len(self.pipeline_metrics.errors),
        }

    def log_summary(self):
        """Log metrics summary"""
        summary = self.get_summary()

        logger.info("PIPELINE EXECUTION METRICS SUMMARY")

        logger.info(f"Total Duration: {summary['total_duration']:.2f}s")
        logger.info(f"Total Queries: {summary['total_queries']}")
        logger.info(
            f"Success: {summary['successful_queries']}, "
            f"Failed: {summary['failed_queries']}, "
            f"Placeholder: {summary['placeholder_queries']}"
        )
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Total LLM Calls: {summary['total_llm_calls']}")
        logger.info(f"Total Tokens Used: {summary['total_tokens_used']:,}")
        logger.info(
            f"  Prompt: {summary['total_prompt_tokens']:,}, "
            f"Completion: {summary['total_completion_tokens']:,}"
        )
        logger.info(f"Total Rows Returned: {summary['total_rows_returned']:,}")
        logger.info(f"Average Query Duration: {summary['average_query_duration']:.2f}s")
        logger.info(f"Average LLM Duration: {summary['average_llm_duration']:.2f}s")

        if summary["error_count"] > 0:
            logger.warning(f"Errors Encountered: {summary['error_count']}")


class PerformanceMonitor:
    """
    Monitor performance metrics and detect anomalies.

    Usage:
        monitor = PerformanceMonitor()
        monitor.check_query_duration(duration)
        monitor.check_token_usage(tokens)
    """

    def __init__(
        self,
        max_query_duration: float = 60.0,
        max_tokens_per_call: int = 10000,
        max_total_tokens: int = 100000,
    ):
        self.max_query_duration = max_query_duration
        self.max_tokens_per_call = max_tokens_per_call
        self.max_total_tokens = max_total_tokens
        self.total_tokens_used = 0
        self.warnings = []

    def check_query_duration(self, duration: float, query_index: int) -> bool:
        """Check if query duration is within acceptable limits"""
        if duration > self.max_query_duration:
            warning = (
                f"Query {query_index + 1} took {duration:.2f}s "
                f"(threshold: {self.max_query_duration}s)"
            )
            logger.warning(warning)
            self.warnings.append(warning)
            return False
        return True

    def check_token_usage(self, tokens: int, operation: str) -> bool:
        """Check if token usage is within acceptable limits"""
        self.total_tokens_used += tokens

        if tokens > self.max_tokens_per_call:
            warning = (
                f"{operation} used {tokens:,} tokens "
                f"(threshold: {self.max_tokens_per_call:,})"
            )
            logger.warning(warning)
            self.warnings.append(warning)
            return False

        if self.total_tokens_used > self.max_total_tokens:
            warning = (
                f"Total tokens used: {self.total_tokens_used:,} "
                f"(threshold: {self.max_total_tokens:,})"
            )
            logger.warning(warning)
            self.warnings.append(warning)
            return False

        return True

    def get_warnings(self) -> List[str]:
        """Get all warnings"""
        return self.warnings
