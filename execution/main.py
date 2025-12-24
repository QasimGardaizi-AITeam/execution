"""
Main entry point for multi-intent query processing
"""

import json
import time
from typing import Any, Dict

from .graph import create_query_processing_graph
from .logging_config import get_logger
from .monitoring import MetricsCollector
from .state import GraphState
from .validation import ValidationError, sanitize_user_question, validate_config_object

logger = get_logger()


def process_query(
    user_question: str,
    config: Any,
    global_catalog_dict: Dict[str, Any],
    catalog_schema: str,
    enable_debug: bool = False,
) -> Dict[str, Any]:
    """
    Process a multi-intent query using LangGraph.

    Args:
        user_question: User's question
        config: Configuration object
        global_catalog_dict: Global catalog dictionary
        catalog_schema: JSON string of catalog
        enable_debug: Enable debug output

    Returns:
        Dictionary with results and metadata
    """
    start_time = time.time()

    # Validate input
    try:
        user_question = sanitize_user_question(user_question)
        validate_config_object(config)
        logger.info("Input validation passed")
    except ValidationError as e:
        error_msg = f"Validation failed: {str(e)}"
        logger.error(error_msg)
        return {
            "original_question": user_question,
            "total_questions": 0,
            "independent_count": 0,
            "dependent_count": 0,
            "results": [],
            "final_summary": None,
            "total_duration": time.time() - start_time,
            "status": "error",
            "error": error_msg,
            "messages": [error_msg],
        }

    # Create graph
    graph = create_query_processing_graph()

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Initialize state
    initial_state = GraphState(
        user_question=user_question,
        config=config,
        global_catalog_dict=global_catalog_dict,
        catalog_schema=catalog_schema,
        enable_debug=enable_debug,
        metrics_collector=metrics_collector,
        analyses=[],
        total_questions=0,
        independent_count=0,
        dependent_count=0,
        executed_results={},
        remaining_indices=[],
        current_batch=[],
        final_results=[],
        final_summary=None,
        total_duration=0.0,
        status="processing",
        error=None,
        messages=[],
    )

    try:
        # Execute graph
        final_state = graph.invoke(initial_state)

        # Calculate total duration
        total_duration = time.time() - start_time
        final_state["total_duration"] = total_duration

        # Finalize and log metrics
        metrics_collector.finalize()
        if enable_debug:
            metrics_collector.log_summary()

        return {
            "original_question": user_question,
            "total_questions": final_state.get("total_questions", 0),
            "independent_count": final_state.get("independent_count", 0),
            "dependent_count": final_state.get("dependent_count", 0),
            "results": final_state.get("final_results", []),
            "final_summary": final_state.get("final_summary"),
            "total_duration": total_duration,
            "status": final_state.get("status", "unknown"),
            "error": final_state.get("error"),
            "messages": final_state.get("messages", []),
            "metrics": metrics_collector.get_summary(),
        }

    except Exception as e:
        error_msg = f"Graph execution failed: {str(e)}"
        logger.error(error_msg)

        return {
            "original_question": user_question,
            "total_questions": 0,
            "independent_count": 0,
            "dependent_count": 0,
            "results": [],
            "final_summary": None,
            "total_duration": time.time() - start_time,
            "status": "error",
            "error": error_msg,
            "messages": [error_msg],
        }


def main():
    """Example usage"""
    from ..config import VectorDBType, get_config

    # Load configuration
    config = get_config(VectorDBType.CHROMADB)

    # Load catalog
    CATALOG_FILE = "./pipeline/execution/asad.json"
    try:
        with open(CATALOG_FILE, "r") as f:
            catalog_data = json.load(f)
            catalog_schema = json.dumps(catalog_data, indent=2)

            # Build catalog dict
            global_catalog_dict = {}
            if isinstance(catalog_data, list):
                for item in catalog_data:
                    file_name = item.get("file_name", "")
                    if file_name:
                        global_catalog_dict[file_name] = item
            elif isinstance(catalog_data, dict):
                global_catalog_dict = catalog_data

    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        return

    # Test queries

    test_queries = [
        # "Categorize the questions based on products and response methods (e.g., verbal, email). Provide examples and keep information specific. Identify which products are receiving the most inquiries and categorise the types of questions being asked about each product. Provide a breakdown of the frequency and nature of the questions for each product. Please mention which file you are picking the response from at the start of the response.",
        "Using Region, Product Category, Q1 Sales, Q2 Sales, Q3 Sales, and Q4 Sales, find the product category in each region that has the highest total annual sales.",
        # "For each region, which product category has the highest yearly sales (Q1+Q2+Q3+Q4)?",
    ]
    for query in test_queries:
        logger.info(f"PROCESSING: {query}")

        result = process_query(
            user_question=query,
            config=config,
            global_catalog_dict=global_catalog_dict,
            catalog_schema=catalog_schema,
            enable_debug=True,
        )

        logger.info("FINAL RESULTS")

        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
