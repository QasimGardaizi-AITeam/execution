"""
Main entry point for multi-intent query processing
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from graph import create_query_processing_graph
from logging_config import get_logger, setup_logging
from monitoring import MetricsCollector
from state import GraphState
from validation import ValidationError, sanitize_user_question, validate_config_object

# Create logs directory if it doesn't exist
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging with file output
# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"pipeline_{timestamp}.log"

logger = setup_logging(
    log_level="INFO",  # Change to "DEBUG" for more detailed logs
    log_file=str(log_file),
    enable_console=True,  # Set to False if you only want file logging
)

logger.info(f"Logging to file: {log_file}")


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
    from config import get_config
    from response_saver import (
        save_response_to_file,
        save_summary_to_markdown,
        save_tables_to_csv,
    )

    config = get_config()

    # Load catalog
    CATALOG_FILE = "./inputs/Hltachi.json"
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

    test_queries = ["""Can you provide a list of users with the lowest usage?"""]
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
        logger.info("=" * 80)
        logger.info(f"Status: {result['status']}")
        logger.info(f"Total Duration: {result['total_duration']:.2f}s")
        logger.info(f"Total Questions: {result['total_questions']}")
        logger.info(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
        logger.info("=" * 80)

        # Log complete response to file
        logger.info("\n" + "=" * 80)
        logger.info("COMPLETE RESPONSE (JSON)")
        logger.info("=" * 80)

        # Log the full JSON response
        response_json = json.dumps(result, indent=2, default=str)
        logger.info(response_json)

        logger.info("=" * 80)
        logger.info("END OF RESPONSE")
        logger.info("=" * 80)

        # Also print to console
        print(response_json)

        # Save response to files
        try:
            # Save complete JSON
            json_file = save_response_to_file(result, output_dir="./outputs")
            logger.info(f"💾 Saved complete JSON response: {json_file}")

            # Save summary as Markdown
            md_file = save_summary_to_markdown(result, output_dir="./outputs")
            logger.info(f"📄 Saved Markdown summary: {md_file}")

            # Save tables as CSV
            csv_files = save_tables_to_csv(result, output_dir="./outputs")
            if csv_files:
                logger.info(f"📊 Saved {len(csv_files)} CSV tables:")
                for csv_file in csv_files:
                    logger.info(f"   - {csv_file}")
        except Exception as e:
            logger.warning(f"Failed to save response files: {e}")

        logger.info(f"\n✅ Full logs saved to: {log_file}")
        logger.info(
            f"✅ Total tables generated: {len(result.get('final_summary', {}).get('tables', []))}"
        )

        # Log table summaries
        if result.get("final_summary") and result["final_summary"].get("tables"):
            logger.info("\n📊 TABLE SUMMARIES:")
            for idx, table in enumerate(result["final_summary"]["tables"], 1):
                logger.info(f"  Table {idx}: {table.get('title', 'Untitled')}")
                logger.info(f"    Rows: {len(table.get('rows', []))}")
                logger.info(f"    Columns: {len(table.get('headers', []))}")
                logger.info(
                    f"    Description: {table.get('description', 'No description')}"
                )


if __name__ == "__main__":
    main()
