"""
Main entry point for multi-intent query processing
"""

import json
import time
from typing import Any, Dict

from .graph import create_query_processing_graph
from .state import GraphState


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

    # Create graph
    graph = create_query_processing_graph()

    # Initialize state
    initial_state = GraphState(
        user_question=user_question,
        config=config,
        global_catalog_dict=global_catalog_dict,
        catalog_schema=catalog_schema,
        enable_debug=enable_debug,
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
        }

    except Exception as e:
        error_msg = f"Graph execution failed: {str(e)}"
        print(f"[ERROR] {error_msg}")

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
    CATALOG_FILE = "./pipeline/execution/weeks.json"
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
        print(f"Error loading catalog: {e}")
        return

    # Test queries

    test_queries = [
        "Categorize the questions based on products and response methods (e.g., verbal, email). Provide examples and keep information specific. Identify which products are receiving the most inquiries and categorise the types of questions being asked about each product. Provide a breakdown of the frequency and nature of the questions for each product. Please mention which file you are picking the response from at the start of the response.",
        # "Using Region, Product Category, Q1 Sales, Q2 Sales, Q3 Sales, and Q4 Sales, find the product category in each region that has the highest total annual sales.",
        # "For each region, which product category has the highest yearly sales (Q1+Q2+Q3+Q4)?",
    ]
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {query}")
        print(f"{'='*80}")

        result = process_query(
            user_question=query,
            config=config,
            global_catalog_dict=global_catalog_dict,
            catalog_schema=catalog_schema,
            enable_debug=True,
        )

        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
