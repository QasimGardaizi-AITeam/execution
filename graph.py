"""
LangGraph workflow definition for multi-intent query processing
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    analyze_query_node,
    execute_sql_query_node,
    execute_summary_search_node,
    finalize_results_node,
    generate_final_summary_node,
    identify_ready_queries_node,
    should_continue_execution,
    validate_input_node,
)
from .state import GraphState


def create_query_processing_graph() -> StateGraph:
    """
    Create the query processing workflow graph.

    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("identify_ready", identify_ready_queries_node)
    workflow.add_node("execute_sql", execute_sql_query_node)
    workflow.add_node("execute_summary", execute_summary_search_node)
    workflow.add_node("finalize", finalize_results_node)
    workflow.add_node("generate_summary", generate_final_summary_node)

    # Define edges
    workflow.set_entry_point("validate_input")

    workflow.add_edge("validate_input", "analyze_query")
    workflow.add_edge("analyze_query", "identify_ready")

    # Conditional routing from identify_ready
    workflow.add_conditional_edges(
        "identify_ready",
        should_continue_execution,
        {
            "continue": "execute_sql",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("execute_sql", "execute_summary")
    workflow.add_edge("execute_summary", "identify_ready")
    workflow.add_edge("finalize", "generate_summary")
    workflow.add_edge("generate_summary", END)

    # Compile graph
    return workflow.compile()


def visualize_graph(graph: StateGraph, output_path: str = "graph.png"):
    """
    Visualize the graph structure.

    Args:
        graph: Compiled StateGraph
        output_path: Path to save visualization
    """
    try:
        from IPython.display import Image, display

        # Generate mermaid diagram
        mermaid_png = graph.get_graph().draw_mermaid_png()

        # Save to file
        with open(output_path, "wb") as f:
            f.write(mermaid_png)

        print(f"Graph visualization saved to {output_path}")

        # Display in notebook
        display(Image(mermaid_png))

    except Exception as e:
        print(f"Could not visualize graph: {e}")
