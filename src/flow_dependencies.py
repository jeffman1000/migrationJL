"""
Dependency tracing utilities for Dataiku bundles.

This module provides functions to trace dependencies between datasets and recipes,
building complete dependency trees for any dataset in the flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle
    from .recipes import Recipe


@dataclass
class DependencyTree:
    """
    Represents a dependency tree for a dataset.

    This tree shows all upstream dependencies (recipes and datasets)
    needed to build a target dataset.

    Attributes:
        target_dataset: The dataset we're building the tree for
        total_recipes: Total number of recipes in the dependency chain
        total_datasets: Total number of datasets in the dependency chain
        root_datasets: Source datasets (not produced by recipes) in the tree
        max_depth: Maximum depth of the dependency tree
        mermaid_diagram: Mermaid diagram representation for visualization
    """

    target_dataset: str
    total_recipes: int = 0
    total_datasets: int = 0
    root_datasets: list[str] = field(default_factory=list)
    max_depth: int = 0
    mermaid_diagram: str = ""


def trace_dataset_dependencies(
    bundle: DataikuBundle, dataset_id: str
) -> DependencyTree:
    """
    Trace the complete dependency tree for a dataset.

    This function recursively traces backward from a target dataset through
    all recipes and upstream datasets until reaching root datasets (source data).

    The result includes:
    - Total count of recipes and datasets in the dependency chain
    - List of root datasets that must be available
    - Maximum depth of the dependency tree
    - A Mermaid diagram for visualization

    Args:
        bundle: DataikuBundle instance
        dataset_id: ID of the target dataset to trace

    Returns:
        DependencyTree object with complete dependency information

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> tree = trace_dataset_dependencies(bundle, "budget_features")
        >>> print(f"Requires {tree.total_recipes} recipes")
        Requires 15 recipes
        >>> print(f"Root datasets: {tree.root_datasets}")
        Root datasets: ['top_line_budget', 'date_ranges', ...]
        >>> print(tree.mermaid_diagram)  # Save to markdown file
    """
    from .recipes import get_recipe_list

    # Normalize dataset ID to lowercase
    dataset_id = dataset_id.lower()

    # Get all recipes to build lookup maps
    all_recipes = get_recipe_list(bundle)

    # Build dataset -> producer recipe mapping
    dataset_to_producer: dict[str, Recipe] = {}
    for recipe in all_recipes:
        for output in recipe.outputs:
            if hasattr(output, "id"):
                dataset_to_producer[output.id] = recipe

    # Track visited nodes to avoid infinite loops and count unique items
    visited_datasets = set()
    visited_recipes = set()
    root_datasets_found = set()

    # For Mermaid diagram
    mermaid_nodes = []
    mermaid_edges = []
    node_counter = 0
    node_id_map = {}  # Map dataset/recipe IDs to Mermaid node IDs

    def get_node_id(name: str, node_type: str) -> str:
        """Generate a unique node ID for Mermaid."""
        nonlocal node_counter
        key = f"{node_type}:{name}"
        if key not in node_id_map:
            node_id_map[key] = f"node{node_counter}"
            node_counter += 1
        return node_id_map[key]

    def trace_recursive(ds_id: str, depth: int = 0) -> int:
        """
        Recursively trace dependencies.
        Returns the maximum depth reached.
        """
        # Check if already visited
        if ds_id in visited_datasets:
            return depth

        visited_datasets.add(ds_id)

        # Create Mermaid node for this dataset
        ds_node_id = get_node_id(ds_id, "dataset")

        # Check if this dataset is produced by a recipe
        producer_recipe = dataset_to_producer.get(ds_id)

        if producer_recipe is None:
            # This is a root dataset (source data)
            root_datasets_found.add(ds_id)
            mermaid_nodes.append(f'    {ds_node_id}["{ds_id}<br/>📊 SOURCE"]')
            mermaid_nodes.append(f"    style {ds_node_id} fill:#90EE90")
            return depth

        # This dataset is produced by a recipe
        recipe_id = producer_recipe.id
        visited_recipes.add(recipe_id)

        # Create Mermaid node for this dataset (non-root)
        if ds_id == dataset_id:
            # Target dataset
            mermaid_nodes.append(f'    {ds_node_id}["{ds_id}<br/>🎯 TARGET"]')
            mermaid_nodes.append(f"    style {ds_node_id} fill:#FFD700")
        else:
            # Intermediate dataset
            mermaid_nodes.append(f'    {ds_node_id}["{ds_id}<br/>📊 DATASET"]')

        # Create Mermaid node for the recipe
        recipe_node_id = get_node_id(recipe_id, "recipe")
        recipe_type = producer_recipe.type.value if producer_recipe.type else "unknown"
        mermaid_nodes.append(f'    {recipe_node_id}["{recipe_id}<br/>⚙️ {recipe_type}"]')
        mermaid_nodes.append(f"    style {recipe_node_id} fill:#87CEEB")

        # Edge from recipe to dataset (recipe produces dataset)
        mermaid_edges.append(f"    {recipe_node_id} --> {ds_node_id}")

        # Recursively trace inputs
        max_child_depth = depth
        for input_ds in producer_recipe.inputs:
            if hasattr(input_ds, "id"):
                input_node_id = get_node_id(input_ds.id, "dataset")
                # Edge from input dataset to recipe (dataset feeds recipe)
                mermaid_edges.append(f"    {input_node_id} --> {recipe_node_id}")

                # Recurse
                child_depth = trace_recursive(input_ds.id, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    # Start tracing from the target dataset
    max_depth = trace_recursive(dataset_id)

    # Build Mermaid diagram
    mermaid_lines = ["```mermaid", "graph TB"]

    # Add all nodes (deduplicated)
    seen_nodes = set()
    for node in mermaid_nodes:
        if node not in seen_nodes:
            mermaid_lines.append(node)
            seen_nodes.add(node)

    # Add all edges
    for edge in mermaid_edges:
        mermaid_lines.append(edge)

    mermaid_lines.append("```")
    mermaid_diagram = "\n".join(mermaid_lines)

    # Create result
    return DependencyTree(
        target_dataset=dataset_id,
        total_recipes=len(visited_recipes),
        total_datasets=len(visited_datasets),
        root_datasets=sorted(root_datasets_found),
        max_depth=max_depth,
        mermaid_diagram=mermaid_diagram,
    )
