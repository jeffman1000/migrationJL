"""
Vertical section identification for flow compression.

This module provides functionality to find and trace vertical sections
in the Dataiku flow graph - linear chains of recipes that can be
compressed into single operations.

A compressible vertical section:
- STARTS at a recipe (which may have multiple inputs from sources/recipes)
- CONTINUES through recipes where each has exactly one output consumed by exactly one downstream recipe
- STOPS before encountering another recipe with multiple inputs (boundary recipe, excluded)

The key criterion for continuing the section is that each recipe's output must be
consumed by exactly one downstream recipe, creating a linear chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets import Dataset
    from .recipes import Recipe


def find_vertical_section(
    start_recipe: Recipe, all_recipes: list[Recipe], all_datasets: list[Dataset]
) -> tuple[list[Recipe], Recipe | None]:
    """
    Find a vertical section starting from a recipe.

    Traces forward through the flow to find all recipes that form a compressible
    linear chain. The start recipe is included even if it has multiple inputs.
    Each subsequent recipe must have exactly one output consumed by exactly one
    downstream recipe.

    Tracing stops when encountering a downstream recipe that has multiple inputs
    (this becomes the boundary recipe and is excluded from the compressible section).

    Args:
        start_recipe: The recipe to start from (included even if multiple inputs)
        all_recipes: List of all recipes in the bundle
        all_datasets: List of all datasets in the bundle

    Returns:
        Tuple of (compressible_recipes, boundary_recipe) where:
        - compressible_recipes: List of recipes that can be compressed together
        - boundary_recipe: The first downstream recipe with multiple inputs (excluded, or None)

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT")
        >>> all_recipes = get_recipe_list(bundle)
        >>> all_datasets = get_datasets_list(bundle)
        >>> start = Recipe(id="compute_daily_deliv_dates", bundle=bundle)
        >>> compressible, boundary = find_vertical_section(start, all_recipes, all_datasets)
        >>> len(compressible)
        3
        >>> boundary.id
        'compute_budget_and_delivery'
        >>> len(boundary.inputs)
        2
    """
    from .flow_compression import can_compress_recipe

    compressible = []
    current_recipe = start_recipe

    while current_recipe is not None:
        # Check if this recipe can be part of the compressible section
        if can_compress_recipe(current_recipe, all_recipes, all_datasets):
            # If this is not the first recipe and it has multiple inputs, it's a boundary
            if len(compressible) > 0 and len(current_recipe.inputs) > 1:
                # This recipe has multiple inputs - it's the boundary, exclude it
                return compressible, current_recipe

            # Add to compressible section
            compressible.append(current_recipe)

            # Find the next recipe in the chain
            output_dataset_id = current_recipe.outputs[0].id
            next_recipes = [
                r
                for r in all_recipes
                if any(inp.id == output_dataset_id for inp in r.inputs)
            ]

            if len(next_recipes) == 1:
                current_recipe = next_recipes[0]
            else:
                current_recipe = None
        else:
            # This recipe cannot be compressed (e.g., multiple outputs or no downstream consumer)
            return compressible, current_recipe

    return compressible, None
