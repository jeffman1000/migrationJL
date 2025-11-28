"""
Flow compression utilities for identifying and compressing vertical sections.

This module provides functions to identify linear "vertical sections" of recipes
in the flow graph that can be compressed into single operations, reducing complexity.

A compressible vertical section:
- STARTS at a recipe (which may have multiple inputs from sources/recipes)
- CONTINUES through recipes where each has exactly one output consumed by exactly one downstream recipe
- STOPS before encountering another recipe with multiple inputs (boundary recipe, excluded)

The key criterion for a recipe within the section is that its output must be consumed
by exactly one downstream recipe, allowing the chain to continue linearly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets import Dataset
    from .recipes import Recipe


def can_compress_recipe(
    recipe: Recipe, all_recipes: list[Recipe], all_datasets: list[Dataset]
) -> bool:
    """
    Determine if a recipe can be included in a vertical section compression.

    A recipe can be compressed if it meets these criteria:
    1. Has exactly one output dataset
    2. The output dataset is consumed by exactly one downstream recipe

    Note: The recipe may have one OR multiple inputs. Having multiple inputs
    is allowed for the START of a vertical section, but the section stops
    before another recipe with multiple inputs (which becomes the boundary).

    Args:
        recipe: The recipe to check
        all_recipes: List of all recipes in the bundle
        all_datasets: List of all datasets in the bundle

    Returns:
        True if the recipe can be compressed, False otherwise

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT")
        >>> all_recipes = get_recipe_list(bundle)
        >>> all_datasets = get_datasets_list(bundle)
        >>> recipe = Recipe(id="compute_daily_deliv_dates", bundle=bundle)
        >>> can_compress_recipe(recipe, all_recipes, all_datasets)
        True
    """
    # Must have exactly one output
    if len(recipe.outputs) != 1:
        return False

    output_dataset_id = recipe.outputs[0].id

    # Count how many recipes consume this output dataset
    consuming_recipes = [
        r for r in all_recipes if any(inp.id == output_dataset_id for inp in r.inputs)
    ]

    # Output must be consumed by exactly one recipe
    return len(consuming_recipes) == 1
