"""
Flow analysis utilities for Dataiku bundles.

This module provides functions to analyze the flow (DAG) structure,
identifying root datasets (source data), leaf datasets (final outputs),
and intermediate datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle
    from .datasets import Dataset


def get_root_datasets(bundle: DataikuBundle) -> list[Dataset]:
    """
    Get all root datasets (source data) in the bundle.

    Root datasets are those that are NEVER produced by any recipe.
    They are the starting points of the flow - typically uploaded files,
    external database tables, or manually created datasets.

    These datasets are consumed by recipes but never appear as recipe outputs.

    Args:
        bundle: DataikuBundle instance

    Returns:
        List of Dataset objects that are root/source datasets, sorted by ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> roots = get_root_datasets(bundle)
        >>> len(roots)
        13
        >>> roots[0].id
        'date_ranges'
    """
    from .datasets import get_datasets_list
    from .recipes import get_recipe_list

    # Get all datasets and recipes
    all_datasets = get_datasets_list(bundle)
    all_recipes = get_recipe_list(bundle)

    # Track which datasets are produced by recipes
    datasets_produced_by_recipes = set()

    for recipe in all_recipes:
        for output in recipe.outputs:
            if hasattr(output, "id"):
                datasets_produced_by_recipes.add(output.id)

    # Root datasets are those NOT produced by any recipe
    root_datasets = [
        dataset
        for dataset in all_datasets
        if dataset.id not in datasets_produced_by_recipes
    ]

    return sorted(root_datasets, key=lambda d: d.id)


def get_leaf_datasets(bundle: DataikuBundle) -> list[Dataset]:
    """
    Get all leaf datasets (final outputs) in the bundle.

    Leaf datasets are those that are NEVER consumed by any recipe.
    They are the end points of the flow - typically final results,
    reports, or outputs that are exported/used outside Dataiku.

    These datasets are produced by recipes but never appear as recipe inputs.

    Args:
        bundle: DataikuBundle instance

    Returns:
        List of Dataset objects that are leaf/final datasets, sorted by ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> leaves = get_leaf_datasets(bundle)
        >>> len(leaves)
        29
        >>> leaves[0].id
        'estate_output'
    """
    from .datasets import get_datasets_list
    from .recipes import get_recipe_list

    # Get all datasets and recipes
    all_datasets = get_datasets_list(bundle)
    all_recipes = get_recipe_list(bundle)

    # Track which datasets are consumed by recipes
    datasets_consumed_by_recipes = set()

    for recipe in all_recipes:
        for input_ds in recipe.inputs:
            if hasattr(input_ds, "id"):
                datasets_consumed_by_recipes.add(input_ds.id)

    # Leaf datasets are those NOT consumed by any recipe
    leaf_datasets = [
        dataset
        for dataset in all_datasets
        if dataset.id not in datasets_consumed_by_recipes
    ]

    return sorted(leaf_datasets, key=lambda d: d.id)


def get_intermediate_datasets(bundle: DataikuBundle) -> list[Dataset]:
    """
    Get all intermediate datasets in the bundle.

    Intermediate datasets are those that are both produced AND consumed by recipes.
    They are temporary/intermediate results in the flow pipeline.

    Args:
        bundle: DataikuBundle instance

    Returns:
        List of Dataset objects that are intermediate datasets, sorted by ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> intermediate = get_intermediate_datasets(bundle)
        >>> len(intermediate)
        92
    """
    from .datasets import get_datasets_list
    from .recipes import get_recipe_list

    # Get all datasets and recipes
    all_datasets = get_datasets_list(bundle)
    all_recipes = get_recipe_list(bundle)

    # Track which datasets are produced and consumed
    datasets_produced = set()
    datasets_consumed = set()

    for recipe in all_recipes:
        for output in recipe.outputs:
            if hasattr(output, "id"):
                datasets_produced.add(output.id)

        for input_ds in recipe.inputs:
            if hasattr(input_ds, "id"):
                datasets_consumed.add(input_ds.id)

    # Intermediate datasets are both produced AND consumed
    intermediate_dataset_ids = datasets_produced & datasets_consumed

    intermediate_datasets = [
        dataset for dataset in all_datasets if dataset.id in intermediate_dataset_ids
    ]

    return sorted(intermediate_datasets, key=lambda d: d.id)


def get_orphaned_datasets(bundle: DataikuBundle) -> list[Dataset]:
    """
    Get all orphaned datasets in the bundle.

    Orphaned datasets are those that are NEVER used by any recipe
    (neither as input nor as output). These might be:
    - Unused test datasets
    - Legacy datasets no longer in use
    - Manually created datasets for ad-hoc analysis

    Args:
        bundle: DataikuBundle instance

    Returns:
        List of Dataset objects that are orphaned, sorted by ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> orphaned = get_orphaned_datasets(bundle)
        >>> len(orphaned)
        7
    """
    from .datasets import get_datasets_list
    from .recipes import get_recipe_list

    # Get all datasets and recipes
    all_datasets = get_datasets_list(bundle)
    all_recipes = get_recipe_list(bundle)

    # Track which datasets are used in any way
    datasets_used = set()

    for recipe in all_recipes:
        for output in recipe.outputs:
            if hasattr(output, "id"):
                datasets_used.add(output.id)

        for input_ds in recipe.inputs:
            if hasattr(input_ds, "id"):
                datasets_used.add(input_ds.id)

    # Orphaned datasets are those never used
    orphaned_datasets = [
        dataset for dataset in all_datasets if dataset.id not in datasets_used
    ]

    return sorted(orphaned_datasets, key=lambda d: d.id)


def get_recipes_by_phase(bundle: DataikuBundle) -> dict[int, list]:
    """
    Organize recipes by dependency phases.

    Phase 1: Recipes that only consume root datasets (source data)
    Phase 2: Recipes that consume outputs from Phase 1
    Phase N: Recipes that consume outputs from Phase N-1
    ...and so on

    This organization is useful for building projects incrementally, as recipes
    in phase N can only be executed after all recipes in phase N-1 are complete.

    Args:
        bundle: DataikuBundle instance

    Returns:
        Dictionary mapping phase number (1-based) to list of Recipe objects
        Recipes within each phase are sorted by ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> phases = get_recipes_by_phase(bundle)
        >>> len(phases[1])  # Number of recipes in Phase 1
        15
        >>> phases[1][0].id  # First recipe in Phase 1
        'compute_capacity'
    """
    from .recipes import Recipe, get_recipe_list

    # Get all recipes and root datasets
    all_recipes = get_recipe_list(bundle)
    root_datasets = get_root_datasets(bundle)
    root_dataset_ids = {ds.id for ds in root_datasets}

    # Build a mapping of dataset_id -> recipe that produces it
    dataset_to_producer: dict[str, Recipe] = {}
    for recipe in all_recipes:
        for output in recipe.outputs:
            if hasattr(output, "id"):
                dataset_to_producer[output.id] = recipe

    # Track which recipes have been assigned to phases
    recipe_to_phase: dict[str, int] = {}
    remaining_recipes = set(r.id for r in all_recipes)

    phase = 1

    # Keep assigning recipes to phases until all are assigned
    while remaining_recipes:
        current_phase_recipes = []

        for recipe in all_recipes:
            if recipe.id not in remaining_recipes:
                continue

            # Check if all input datasets are either:
            # 1. Root datasets (not produced by recipes), OR
            # 2. Produced by recipes in earlier phases
            can_be_placed = True

            for input_ds in recipe.inputs:
                if not hasattr(input_ds, "id"):
                    continue

                input_id = input_ds.id

                # If it's a root dataset, it's always available
                if input_id in root_dataset_ids:
                    continue

                # If it's produced by a recipe, that recipe must be in an earlier phase
                producer = dataset_to_producer.get(input_id)
                if producer is None:
                    # Input dataset doesn't exist or is orphaned - treat as root
                    continue

                # Check if the producer has been assigned to an earlier phase
                producer_phase = recipe_to_phase.get(producer.id)
                if producer_phase is None or producer_phase >= phase:
                    # Producer hasn't been placed yet, or is in current/later phase
                    can_be_placed = False
                    break

            if can_be_placed:
                current_phase_recipes.append(recipe)
                recipe_to_phase[recipe.id] = phase
                remaining_recipes.remove(recipe.id)

        # If no recipes were placed in this phase, we might have cycles or orphaned recipes
        if not current_phase_recipes:
            # Place any remaining recipes in a final "unresolved" phase
            for recipe_id in remaining_recipes:
                recipe = next(r for r in all_recipes if r.id == recipe_id)
                current_phase_recipes.append(recipe)
                recipe_to_phase[recipe_id] = phase
            remaining_recipes.clear()

        phase += 1

    # Build the result dictionary
    phases: dict[int, list[Recipe]] = {}
    for recipe in all_recipes:
        phase_num = recipe_to_phase[recipe.id]
        if phase_num not in phases:
            phases[phase_num] = []
        phases[phase_num].append(recipe)

    # Sort recipes within each phase by ID
    for phase_num in phases:
        phases[phase_num] = sorted(phases[phase_num], key=lambda r: r.id)

    return phases


def get_vertical_sections(
    all_recipes: list, all_datasets: list
) -> tuple[list[list], set[str], set[str]]:
    """
    Identify all compressible vertical sections in the flow.

    A vertical section is a linear chain of recipes where each recipe has exactly
    one input and one output, forming a compressible sequence. This function
    scans the entire flow to find all such sections.

    When multiple vertical sections overlap (one is a subset of another), only
    the longest section is kept to avoid duplication.

    Args:
        all_recipes: List of all Recipe objects in the flow
        all_datasets: List of all Dataset objects in the flow

    Returns:
        Tuple of (vertical_sections, boundary_recipe_ids, recipe_ids_in_sections) where:
        - vertical_sections: List of lists of recipe IDs forming vertical sections
        - boundary_recipe_ids: Set of recipe IDs that are boundaries (excluded from compression)
        - recipe_ids_in_sections: Set of all recipe IDs that are in vertical sections

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> all_recipes = get_recipe_list(bundle)
        >>> all_datasets = get_datasets_list(bundle)
        >>> sections, boundaries, in_sections = get_vertical_sections(all_recipes, all_datasets)
        >>> len(sections)  # Number of vertical sections found
        5
        >>> len(in_sections)  # Total recipes in all sections
        23
    """
    from .vertical_section import find_vertical_section

    vertical_sections = []
    boundary_recipe_ids = set()

    # Try each recipe as a potential start of a vertical section
    for recipe in all_recipes:
        # Find vertical section starting from this recipe
        compressible, boundary = find_vertical_section(
            recipe, all_recipes, all_datasets
        )

        # Only consider it a valid section if it has 2+ recipes
        if len(compressible) >= 2:
            section_ids = [r.id for r in compressible]
            vertical_sections.append(section_ids)

            # Track boundary recipe if exists
            if boundary:
                boundary_recipe_ids.add(boundary.id)

    # Remove overlapping sections - keep only the longest ones
    # Sort sections by length (descending) to prioritize longer sections
    vertical_sections.sort(key=len, reverse=True)

    non_overlapping_sections = []
    used_recipes = set()

    for section in vertical_sections:
        section_set = set(section)
        # Check if this section overlaps with any already selected section
        if not section_set & used_recipes:
            # No overlap - keep this section
            non_overlapping_sections.append(section)
            used_recipes.update(section_set)

    # Collect all recipe IDs that are in the final non-overlapping sections
    recipe_ids_in_sections = set()
    for section in non_overlapping_sections:
        recipe_ids_in_sections.update(section)

    return non_overlapping_sections, boundary_recipe_ids, recipe_ids_in_sections
