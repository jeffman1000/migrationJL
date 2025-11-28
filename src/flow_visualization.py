"""
Flow visualization utilities for Dataiku bundles.

This module provides functions to generate complete flow visualizations
as Mermaid diagrams in markdown format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle
    from .recipes import Recipe


@dataclass
class FlowVisualizationConfig:
    """
    Configuration for complete flow visualization.

    Customize colors, icons, and display options for the Mermaid diagram.

    Attributes:
        root_color: Fill color for root datasets (source data)
        leaf_color: Fill color for leaf datasets (final outputs)
        intermediate_color: Fill color for intermediate datasets
        orphaned_color: Fill color for orphaned datasets (unused)
        recipe_color: Fill color for recipes
        vertical_section_color: Fill color for recipes in compressible vertical sections
        boundary_recipe_color: Fill color for boundary recipes (excluded from compression)
        show_dataset_columns: Whether to show column names in dataset nodes
        max_columns_display: Maximum number of columns to show per dataset
        show_recipe_type: Whether to show recipe type in recipe nodes
        include_orphaned: Whether to include orphaned datasets in diagram
        highlight_vertical_sections: Whether to highlight compressible vertical sections
        use_subgraphs_for_sections: Whether to group vertical sections in subgraphs
        graph_direction: Mermaid graph direction (TB=top-bottom, LR=left-right)
    """

    root_color: str = "#90EE90"  # Light green
    leaf_color: str = "#FFD700"  # Gold
    intermediate_color: str = "#E8E8E8"  # Light gray
    orphaned_color: str = "#FFB6C6"  # Light red
    recipe_color: str = "#87CEEB"  # Sky blue
    vertical_section_color: str = "#FFA500"  # Orange
    boundary_recipe_color: str = "#DC143C"  # Crimson red
    show_dataset_columns: bool = True
    max_columns_display: int = 5
    show_recipe_type: bool = True
    include_orphaned: bool = True
    highlight_vertical_sections: bool = True
    use_subgraphs_for_sections: bool = True
    graph_direction: str = "TB"  # Top to bottom


def generate_complete_flow_visualization(
    bundle: DataikuBundle, config: FlowVisualizationConfig | None = None
) -> str:
    """
    Generate a complete Mermaid diagram of the entire flow and save to markdown.

    This function creates a comprehensive visualization of all datasets and recipes
    in the bundle, color-coded by type (root, leaf, intermediate, orphaned).

    The output markdown file is named after the project key and saved in the
    current working directory.

    Args:
        bundle: DataikuBundle instance
        config: Optional FlowVisualizationConfig for customization

    Returns:
        Path to the generated markdown file

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> config = FlowVisualizationConfig(show_dataset_columns=True)
        >>> filepath = generate_complete_flow_visualization(bundle, config)
        >>> print(f"Saved to: {filepath}")
        Saved to: MYPROJECT_flow.md
    """
    from .datasets import get_datasets_list
    from .flow_analysis import (
        get_intermediate_datasets,
        get_leaf_datasets,
        get_orphaned_datasets,
        get_root_datasets,
        get_vertical_sections,
    )
    from .managed_folders import get_managed_folders_list
    from .recipes import get_recipe_list

    # Use default config if none provided
    if config is None:
        config = FlowVisualizationConfig()

    # Get all components
    all_datasets = get_datasets_list(bundle)
    all_recipes = get_recipe_list(bundle)
    all_managed_folders = get_managed_folders_list(bundle)

    # Categorize datasets
    root_datasets_list = get_root_datasets(bundle)
    leaf_datasets_list = get_leaf_datasets(bundle)
    intermediate_datasets_list = get_intermediate_datasets(bundle)
    orphaned_datasets_list = get_orphaned_datasets(bundle)

    # Convert to sets for fast lookup
    root_ids = {d.id for d in root_datasets_list}
    leaf_ids = {d.id for d in leaf_datasets_list}
    intermediate_ids = {d.id for d in intermediate_datasets_list}
    orphaned_ids = {d.id for d in orphaned_datasets_list}

    # Identify vertical sections if configured
    vertical_sections = []
    boundary_recipes = set()
    recipes_in_sections = set()

    if config.highlight_vertical_sections:
        vertical_sections, boundary_recipes, recipes_in_sections = (
            get_vertical_sections(all_recipes, all_datasets)
        )

    # Build lookup maps
    dataset_to_producer: dict[str, Recipe] = {}
    for recipe in all_recipes:
        for output in recipe.outputs:
            if hasattr(output, "id"):
                dataset_to_producer[output.id] = recipe

    # Generate Mermaid diagram
    mermaid_diagram = _generate_mermaid_diagram(
        all_datasets=all_datasets,
        all_recipes=all_recipes,
        all_managed_folders=all_managed_folders,
        root_ids=root_ids,
        leaf_ids=leaf_ids,
        intermediate_ids=intermediate_ids,
        orphaned_ids=orphaned_ids,
        vertical_sections=vertical_sections,
        boundary_recipes=boundary_recipes,
        recipes_in_sections=recipes_in_sections,
        config=config,
    )

    # Extract project key from bundle name
    project_key = (
        bundle.path.name.split("-")[2] if "-" in bundle.path.name else "PROJECT"
    )

    # Filter datasets for template
    root_datasets_filtered = [d for d in root_datasets_list if d.id not in orphaned_ids]
    leaf_datasets_filtered = [d for d in leaf_datasets_list if d.id not in orphaned_ids]

    # Get project and local variables
    project_variables = bundle.variables.project
    local_variables = bundle.variables.local

    # Get recipes organized by dependency phase
    from .flow_analysis import get_recipes_by_phase

    recipes_by_phase = get_recipes_by_phase(bundle)

    # Get scenario YAML data
    from .scenarios import get_scenario_yaml

    scenario_yaml_data = get_scenario_yaml(bundle)

    # Prepare template data
    template_data = {
        "project_key": project_key,
        "stats": {
            "total_datasets": len(all_datasets),
            "total_recipes": len(all_recipes),
            "total_managed_folders": len(all_managed_folders),
            "root_datasets": len(root_datasets_filtered),
            "intermediate_datasets": len(intermediate_datasets_list),
            "leaf_datasets": len(leaf_datasets_filtered),
            "orphaned_datasets": len(orphaned_datasets_list),
            "total_phases": len(recipes_by_phase),
            "vertical_sections": (
                len(vertical_sections) if config.highlight_vertical_sections else 0
            ),
            "recipes_in_sections": (
                len(recipes_in_sections) if config.highlight_vertical_sections else 0
            ),
            "boundary_recipes": (
                len(boundary_recipes) if config.highlight_vertical_sections else 0
            ),
        },
        "config": config,
        "root_datasets": sorted(root_datasets_filtered, key=lambda d: d.id),
        "leaf_datasets": sorted(leaf_datasets_filtered, key=lambda d: d.id),
        "orphaned_datasets": sorted(orphaned_datasets_list, key=lambda d: d.id),
        "managed_folders": sorted(all_managed_folders, key=lambda mf: mf.name or mf.id),
        "recipes_by_phase": recipes_by_phase,
        "vertical_sections": (
            vertical_sections if config.highlight_vertical_sections else []
        ),
        "mermaid_diagram": mermaid_diagram,
        "project_variables": project_variables,
        "local_variables": local_variables,
        "scenario_yaml_data": scenario_yaml_data,
    }

    # Render template
    markdown_content = _render_template(template_data)

    # Save to file
    output_filename = f"{project_key}_flow.md"
    output_path = Path.cwd() / output_filename

    with open(output_path, "w") as f:
        f.write(markdown_content)

    return str(output_path)


def _generate_mermaid_diagram(
    all_datasets: list,
    all_recipes: list,
    all_managed_folders: list,
    root_ids: set[str],
    leaf_ids: set[str],
    intermediate_ids: set[str],
    orphaned_ids: set[str],
    vertical_sections: list[list],
    boundary_recipes: set[str],
    recipes_in_sections: set[str],
    config: FlowVisualizationConfig,
) -> str:
    """Generate the Mermaid diagram string."""
    # Start building Mermaid diagram
    mermaid_lines = ["```mermaid", f"graph {config.graph_direction}"]

    # Node ID mapping
    node_counter = 0
    node_id_map = {}

    # Build a set of managed folder IDs (lowercase) for quick lookup
    managed_folder_ids = {mf.id.lower() for mf in all_managed_folders}

    # Track which recipes are in which section for subgraph generation
    recipe_to_section = {}
    if config.use_subgraphs_for_sections and vertical_sections:
        for section_idx, section_recipe_ids in enumerate(vertical_sections):
            for recipe_id in section_recipe_ids:
                recipe_to_section[recipe_id] = section_idx

    def get_node_id(name: str, node_type: str) -> str:
        """Generate unique node ID for Mermaid."""
        nonlocal node_counter
        key = f"{node_type}:{name}"
        if key not in node_id_map:
            node_id_map[key] = f"node{node_counter}"
            node_counter += 1
        return node_id_map[key]

    # Add dataset nodes - ONLY root, leaf, and optionally orphaned
    # Intermediate datasets are excluded for diagram clarity
    for dataset in all_datasets:
        ds_id = dataset.id

        # Skip intermediate datasets - they clutter the diagram
        if ds_id in intermediate_ids:
            continue

        # Skip orphaned if not configured to include
        if ds_id in orphaned_ids and not config.include_orphaned:
            continue

        node_id = get_node_id(ds_id, "dataset")

        # Build node label
        label_parts = [ds_id]

        # Add columns if configured
        if config.show_dataset_columns and ds_id in root_ids:
            # Get column names for root datasets
            columns = dataset.columns
            if columns:
                col_names = [col.name for col in columns[: config.max_columns_display]]
                if len(columns) > config.max_columns_display:
                    col_names.append(
                        f"... +{len(columns) - config.max_columns_display} more"
                    )
                label_parts.append("---")
                label_parts.extend(col_names)

        # Determine icon and category
        if ds_id in orphaned_ids:
            icon = "🏝️ ORPHANED"
            color = config.orphaned_color
        elif ds_id in root_ids:
            icon = "📊 SOURCE"
            color = config.root_color
        elif ds_id in leaf_ids:
            icon = "🎯 OUTPUT"
            color = config.leaf_color
        else:
            # Should not reach here due to skip logic above
            icon = "📊 DATASET"
            color = config.intermediate_color

        label_parts.insert(1, icon)

        # Create node
        label = "<br/>".join(label_parts)
        mermaid_lines.append(f'    {node_id}["{label}"]')
        mermaid_lines.append(f"    style {node_id} fill:{color}")

    # Add managed folder nodes
    managed_folder_color = "#DDA0DD"  # Plum color for managed folders
    for managed_folder in all_managed_folders:
        mf_id = managed_folder.id.lower()
        node_id = get_node_id(mf_id, "managed_folder")

        # Build node label
        label_parts = [managed_folder.name or mf_id, "📁 STORAGE"]
        if managed_folder.type:
            label_parts.append(f"({managed_folder.type})")
        if managed_folder.length > 0:
            label_parts.append(f"{managed_folder.length} files")

        label = "<br/>".join(label_parts)
        mermaid_lines.append(f'    {node_id}["{label}"]')
        mermaid_lines.append(f"    style {node_id} fill:{managed_folder_color}")

    def _add_recipe_node(
        recipe,
        mermaid_lines: list[str],
        get_node_id,
        config: FlowVisualizationConfig,
        boundary_recipes: set[str],
        recipes_in_sections: set[str],
        indent: str = "    ",
    ):
        """Helper function to add a recipe node with proper styling."""
        recipe_id = recipe.id
        node_id = get_node_id(recipe_id, "recipe")

        # Build label
        label_parts = [recipe_id]

        if config.show_recipe_type and recipe.type:
            recipe_type = recipe.type.value
            label_parts.append(f"⚙️ {recipe_type}")
        else:
            label_parts.append("⚙️ RECIPE")

        label = "<br/>".join(label_parts)
        mermaid_lines.append(f'{indent}{node_id}["{label}"]')

        # Determine color based on vertical section membership
        if config.highlight_vertical_sections:
            if recipe_id in boundary_recipes:
                # Boundary recipe - red with thick border
                mermaid_lines.append(
                    f"{indent}style {node_id} fill:{config.boundary_recipe_color},stroke:#8B0000,stroke-width:3px"
                )
            elif recipe_id in recipes_in_sections:
                # Recipe in vertical section - orange with thick border
                mermaid_lines.append(
                    f"{indent}style {node_id} fill:{config.vertical_section_color},stroke:#FF4500,stroke-width:3px"
                )
            else:
                # Regular recipe
                mermaid_lines.append(
                    f"{indent}style {node_id} fill:{config.recipe_color}"
                )
        else:
            # No highlighting - use default color
            mermaid_lines.append(f"{indent}style {node_id} fill:{config.recipe_color}")

    # Add recipe nodes
    # If using subgraphs, group vertical section recipes together
    if config.use_subgraphs_for_sections and vertical_sections:
        # First, add recipes that are NOT in vertical sections
        for recipe in all_recipes:
            if recipe.id not in recipes_in_sections:
                _add_recipe_node(
                    recipe,
                    mermaid_lines,
                    get_node_id,
                    config,
                    boundary_recipes,
                    recipes_in_sections,
                    indent="    ",
                )

        # Then add each vertical section as a subgraph
        for section_idx, section_recipe_ids in enumerate(vertical_sections):
            # Create subgraph with unique ID
            subgraph_id = f"section_{section_idx}"
            section_label = f"🔄 Vertical Section {section_idx + 1}"
            mermaid_lines.append(f'    subgraph {subgraph_id}["{section_label}"]')
            mermaid_lines.append(f"        direction {config.graph_direction}")

            # Add recipes in this section
            for recipe_id in section_recipe_ids:
                # Find the recipe object
                recipe = next((r for r in all_recipes if r.id == recipe_id), None)
                if recipe:
                    _add_recipe_node(
                        recipe,
                        mermaid_lines,
                        get_node_id,
                        config,
                        boundary_recipes,
                        recipes_in_sections,
                        indent="        ",
                    )

            mermaid_lines.append("    end")
            # Style the subgraph with dotted border
            mermaid_lines.append(
                f"    style {subgraph_id} fill:#FFF8DC,stroke:#FF4500,stroke-width:4px,stroke-dasharray: 5 5"
            )
    else:
        # No subgraphs - add all recipes normally
        for recipe in all_recipes:
            _add_recipe_node(
                recipe,
                mermaid_lines,
                get_node_id,
                config,
                boundary_recipes,
                recipes_in_sections,
                indent="    ",
            )

    # Build a map of intermediate datasets to their producer recipes
    intermediate_to_producer: dict[str, str] = {}
    for recipe in all_recipes:
        for output in recipe.outputs:
            if hasattr(output, "id") and output.id in intermediate_ids:
                intermediate_to_producer[output.id] = recipe.id

    # Build a map of intermediate datasets to their consumer recipes
    intermediate_to_consumers: dict[str, list[str]] = {}
    for recipe in all_recipes:
        for input_ds in recipe.inputs:
            if hasattr(input_ds, "id") and input_ds.id in intermediate_ids:
                if input_ds.id not in intermediate_to_consumers:
                    intermediate_to_consumers[input_ds.id] = []
                intermediate_to_consumers[input_ds.id].append(recipe.id)

    # Add edges (dataset -> recipe -> dataset)
    # For intermediate datasets, create direct recipe-to-recipe edges
    for recipe in all_recipes:
        recipe_node_id = get_node_id(recipe.id, "recipe")

        # Edges from input datasets/recipes to this recipe
        for input_ds in recipe.inputs:
            if hasattr(input_ds, "id"):
                input_id = input_ds.id
                input_id_lower = input_id.lower()

                # Check if this is a managed folder
                if input_id_lower in managed_folder_ids:
                    # Managed folder - create managed-folder-to-recipe edge
                    input_node_id = get_node_id(input_id_lower, "managed_folder")
                    mermaid_lines.append(f"    {input_node_id} --> {recipe_node_id}")
                    continue

                # Skip if orphaned and not including
                if input_id in orphaned_ids and not config.include_orphaned:
                    continue

                # If intermediate, create recipe-to-recipe edge
                if input_id in intermediate_ids:
                    if input_id in intermediate_to_producer:
                        producer_recipe_id = intermediate_to_producer[input_id]
                        producer_node_id = get_node_id(producer_recipe_id, "recipe")
                        mermaid_lines.append(
                            f"    {producer_node_id} --> {recipe_node_id}"
                        )
                else:
                    # Root/leaf dataset - create dataset-to-recipe edge
                    input_node_id = get_node_id(input_id, "dataset")
                    mermaid_lines.append(f"    {input_node_id} --> {recipe_node_id}")

        # Edges from recipe to output datasets
        # Only show connections to root/leaf datasets (not intermediate)
        for output_ds in recipe.outputs:
            if hasattr(output_ds, "id"):
                # Skip if intermediate (recipe-to-recipe edges handled above)
                if output_ds.id in intermediate_ids:
                    continue
                # Skip if orphaned and not including
                if output_ds.id in orphaned_ids and not config.include_orphaned:
                    continue
                output_node_id = get_node_id(output_ds.id, "dataset")
                mermaid_lines.append(f"    {recipe_node_id} --> {output_node_id}")

    mermaid_lines.append("```")
    return "\n".join(mermaid_lines)


def _render_template(template_data: dict) -> str:
    """Render the Jinja template with the provided data."""
    import yaml

    # Get the templates directory (in the src folder)
    templates_dir = Path(__file__).parent / "templates"

    # Set up Jinja environment
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )

    # Add YAML filter
    def to_yaml(value):
        """Convert a Python object to YAML format."""
        return yaml.dump(
            value, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    env.filters["to_yaml"] = to_yaml

    # Load and render template
    template = env.get_template("flow_visualization.md.j2")
    return template.render(**template_data)
