"""
Pipeline generator for creating Kubeflow pipelines from Dataiku scenarios.

This module analyzes Dataiku scenarios and generates complete Kubeflow pipeline
definitions with properly wired component dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle
    from .scenarios import Scenario


@dataclass
class PipelineComponent:
    """Represents a component in the generated pipeline."""

    id: str
    type: str  # "source", "recipe", "sub_pipeline", "operational"
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    recipe_type: str | None = None  # python, sql, etc.
    description: str = ""
    step_type: str | None = (
        None  # For operational steps: custom_python, pull_git_refs, etc.
    )
    script: str | None = None  # For custom_python operational steps
    managed_folder_inputs: list[str] = field(
        default_factory=list
    )  # IDs of managed folders used by this recipe


@dataclass
class PipelineStructure:
    """Complete structure of a pipeline to be generated."""

    scenario_id: str
    scenario_name: str
    scenario_description: str
    steps: list[PipelineStep] = field(default_factory=list)
    all_components: list[PipelineComponent] = field(default_factory=list)
    root_datasets: list[str] = field(default_factory=list)
    target_datasets: list[str] = field(default_factory=list)
    is_sub_pipeline: bool = False  # Is this called by another pipeline?
    calls_sub_pipelines: bool = False  # Does this call other pipelines?
    sub_pipeline_ids: list[str] = field(
        default_factory=list
    )  # List of called pipeline scenario IDs


@dataclass
class PipelineStep:
    """A step in the pipeline (from scenario steps)."""

    step_id: str
    description: str
    target_datasets: list[str] = field(default_factory=list)
    components: list[PipelineComponent] = field(default_factory=list)
    step_type: str | None = (
        None  # Type of step: run_scenario, build, custom_python, etc.
    )
    sub_pipeline_id: str | None = (
        None  # For run_scenario steps, the scenario ID to call
    )
    requires_after: bool = True  # Whether this step needs .after() previous step


def trace_dataset_chain(
    bundle: DataikuBundle, dataset_id: str
) -> list[PipelineComponent]:
    """
    Trace the complete recipe chain needed to build a dataset.

    Returns a list of components in dependency order (roots first).

    Args:
        bundle: DataikuBundle instance
        dataset_id: Target dataset to trace

    Returns:
        List of PipelineComponent objects in execution order
    """
    from .flow_analysis import get_root_datasets
    from .recipes import get_recipe_list

    all_recipes = get_recipe_list(bundle)
    root_dataset_ids = {ds.id for ds in get_root_datasets(bundle)}

    components = []
    visited_datasets = set()
    visited_recipes = set()

    def trace_recursive(ds_id: str):
        """Recursively trace dependencies."""
        if ds_id in visited_datasets:
            return
        visited_datasets.add(ds_id)

        # If this is a root dataset, create source component
        if ds_id in root_dataset_ids:
            if ds_id not in [c.id for c in components if c.type == "source"]:
                # dataset variable was unused - removed
                components.append(
                    PipelineComponent(
                        id=f"read_{ds_id}",
                        type="source",
                        inputs=[],
                        outputs=[ds_id],
                        description=f"Read {ds_id} from Snowflake",
                    )
                )
            return

        # Find recipe that produces this dataset
        for recipe in all_recipes:
            recipe_outputs = [o.id for o in recipe.outputs if hasattr(o, "id")]
            if ds_id in recipe_outputs:
                # First, trace dependencies for recipe inputs
                recipe_inputs = [i.id for i in recipe.inputs if hasattr(i, "id")]
                for input_ds in recipe_inputs:
                    trace_recursive(input_ds)

                # Now add this recipe component
                if recipe.id not in visited_recipes:
                    visited_recipes.add(recipe.id)

                    # Detect managed folders used by this recipe
                    from .utils import extract_managed_folder_ids_from_code

                    managed_folders = []
                    if recipe.code:
                        managed_folders = extract_managed_folder_ids_from_code(
                            recipe.code
                        )

                    components.append(
                        PipelineComponent(
                            id=recipe.id,
                            type="recipe",
                            inputs=recipe_inputs,
                            outputs=recipe_outputs,
                            recipe_type=recipe.type,
                            description=f"Recipe: {recipe.id}",
                            managed_folder_inputs=managed_folders,
                        )
                    )
                break

    trace_recursive(dataset_id)
    return components


def identify_scenarios_to_generate(
    bundle: DataikuBundle, root_scenario_id: str
) -> set[str]:
    """
    Recursively identify all scenarios that need pipeline files.

    When a scenario calls other scenarios via run_scenario steps, we need to generate
    pipeline files for all of them. This function traverses the scenario tree and
    returns the complete set of scenario IDs that need pipelines.

    Args:
        bundle: DataikuBundle instance
        root_scenario_id: Starting scenario ID to traverse from

    Returns:
        Set of scenario IDs that need their own pipeline files
    """
    from .scenarios import get_scenario
    from .utils import StepTypes

    scenarios_to_generate = set()

    def traverse(scenario_id: str):
        if scenario_id in scenarios_to_generate:
            return  # Already processed

        scenarios_to_generate.add(scenario_id)

        try:
            scenario = get_scenario(scenario_id, bundle)

            # Find all run_scenario steps
            for step in scenario.steps:
                if step.type == StepTypes.run_scenario and step.scenario:
                    traverse(step.scenario.id)
        except FileNotFoundError:
            # Scenario file doesn't exist, skip it
            pass

    traverse(root_scenario_id)
    return scenarios_to_generate


def generate_pipeline_structure_nested(
    bundle: DataikuBundle, scenario: Scenario, is_sub_pipeline: bool = False
) -> PipelineStructure:
    """
    Generate pipeline structure with nested pipeline awareness.

    This version does NOT recursively expand run_scenario steps. Instead, it marks
    them as sub-pipeline calls and tracks dependencies. This preserves the Dataiku
    scenario structure and enables true nested pipelines in KFP.

    Key differences from generate_pipeline_structure():
    1. run_scenario steps become sub-pipeline calls (not flattened)
    2. Tracks which scenarios this pipeline depends on
    3. Preserves sequential step boundaries

    Args:
        bundle: DataikuBundle instance
        scenario: Scenario to convert to pipeline
        is_sub_pipeline: Whether this pipeline is called by another pipeline

    Returns:
        PipelineStructure with nested pipeline information
    """
    from .generate_pipelines import create_vertex_pipeline_name

    structure = PipelineStructure(
        scenario_id=scenario.id,
        scenario_name=create_vertex_pipeline_name(scenario.id),
        scenario_description=scenario.desc or f"Pipeline for {scenario.id}",
        is_sub_pipeline=is_sub_pipeline,
    )

    all_components_map = {}  # id -> component

    # Process each step in the scenario
    for step_idx, step in enumerate(scenario.steps):
        pipeline_step = PipelineStep(
            step_id=f"step_{step_idx}",
            description=step.desc or f"Step {step_idx + 1}",
            step_type=step.type.value,
        )

        # Check if this is a run_scenario step
        if hasattr(step, "scenario") and step.scenario is not None:
            # This step calls another scenario -> sub-pipeline
            pipeline_step.sub_pipeline_id = step.scenario.id
            structure.calls_sub_pipelines = True
            if step.scenario.id not in structure.sub_pipeline_ids:
                structure.sub_pipeline_ids.append(step.scenario.id)

        # Check if this is a dataset build step (has items)
        elif hasattr(step, "items") and step.items:
            # Process dataset build steps
            for item in step.items:
                if hasattr(item, "id"):
                    dataset_id = item.id
                    pipeline_step.target_datasets.append(dataset_id)
                    structure.target_datasets.append(dataset_id)

                    # Trace the complete chain for this dataset
                    components = trace_dataset_chain(bundle, dataset_id)

                    # Add to global component map
                    for comp in components:
                        if comp.id not in all_components_map:
                            all_components_map[comp.id] = comp
                            pipeline_step.components.append(comp)

                            # Track root datasets
                            if comp.type == "source":
                                output_ds = comp.outputs[0] if comp.outputs else None
                                if (
                                    output_ds
                                    and output_ds not in structure.root_datasets
                                ):
                                    structure.root_datasets.append(output_ds)
        else:
            # This is an operational step (custom_python, pull_git_refs, etc.)
            # Create an operational component for it
            comp_id = f"{scenario.id}_{step.type.value}_{step_idx + 1}"
            comp = PipelineComponent(
                id=comp_id,
                type="operational",
                inputs=[],
                outputs=[],
                step_type=step.type.value,
                description=step.desc or f"{step.type.value} step",
                script=step.script if hasattr(step, "script") else None,
            )

            if comp_id not in all_components_map:
                all_components_map[comp_id] = comp
                pipeline_step.components.append(comp)

        structure.steps.append(pipeline_step)

    # All unique components across all steps
    structure.all_components = list(all_components_map.values())

    return structure


def generate_pipeline_structure(
    bundle: DataikuBundle, scenario: Scenario
) -> PipelineStructure:
    """
    Analyze a scenario and generate the complete pipeline structure (FLATTENED).

    This is the original implementation that recursively expands run_scenario steps
    into a single flat pipeline. Use generate_pipeline_structure_nested() for the
    nested pipeline approach that preserves Dataiku's scenario hierarchy.

    Args:
        bundle: DataikuBundle instance
        scenario: Scenario to convert to pipeline

    Returns:
        PipelineStructure with all components and wiring information
    """
    from .generate_pipelines import create_vertex_pipeline_name

    structure = PipelineStructure(
        scenario_id=scenario.id,
        scenario_name=create_vertex_pipeline_name(scenario.id),
        scenario_description=scenario.desc or f"Pipeline for {scenario.id}",
    )

    all_components_map = {}  # id -> component

    # Process each step in the scenario
    for step_idx, step in enumerate(scenario.steps):
        pipeline_step = PipelineStep(
            step_id=f"step_{step_idx}",
            description=step.desc or f"Step {step_idx + 1}",
            target_datasets=[],
        )

        # Check if this is a run_scenario step
        if hasattr(step, "scenario") and step.scenario is not None:
            # Recursively process the sub-scenario's steps
            sub_scenario = step.scenario

            # Generate pipeline structure for the sub-scenario
            sub_structure = generate_pipeline_structure(bundle, sub_scenario)

            # Merge components from sub-scenario into the parent scenario
            for comp in sub_structure.all_components:
                if comp.id not in all_components_map:
                    all_components_map[comp.id] = comp
                    pipeline_step.components.append(comp)

                    # Track root datasets
                    if comp.type == "source":
                        output_ds = comp.outputs[0] if comp.outputs else None
                        if output_ds and output_ds not in structure.root_datasets:
                            structure.root_datasets.append(output_ds)

            # Merge target datasets
            for dataset_id in sub_structure.target_datasets:
                if dataset_id not in structure.target_datasets:
                    pipeline_step.target_datasets.append(dataset_id)
                    structure.target_datasets.append(dataset_id)

        # Check if this is a dataset build step (has items)
        elif hasattr(step, "items") and step.items:
            # Process dataset build steps
            for item in step.items:
                if hasattr(item, "id"):
                    dataset_id = item.id
                    pipeline_step.target_datasets.append(dataset_id)
                    structure.target_datasets.append(dataset_id)

                    # Trace the complete chain for this dataset
                    components = trace_dataset_chain(bundle, dataset_id)

                    # Add to global component map
                    for comp in components:
                        if comp.id not in all_components_map:
                            all_components_map[comp.id] = comp
                            pipeline_step.components.append(comp)

                            # Track root datasets
                            if comp.type == "source":
                                output_ds = comp.outputs[0] if comp.outputs else None
                                if (
                                    output_ds
                                    and output_ds not in structure.root_datasets
                                ):
                                    structure.root_datasets.append(output_ds)
        else:
            # This is an operational step (custom_python, pull_git_refs, etc.)
            # Create an operational component for it
            # Use scenario_id to make component IDs unique across scenarios
            comp_id = f"{scenario.id}_{step.type.value}_{step_idx + 1}"
            comp = PipelineComponent(
                id=comp_id,
                type="operational",
                inputs=[],
                outputs=[],
                step_type=step.type.value,
                description=step.desc or f"{step.type.value} step",
                script=step.script if hasattr(step, "script") else None,
            )

            if comp_id not in all_components_map:
                all_components_map[comp_id] = comp
                pipeline_step.components.append(comp)

        structure.steps.append(pipeline_step)

    # All unique components across all steps
    structure.all_components = list(all_components_map.values())

    return structure


def get_component_columns(bundle: DataikuBundle, dataset_id: str) -> list[str]:
    """
    Get column names for a dataset.

    Args:
        bundle: DataikuBundle instance
        dataset_id: Dataset ID

    Returns:
        List of column names
    """
    from .datasets import get_datasets_list

    all_datasets = {ds.id: ds for ds in get_datasets_list(bundle)}
    dataset = all_datasets.get(dataset_id)

    if dataset and dataset.columns:
        return [col.name for col in dataset.columns]
    return []


def get_recipe_code(bundle: DataikuBundle, recipe_id: str) -> str:
    """
    Get the original Python code for a recipe.

    Args:
        bundle: DataikuBundle instance
        recipe_id: Recipe ID

    Returns:
        Recipe code as string, or empty string if not found
    """
    recipe_file = bundle.path / "project_config" / "recipes" / f"{recipe_id}.py"
    if recipe_file.exists():
        return recipe_file.read_text()
    return "# Recipe code not found"


def get_recipe_sql(bundle: DataikuBundle, recipe_id: str) -> str:
    """
    Get the original SQL code for a recipe.

    Args:
        bundle: DataikuBundle instance
        recipe_id: Recipe ID

    Returns:
        SQL code as string, or empty string if not found
    """
    recipe_file = bundle.path / "project_config" / "recipes" / f"{recipe_id}.sql"
    if recipe_file.exists():
        return recipe_file.read_text()
    return ""


def get_recipe_original_code(
    bundle: DataikuBundle, recipe_id: str, recipe_type: str
) -> str:
    """
    Get the original recipe code (Python, SQL, or other).

    Args:
        bundle: DataikuBundle instance
        recipe_id: Recipe ID
        recipe_type: Recipe type (python, sql, etc.)

    Returns:
        Original code as string
    """
    if recipe_type == "python":
        return get_recipe_code(bundle, recipe_id)
    elif recipe_type == "sql":
        return get_recipe_sql(bundle, recipe_id)
    else:
        # Try to find any recipe file
        recipe_dir = bundle.path / "project_config" / "recipes"
        for ext in [".py", ".sql", ".json"]:
            recipe_file = recipe_dir / f"{recipe_id}{ext}"
            if recipe_file.exists() and ext != ".json":
                return recipe_file.read_text()
        return f"# {recipe_type} recipe - code not found"


def generate_unified_pipeline_structure(
    bundle: DataikuBundle,
    pipeline_name: str = "unified_pipeline",
    pipeline_description: str = "Complete Dataiku flow - all leaf datasets",
) -> PipelineStructure:
    """
    Generate a unified pipeline structure that computes ALL leaf datasets.

    This ignores scenario structure and instead uses flow analysis to build
    a complete DAG from root datasets to all leaf datasets.

    Strategy:
    1. Get all leaf datasets (final outputs)
    2. Get all root datasets (sources)
    3. Trace dependencies for all leaves to build complete component set
    4. Generate one unified pipeline with proper dependency ordering

    Args:
        bundle: DataikuBundle instance
        pipeline_name: Name for the generated pipeline
        pipeline_description: Description for the pipeline

    Returns:
        PipelineStructure with all components needed to compute all leaf datasets
    """
    from .flow_analysis import get_leaf_datasets, get_root_datasets

    # Get all leaf and root datasets
    leaf_datasets = get_leaf_datasets(bundle)
    root_datasets = get_root_datasets(bundle)

    structure = PipelineStructure(
        scenario_id="unified_flow",
        scenario_name=pipeline_name,
        scenario_description=pipeline_description,
        root_datasets=[ds.id for ds in root_datasets],
        target_datasets=[ds.id for ds in leaf_datasets],
    )

    # Track all unique components
    all_components_map = {}  # component_id -> component

    # Create a single step that builds all leaf datasets
    unified_step = PipelineStep(
        step_id="compute_all_leaves",
        description="Compute all leaf datasets",
        target_datasets=[ds.id for ds in leaf_datasets],
    )

    # Trace dependencies for ALL leaf datasets
    for leaf_ds in leaf_datasets:
        components = trace_dataset_chain(bundle, leaf_ds.id)

        for comp in components:
            if comp.id not in all_components_map:
                all_components_map[comp.id] = comp
                unified_step.components.append(comp)

    structure.steps.append(unified_step)
    structure.all_components = list(all_components_map.values())

    return structure
