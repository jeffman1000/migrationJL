"""
Generate Kubeflow pipelines from Dataiku scenarios.

This script:
1. Loads the Dataiku bundle
2. Gets all scenarios
3. For each scenario:
   - Analyzes dependencies
   - Generates complete pipeline structure
   - Renders pipeline template
   - Writes to vertex_pipelines/pipelines/{scenario_id}.py

Usage:
    uv run python generate_pipelines.py <bundle_path>

Example:
    uv run python generate_pipelines.py dss-bundle-SDKEXPLORATION-2025-10-15
"""

import re
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.dataiku_bundle import DataikuBundle
from src.pipeline_generator import (
    PipelineComponent,
    generate_pipeline_structure,
    get_component_columns,
    get_recipe_original_code,
)
from src.recipe_analyzer import RecipeAnalyzer
from src.scenarios import get_scenario_list


def sanitize_dataset_name(dataset_name: str) -> str:
    """
    Convert dataset name to valid Python identifier with dataset_ prefix.

    Examples:
        "demand" -> "dataset_demand"
        "predict totalrowsfordate (regression)" -> "dataset_predict_totalrowsfordate_regression"
        "some-dataset" -> "dataset_some_dataset"
        "123dataset" -> "dataset_123dataset"
        "dataset_criteria" -> "dataset_criteria" (already has prefix)
        "0_mydata" -> "dataset_0_mydata"

    Args:
        dataset_name: Original dataset name from Dataiku

    Returns:
        Valid Python identifier with dataset_ prefix
    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_name)

    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Handle empty result
    if not sanitized:
        return "dataset"

    # If already starts with "dataset_", return as-is
    if sanitized.startswith("dataset_"):
        return sanitized

    # Otherwise, add dataset_ prefix (whether it starts with a digit or not)
    return "dataset_" + sanitized


def create_pipeline_function_name(scenario_id: str) -> str:
    """
    Create a valid Python function name from a scenario ID.

    Always prepends 'scenario_' to ensure consistency and avoid Python keyword conflicts.

    Examples:
        "0_VARIABLES" -> "scenario_0_variables"
        "1_build_data" -> "scenario_1_build_data"
        "BUILD_FEATURES" -> "scenario_build_features"

    Args:
        scenario_id: Original scenario ID from Dataiku

    Returns:
        Valid Python function name with scenario_ prefix
    """
    # Convert to lowercase and replace hyphens with underscores
    sanitized = scenario_id.lower().replace("-", "_")

    # Replace any other special characters with underscores
    sanitized = re.sub(r"[^a-z0-9_]", "_", sanitized)

    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Handle empty result
    if not sanitized:
        return "scenario_pipeline"

    # If already starts with "scenario_", return as-is
    if sanitized.startswith("scenario_"):
        return sanitized

    # Otherwise, always prepend 'scenario_'
    return "scenario_" + sanitized


def create_vertex_pipeline_name(scenario_id: str) -> str:
    """
    Create a valid Vertex AI pipeline name from a scenario ID.

    Vertex AI pipeline names must follow regex: [a-z][-a-z0-9]{0,127}
    - Start with a lowercase letter
    - Followed by lowercase letters, digits, or hyphens
    - Be 1-128 characters long

    Always prepends 'scenario-' to ensure consistency.

    Examples:
        "0_VARIABLES" -> "scenario-0-variables"
        "1_build_data" -> "scenario-1-build-data"
        "BUILD_FEATURES" -> "scenario-build-features"

    Args:
        scenario_id: Original scenario ID from Dataiku

    Returns:
        Valid Vertex AI pipeline name
    """
    # Convert to lowercase
    sanitized = scenario_id.lower()

    # Replace all underscores with hyphens
    sanitized = sanitized.replace("_", "-")

    # Replace other special characters with hyphens
    sanitized = re.sub(r"[^a-z0-9_-]", "-", sanitized)

    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Strip leading/trailing hyphens
    sanitized = sanitized.strip("-")

    # Handle empty result
    if not sanitized:
        return "scenario-pipeline"

    # If already starts with "scenario-", return as-is
    if sanitized.startswith("scenario-"):
        return sanitized

    # Otherwise, always prepend 'scenario-'
    sanitized = "scenario-" + sanitized

    # Truncate to 128 characters if needed
    if len(sanitized) > 128:
        sanitized = sanitized[:128].rstrip("-")

    return sanitized


def extract_scenario_scripts(bundle: DataikuBundle, output_dir: Path) -> int:
    """
    Extract all custom Python and SQL scripts from scenarios and write them to files.

    Walks through all scenarios and their steps, extracting:
    - Custom Python scripts from custom_python steps
    - SQL scripts from exec_sql steps
    - Custom Python scripts from entire custom_python scenarios

    Files are named: scenario_{scenario_id}_step_{step_number}.{py|sql}

    Args:
        bundle: DataikuBundle instance to extract scripts from
        output_dir: Directory to write script files to (e.g., vertex_pipelines/components/operational)

    Returns:
        Number of script files written
    """
    from src.scenarios import get_scenario_list
    from src.utils import StepTypes

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all scenarios
    scenarios = get_scenario_list(bundle)

    script_count = 0

    for scenario in scenarios:
        # Check if entire scenario is custom_python type
        scenario_data = scenario._load_scenario_file()
        if scenario_data and scenario_data.get("type") == "custom_python":
            # This is a custom_python scenario - extract the script
            script = scenario._load_custom_python_script()
            if script:
                # Write as scenario_{id}_step_1.py (treat whole scenario as step 1)
                filename = f"scenario_{scenario.id}_step_1.py"
                output_file = output_dir / filename
                output_file.write_text(script)
                script_count += 1
                print(f"  ✅ Extracted: {filename}")
            continue

        # Process step-based scenarios
        for step in scenario.steps:
            if step.type == StepTypes.custom_python and step.script:
                # Custom Python step
                filename = f"scenario_{scenario.id}_step_{step.id}.py"
                output_file = output_dir / filename
                output_file.write_text(step.script)
                script_count += 1
                print(f"  ✅ Extracted: {filename}")

            elif step.type == StepTypes.exec_sql:
                # SQL step - need to extract SQL from step data
                # Load the scenario JSON to get the SQL
                if scenario_data is None:
                    scenario_data = scenario._load_scenario_file()

                if scenario_data:
                    steps_data = scenario_data.get("params", {}).get("steps", [])
                    # Find the corresponding step data (step.id is 1-indexed)
                    if 0 < step.id <= len(steps_data):
                        step_data = steps_data[step.id - 1]
                        sql_script = step_data.get("params", {}).get("sql")
                        if sql_script:
                            filename = f"scenario_{scenario.id}_step_{step.id}.sql"
                            output_file = output_dir / filename
                            output_file.write_text(sql_script)
                            script_count += 1
                            print(f"  ✅ Extracted: {filename}")

    # Create __init__.py
    init_file = output_dir / "__init__.py"
    init_file.write_text(
        '"""Operational scripts extracted from Dataiku scenarios."""\n'
    )

    return script_count


def sanitize_component_id(component_id: str) -> str:
    """
    Create a valid Python identifier from a component ID.

    Always prepends 'component_' unless the ID already starts with 'compute' or 'component'.
    This is used for module names and function names.

    Examples:
        "0_variables_custom_python_1" -> "component_0_variables_custom_python_1"
        "prepare_data" -> "component_prepare_data"
        "compute_features" -> "compute_features"
        "component_features" -> "component_features"

    Args:
        component_id: Original component ID

    Returns:
        Valid Python identifier for use as module/function name
    """
    # Replace hyphens with underscores
    sanitized = component_id.replace("-", "_")

    # Replace any other special characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized)

    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Handle empty result
    if not sanitized:
        return "component"

    # If already starts with "compute" or "component", return as-is
    if sanitized.startswith("compute") or sanitized.startswith("component"):
        return sanitized

    # Otherwise, always prepend 'component_'
    return "component_" + sanitized


def render_pipeline(structure, template_env: Environment) -> str:
    """Render pipeline Python code from structure."""
    template = template_env.get_template("pipeline.py.j2")

    # Create valid Python function name from scenario ID
    function_name = create_pipeline_function_name(structure.scenario_id)

    # Collect all managed folder IDs used by components
    managed_folder_ids = set()
    for component in structure.all_components:
        if (
            hasattr(component, "managed_folder_inputs")
            and component.managed_folder_inputs
        ):
            managed_folder_ids.update(component.managed_folder_inputs)

    return template.render(
        scenario_id=structure.scenario_id,
        scenario_name=structure.scenario_name,
        scenario_description=structure.scenario_description,
        function_name=function_name,
        steps=structure.steps,
        all_components=structure.all_components,
        root_datasets=structure.root_datasets,
        target_datasets=structure.target_datasets,
        managed_folder_ids=list(managed_folder_ids),
    )


def render_component(
    component,
    bundle: DataikuBundle,
    template_env: Environment,
    analyzer: RecipeAnalyzer,
    root_datasets: set[str] | None = None,
    dataset_to_producer: dict[str, str] | None = None,
    dataset_to_consumers: dict[str, list[str]] | None = None,
) -> str:
    """Render a component file (source, recipe, or operational)."""
    if component.type == "source":
        template = template_env.get_template("snowflake_source_component.py.j2")
        columns = get_component_columns(bundle, component.outputs[0])
        dataset_id = component.outputs[0]
        # For source components, we want "read_<dataset>" not "component_read_<dataset>"
        # So we sanitize the dataset name directly, not the full component id
        sanitized_dataset_id = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_id)
        sanitized_dataset_id = re.sub(r"_+", "_", sanitized_dataset_id)
        sanitized_dataset_id = sanitized_dataset_id.strip("_")
        return template.render(
            dataset_id=sanitized_dataset_id,
            dataset_param=sanitize_dataset_name(dataset_id),
            columns=columns,
        )
    elif component.type == "recipe":
        template = template_env.get_template("recipe_component.py.j2")
        original_code = get_recipe_original_code(
            bundle, component.id, component.recipe_type or "python"
        )

        # Filter out root datasets from inputs (they won't be KFP Input parameters)
        root_datasets = root_datasets or set()
        non_root_inputs = [ds for ds in component.inputs if ds not in root_datasets]

        # Create mapping of original dataset names to sanitized parameter names
        input_params = {ds: sanitize_dataset_name(ds) for ds in non_root_inputs}
        output_params = {ds: sanitize_dataset_name(ds) for ds in component.outputs}

        # Build lineage information for docstring
        dataset_to_producer = dataset_to_producer or {}
        dataset_to_consumers = dataset_to_consumers or {}

        input_lineage = {}
        for ds in component.inputs:
            if ds in root_datasets:
                input_lineage[ds] = "[root dataset]"
            elif ds in dataset_to_producer:
                input_lineage[ds] = f"[from {dataset_to_producer[ds]}]"
            else:
                input_lineage[ds] = ""

        output_lineage = {}
        for ds in component.outputs:
            consumers = dataset_to_consumers.get(ds, [])
            if consumers:
                output_lineage[ds] = f"[used by {', '.join(consumers)}]"
            else:
                output_lineage[ds] = ""

        # Analyze the recipe to get comprehensive metadata
        recipe_metadata = None
        try:
            recipe_info = analyzer.analyze_recipe(component.id)
            recipe_metadata = analyzer.format_as_comment(recipe_info, max_width=96)
        except Exception as e:
            print(f"  ⚠️  Warning: Could not analyze recipe {component.id}: {e}")

        return template.render(
            recipe_id=sanitize_component_id(component.id),
            recipe_type=component.recipe_type or "python",
            input_datasets=component.inputs,
            non_root_inputs=non_root_inputs,
            output_datasets=component.outputs,
            input_params=input_params,
            output_params=output_params,
            input_lineage=input_lineage,
            output_lineage=output_lineage,
            recipe_description=component.description,
            original_code=original_code,
            recipe_metadata=recipe_metadata,
            managed_folder_inputs=component.managed_folder_inputs,
        )
    elif component.type == "operational" and component.step_type == "custom_python":
        # Generate a separate component file for custom_python operational steps
        template = template_env.get_template("custom_python_component.py.j2")
        return template.render(
            component_name=sanitize_component_id(component.id),
            scenario_id=component.id,
            description=component.description,
            script=component.script or "# No script found",
        )
    else:
        return f"# TODO: Component type {component.type} not implemented"


def generate_all_pipelines(
    bundle_path: Path,
    output_dir: Path,
    limit: int | None = None,
    nested_mode: bool = False,
    unified_mode: bool = True,
):
    """
    Generate pipeline files for all scenarios in a bundle.

    Args:
        bundle_path: Path to Dataiku bundle
        output_dir: Directory to write pipeline files to
        limit: Optional limit on number of scenarios to process (for testing)
        nested_mode: If True, generate nested pipelines (scenarios call sub-scenarios).
        unified_mode: If True (DEFAULT), generate ONE pipeline for the entire flow (all leaf datasets).
                      Ignores scenario structure. Takes precedence over nested_mode.
    """
    mode_name = "UNIFIED" if unified_mode else ("NESTED" if nested_mode else "FLAT")
    print("=" * 80)
    print(f"KUBEFLOW PIPELINE GENERATION - {mode_name} MODE")
    print("=" * 80)
    print()

    # Load bundle
    print(f"Loading bundle: {bundle_path}")
    bundle = DataikuBundle(bundle_path)
    print()

    # Create recipe analyzer
    print("Initializing recipe analyzer...")
    analyzer = RecipeAnalyzer(bundle)
    print()

    # Initialize counters for tracking generated files
    script_count = 0

    # UNIFIED MODE: Generate one pipeline for entire flow
    if unified_mode:
        print("🔄 UNIFIED MODE: Generating single pipeline for entire flow")
        print("-" * 80)
        from src.pipeline_generator import generate_unified_pipeline_structure
        from src.flow_analysis import get_leaf_datasets, get_root_datasets

        # Get flow statistics
        leaf_datasets = get_leaf_datasets(bundle)
        root_datasets = get_root_datasets(bundle)

        print("Flow analysis:")
        print(f"  - Root datasets (sources): {len(root_datasets)}")
        print(f"  - Leaf datasets (targets): {len(leaf_datasets)}")
        print()

        # Generate unified pipeline structure
        print("Generating unified pipeline structure...")
        structure = generate_unified_pipeline_structure(bundle)

        print(f"  - Total components: {len(structure.all_components)}")
        print(
            f"  - Recipe components: {len([c for c in structure.all_components if c.type == 'recipe'])}"
        )
        print(
            f"  - Source components: {len([c for c in structure.all_components if c.type == 'source'])}"
        )
        print()

        # Setup Jinja environment (same as scenario mode)
        script_dir = Path(__file__).parent
        templates_dir = script_dir / "templates"
        env = Environment(
            loader=FileSystemLoader(templates_dir),
            extensions=["jinja2.ext.do"],
        )
        env.filters["sanitize"] = sanitize_dataset_name
        env.filters["sanitize_component"] = sanitize_component_id

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Render and write unified pipeline
        print("Rendering unified pipeline...")
        pipeline_code = render_pipeline(structure, env)
        output_file = output_dir / "unified_flow.py"
        output_file.write_text(pipeline_code)
        print(f"✅ Generated: {output_file}")
        print()

        # Extract scenario scripts to operational folder
        print("=" * 80)
        print("EXTRACTING SCENARIO SCRIPTS")
        print("=" * 80)
        print()
        operational_scripts_dir = Path("vertex_pipelines/components/operational")
        script_count = extract_scenario_scripts(bundle, operational_scripts_dir)
        print()
        print(f"✅ Extracted {script_count} script files to: {operational_scripts_dir}")
        print()

        # Generate all component files (same as scenario mode - see below)
        # We'll skip returning early and let it fall through to component generation
        scenarios = []  # Empty list so we skip scenario loop
    else:
        # Get all scenarios
        scenarios = get_scenario_list(bundle)
        print(f"Found {len(scenarios)} scenarios")

    if limit:
        scenarios = scenarios[:limit]
        print(f"Processing first {limit} scenario(s) only")

    print()
    for scenario in scenarios:
        print(f"  - {scenario.id}: {scenario.desc}")
    print()

    # Setup Jinja environment
    # Use path relative to this script file, not the current working directory
    script_dir = Path(__file__).parent
    templates_dir = script_dir / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        extensions=["jinja2.ext.do"],  # Enable do extension for variable assignment
    )

    # Add custom filters
    env.filters["sanitize"] = sanitize_dataset_name
    env.filters["sanitize_component"] = sanitize_component_id

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    components_dir = Path("vertex_pipelines/components/recipes")
    components_dir.mkdir(parents=True, exist_ok=True)

    # Track all unique components across scenarios
    all_components_map = {}  # component_id -> component

    # In nested mode, we need to identify all scenarios to generate (including dependencies)
    if nested_mode:
        from src.pipeline_generator import (
            identify_scenarios_to_generate,
            generate_pipeline_structure_nested,
        )
        from src.scenarios import get_scenario

        print("Identifying all scenarios to generate (including dependencies)...")
        all_scenario_ids = set()
        for scenario in scenarios:
            scenario_ids = identify_scenarios_to_generate(bundle, scenario.id)
            all_scenario_ids.update(scenario_ids)

        print(f"Total scenarios to generate: {len(all_scenario_ids)}")
        print()

        # Load all scenarios
        scenarios_to_generate = []
        for scenario_id in sorted(all_scenario_ids):
            try:
                scenarios_to_generate.append(get_scenario(scenario_id, bundle))
            except FileNotFoundError:
                print(f"  ⚠️  Warning: Scenario {scenario_id} not found, skipping")

        scenarios = scenarios_to_generate
        print(f"Loaded {len(scenarios)} scenario(s)")
        for scenario in scenarios:
            print(f"  - {scenario.id}: {scenario.desc}")
        print()

    # Generate pipeline for each scenario
    print("Generating pipelines...")
    print("-" * 80)

    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario.id}")
        print(f"   Description: {scenario.desc}")
        print(f"   Steps: {len(scenario.steps)}")

        try:
            # Generate pipeline structure
            if nested_mode:
                from src.pipeline_generator import generate_pipeline_structure_nested

                structure = generate_pipeline_structure_nested(bundle, scenario)
            else:
                structure = generate_pipeline_structure(bundle, scenario)

            print(f"   - Root datasets: {len(structure.root_datasets)}")
            print(f"   - Components: {len(structure.all_components)}")
            print(f"   - Target datasets: {len(structure.target_datasets)}")
            if nested_mode and structure.calls_sub_pipelines:
                print(f"   - Sub-pipelines: {len(structure.sub_pipeline_ids)}")

            # Collect unique components
            for component in structure.all_components:
                if component.id not in all_components_map:
                    all_components_map[component.id] = component

            # Render pipeline template
            if nested_mode:
                template = env.get_template("nested_pipeline.py.j2")
                function_name = create_pipeline_function_name(scenario.id)

                # Create mapping of sub-pipeline IDs to their function names
                sub_pipeline_function_names = {
                    sp_id: create_pipeline_function_name(sp_id)
                    for sp_id in structure.sub_pipeline_ids
                }

                pipeline_code = template.render(
                    scenario_id=structure.scenario_id,
                    scenario_name=structure.scenario_name,
                    scenario_description=structure.scenario_description,
                    function_name=function_name,
                    steps=structure.steps,
                    all_components=structure.all_components,
                    root_datasets=structure.root_datasets,
                    target_datasets=structure.target_datasets,
                    calls_sub_pipelines=structure.calls_sub_pipelines,
                    sub_pipeline_ids=structure.sub_pipeline_ids,
                    sub_pipeline_function_names=sub_pipeline_function_names,
                )
            else:
                pipeline_code = render_pipeline(structure, env)

            # Write to file with sanitized filename
            sanitized_filename = create_pipeline_function_name(scenario.id)
            output_file = output_dir / f"{sanitized_filename}.py"
            output_file.write_text(pipeline_code)

            print(f"   ✅ Generated: {output_file}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    # Generate component files for ALL recipes in the bundle
    # Not just the ones referenced by scenarios
    print()
    print("=" * 80)
    print("GENERATING RECIPE COMPONENTS")
    print("=" * 80)
    print()

    # Get all recipes from the bundle (not just those in scenarios)
    from src.flow_analysis import get_root_datasets
    from src.recipes import get_recipe_list

    all_recipes = get_recipe_list(bundle)
    root_dataset_ids = {ds.id for ds in get_root_datasets(bundle)}

    print(f"Found {len(all_recipes)} recipes in bundle")
    print(f"Found {len(all_components_map)} components referenced in scenarios")
    print("Generating component files for all recipes...")
    print()

    # Build lineage maps for all datasets
    dataset_to_producer: dict[str, str] = {}  # dataset_id -> recipe_id that produces it
    dataset_to_consumers: dict[
        str, list[str]
    ] = {}  # dataset_id -> [recipe_ids that consume it]

    for recipe in all_recipes:
        # Map outputs to their producer recipe
        for output in recipe.outputs:
            if hasattr(output, "id"):
                dataset_to_producer[output.id] = recipe.id

        # Map inputs to their consumer recipes
        for input_ds in recipe.inputs:
            if hasattr(input_ds, "id"):
                if input_ds.id not in dataset_to_consumers:
                    dataset_to_consumers[input_ds.id] = []
                dataset_to_consumers[input_ds.id].append(recipe.id)

    # Generate components for all recipes (whether or not they're in scenarios)
    component_count = 0
    for recipe in all_recipes:
        recipe_inputs = [i.id for i in recipe.inputs if hasattr(i, "id")]
        recipe_outputs = [o.id for o in recipe.outputs if hasattr(o, "id")]

        # Detect managed folders used by this recipe
        from src.utils import extract_managed_folder_ids_from_code

        managed_folders = []
        if recipe.code:
            managed_folders = extract_managed_folder_ids_from_code(recipe.code)

        component = PipelineComponent(
            id=recipe.id,
            type="recipe",
            inputs=recipe_inputs,
            outputs=recipe_outputs,
            recipe_type=recipe.type,
            description=f"Recipe: {recipe.id}",
            managed_folder_inputs=managed_folders,
        )

        try:
            component_code = render_component(
                component,
                bundle,
                env,
                analyzer,
                root_datasets=root_dataset_ids,
                dataset_to_producer=dataset_to_producer,
                dataset_to_consumers=dataset_to_consumers,
            )
            sanitized_id = sanitize_component_id(recipe.id)
            component_file = components_dir / f"{sanitized_id}.py"
            component_file.write_text(component_code)
            component_count += 1
            if component_count % 10 == 0:
                print(f"  Generated {component_count}/{len(all_recipes)} components...")
        except Exception as e:
            print(f"  ❌ Error generating {recipe.id}: {e}")

    # Also generate source components for root datasets
    source_count = 0
    for root_ds_id in root_dataset_ids:
        source_component = PipelineComponent(
            id=f"read_{root_ds_id}",
            type="source",
            inputs=[],
            outputs=[root_ds_id],
            description=f"Read {root_ds_id} from Snowflake",
        )

        try:
            component_code = render_component(source_component, bundle, env, analyzer)
            sanitized_id = sanitize_component_id(f"read_{root_ds_id}")
            component_file = components_dir / f"{sanitized_id}.py"
            component_file.write_text(component_code)
            source_count += 1
        except Exception as e:
            print(f"  ❌ Error generating read_{root_ds_id}: {e}")

    print(f"  ✅ Generated {component_count} recipe component files")
    print(f"  ✅ Generated {source_count} source component files")

    # Create __init__.py for recipe components
    init_file = components_dir / "__init__.py"
    init_file.write_text('"""Recipe components generated from Dataiku bundle."""\n')

    print()
    print("=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"📁 Generated {len(scenarios)} pipeline files in: {output_dir}")
    print(f"📁 Generated {component_count} recipe components in: {components_dir}")
    print(f"📁 Generated {source_count} source components in: {components_dir}")
    if script_count > 0:
        operational_scripts_dir = Path("vertex_pipelines/components/operational")
        print(
            f"📁 Extracted {script_count} operational scripts in: {operational_scripts_dir}"
        )
    print()
    print("Next steps:")
    print("  1. Review generated pipeline files")
    print(
        "  2. Review generated component files (placeholders + original code as comments)"
    )
    print("  3. Add actual transformation logic to components")
    print("  4. Test individual components")
    print("  5. Test pipelines end-to-end")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_pipelines.py <bundle_path> [OPTIONS]")
        print()
        print("Options:")
        print("  --limit N       Process only first N scenarios")
        print("  --all           Process all scenarios")
        print(
            "  --nested        Generate nested pipelines (scenarios call sub-scenarios)"
        )
        print("  --flat          Generate flat pipelines (recursive expansion)")
        print("  --unified       Generate ONE pipeline for entire flow (DEFAULT)")
        print()
        print("Examples:")
        print("  python generate_pipelines.py dss-bundle-SDKEXPLORATION-2025-10-15")
        print(
            "  python generate_pipelines.py dss-bundle-SDKEXPLORATION-2025-10-15 --limit 1"
        )
        print(
            "  python generate_pipelines.py dss-bundle-SDKEXPLORATION-2025-10-15 --nested"
        )
        print(
            "  python generate_pipelines.py dss-bundle-SDKEXPLORATION-2025-10-15 --flat"
        )
        sys.exit(1)

    bundle_path = Path(sys.argv[1])
    output_dir = Path("vertex_pipelines/pipelines")

    # Parse options
    limit = None  # Default: process all scenarios
    nested_mode = "--nested" in sys.argv
    flat_mode = "--flat" in sys.argv

    # Unified is the default unless --nested or --flat is specified
    unified_mode = not (nested_mode or flat_mode)

    if "--limit" in sys.argv:
        limit_idx = sys.argv.index("--limit")
        if limit_idx + 1 < len(sys.argv):
            limit = int(sys.argv[limit_idx + 1])

    if not bundle_path.exists():
        print(f"Error: Bundle path does not exist: {bundle_path}")
        sys.exit(1)

    generate_all_pipelines(
        bundle_path,
        output_dir,
        limit=limit,
        nested_mode=nested_mode,
        unified_mode=unified_mode,
    )


if __name__ == "__main__":
    main()
