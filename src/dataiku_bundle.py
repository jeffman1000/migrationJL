"""
Main entry point for Dataiku Bundle SDK.

This module provides the primary DataikuBundle class and exposes all
public API functions for interacting with Dataiku bundles.
"""

from __future__ import annotations

import json
from pathlib import Path

from .datasets import (
    Column,
    Dataset,
    datatype,
    get_datasets_list,
)
from .managed_folders import (
    ManagedFolder,
    get_managed_folder,
    get_managed_folders_list,
)
from .recipes import (
    Predict,
    Recipe,
    get_recipe_list,
)
from .scenarios import (
    Scenario,
    SnowflakeDataset,
    Step,
    get_scenario,
    get_scenario_list,
    get_scenario_steps,
    get_scenario_yaml,
)

# Import all the public classes and functions from submodules
from .utils import (
    DatasetModes,
    DatasetTypes,
    RecipeTypes,
    StepsList,
    StepTypes,
)
from .variables import Variables

# Export all public API
__all__ = [
    # Main class
    "DataikuBundle",
    # Enums
    "StepTypes",
    "RecipeTypes",
    "DatasetTypes",
    "DatasetModes",
    # Classes
    "Step",
    "Scenario",
    "SnowflakeDataset",
    "Dataset",
    "Column",
    "Recipe",
    "Predict",
    "StepsList",
    "Variables",
    "ManagedFolder",
    # Data types
    "datatype",
    # Functions
    "get_scenario_list",
    "get_scenario",
    "get_scenario_steps",
    "get_scenario_yaml",
    "get_datasets_list",
    "get_recipe_list",
    "get_managed_folder",
    "get_managed_folders_list",
    "generate_component_recipes",
]


class DataikuBundle:
    """
    Represents a Dataiku bundle exported from DSS.

    A Dataiku bundle is an exported project archive containing:
    - Project configuration (scenarios, recipes, datasets)
    - Python libraries and custom code
    - Metadata and export manifests

    This class provides the main entry point for exploring bundle contents.
    """

    def __init__(self, path: str):
        """
        Initialize a DataikuBundle.

        Args:
            path: Path to the Dataiku bundle directory (typically named "dss-bundle-*")

        Raises:
            FileNotFoundError: If the bundle path or required directories don't exist
        """
        # Convert to Path object for easier file system operations
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Bundle path does not exist: {path}")

        # Validate that this looks like a valid Dataiku bundle
        self.scenarios_path = self.path / "project_config" / "scenarios"
        if not self.scenarios_path.exists():
            raise FileNotFoundError(
                f"Scenarios directory not found in bundle: {self.scenarios_path}"
            )

        # Lazy-loaded project key (extracted on first access)
        self._project_key: str | None = None

        # Lazy-loaded variables (extracted on first access)
        self._variables: Variables | None = None

    @property
    def project_key(self) -> str:
        """
        Get the project key for this bundle.

        The project key is extracted from the export manifest JSON file if available,
        or falls back to parsing the bundle directory name.

        Returns:
            The project key (e.g., "SDKEXPLORATION") or empty string if not found

        Example:
            >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
            >>> bundle.project_key
            'MYPROJECT'
        """
        # Use cached value if already loaded
        if self._project_key is None:
            manifest_file = self.path / "export-manifest.json"

            # Primary method: Read from export manifest (most reliable)
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest_data = json.load(f)
                self._project_key = manifest_data.get("originalProjectKey", "")
            else:
                # Fallback: Extract from bundle directory name
                # This ensures the SDK works with any bundle structure, even if manifest is missing
                bundle_name = self.path.name

                # Expected format: "dss-bundle-PROJECTKEY-date"
                if bundle_name.startswith("dss-bundle-"):
                    parts = bundle_name.split("-")
                    if len(parts) >= 3:
                        # parts[0] = "dss", parts[1] = "bundle", parts[2] = "PROJECTKEY"
                        self._project_key = parts[2]
                    else:
                        self._project_key = ""
                else:
                    # Unknown bundle naming convention
                    self._project_key = ""

        return self._project_key if self._project_key is not None else ""

    @property
    def variables(self) -> Variables:
        """
        Get variables for this bundle.

        Variables in Dataiku can be defined at two levels:
        - Global variables (project-level): Stored in variables.json
        - Local variables: Stored in localvariables.json

        Returns:
            Variables object with access to global and local variables

        Example:
            >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
            >>> bundle.variables._global["my_var"]
            'some_value'
            >>> bundle.variables.local
            {}
        """
        if self._variables is None:
            self._variables = Variables(self.path)
        return self._variables

    def get_managed_folder(self, name: str) -> ManagedFolder | None:
        """
        Get a specific managed folder by name.

        Args:
            name: Name of the managed folder to find

        Returns:
            ManagedFolder object if found, None otherwise

        Example:
            >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
            >>> folder = bundle.get_managed_folder("model-store")
            >>> if folder:
            ...     print(f"Bucket: {folder.bucket}")
        """
        return get_managed_folder(self, name)


def generate_component_recipes(bundle: DataikuBundle, path: str) -> None:
    """
    Generate Kubeflow component Python files for all recipes in a Dataiku bundle.

    This function creates individual Python component files for each recipe in the bundle,
    using Jinja2 templates to generate KFP (Kubeflow Pipelines) component code. Each
    component file includes the recipe's inputs, outputs, original transformation code,
    comprehensive metadata (description, tags, labels, modification history), and
    detailed information about input/output datasets.

    Args:
        bundle: DataikuBundle instance to extract recipes from
        path: Directory path where component files will be written

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> generate_component_recipes(bundle, "/tmp/components")
        # Creates files like:
        # /tmp/components/compute_adjusting_daily_sales.py
        # /tmp/components/join_customer_data.py
        # etc.
    """
    import re
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    from .recipe_analyzer import RecipeAnalyzer

    # Get all recipes from the bundle
    recipes = get_recipe_list(bundle)

    # Setup Jinja environment for templates
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))

    # Ensure output directory exists
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize RecipeAnalyzer for extracting comprehensive metadata
    analyzer = RecipeAnalyzer(bundle)

    # Generate a component file for each recipe
    for recipe in recipes:
        # Sanitize dataset names for Python identifiers
        def sanitize_name(name: str) -> str:
            """Convert dataset name to valid Python identifier."""
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
            sanitized = re.sub(r"_+", "_", sanitized)
            sanitized = sanitized.strip("_")
            if sanitized and sanitized[0].isdigit():
                sanitized = "_" + sanitized
            return sanitized or "dataset"

        # Get recipe metadata
        inputs = [ds.id for ds in recipe.inputs]
        outputs = [ds.id for ds in recipe.outputs]
        recipe_type = recipe.type.value if recipe.type else "unknown"

        # Get original code (SQL or Python)
        original_code = recipe.sql or recipe.code or "# No code available"

        # Create parameter mappings (original name -> sanitized name)
        input_params = {ds_id: sanitize_name(ds_id) for ds_id in inputs}
        output_params = {ds_id: sanitize_name(ds_id) for ds_id in outputs}

        # Extract comprehensive metadata using RecipeAnalyzer
        recipe_metadata = None
        try:
            recipe_info = analyzer.analyze_recipe(recipe.id)
            recipe_metadata = analyzer.format_as_comment(recipe_info, max_width=96)
        except Exception:
            # If analysis fails, continue without metadata
            # This ensures the function works even if some recipes can't be fully analyzed
            pass

        # Render the template
        template = env.get_template("recipe_component.py.j2")
        rendered = template.render(
            recipe_id=recipe.id,
            recipe_type=recipe_type,
            input_datasets=inputs,
            output_datasets=outputs,
            input_params=input_params,
            output_params=output_params,
            input_lineage={},  # Empty lineage for now
            output_lineage={},  # Empty lineage for now
            non_root_inputs=inputs,  # All inputs are non-root in this context
            recipe_description=f"Recipe: {recipe.id}",
            original_code=original_code,
            recipe_metadata=recipe_metadata,
            base_image="python:3.10-slim",
        )

        # Write to file
        output_file = output_path / f"{recipe.id}.py"
        with open(output_file, "w") as f:
            f.write(rendered)
