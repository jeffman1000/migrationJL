"""
Recipe management for Dataiku bundles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Predict
from .utils import RecipeTypes, find_file_case_insensitive

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle
    from .datasets import Dataset


@dataclass
class Recipe:
    """
    Represents a Dataiku recipe (data transformation).

    Recipes are the core transformation components in Dataiku that take input
    datasets and produce output datasets or models. This class uses lazy loading
    to defer expensive file I/O until properties are actually accessed.

    Attributes:
        id: Recipe identifier (normalized to lowercase for case-insensitive matching)
        bundle: DataikuBundle instance that this recipe belongs to
        _type: Recipe type (lazy-loaded from JSON metadata)
        _inputs: List of input datasets (lazy-loaded)
        _outputs: List of outputs - can be datasets or models for prediction recipes (lazy-loaded)
        _sql: SQL code for SQL-based recipes (lazy-loaded)
        _code: Python code for Python recipes (lazy-loaded)
        _algorithm: ML algorithm for prediction_training recipes (lazy-loaded)
        _bundle_path: Path to bundle root (derived from bundle parameter)
    """

    id: str
    bundle: DataikuBundle | None = field(default=None, repr=False)
    _type: RecipeTypes | None = field(default=None, repr=False)
    _inputs: list[Dataset] | None = field(default=None, repr=False)
    _outputs: list | None = field(
        default=None, repr=False
    )  # Can contain Dataset or Predict
    _sql: str | None = field(default=None, repr=False)
    _code: str | None = field(default=None, repr=False)
    _algorithm: str | None = field(default=None, repr=False)
    _bundle_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to handle bundle path resolution and ID normalization."""
        # Normalize ID to lowercase for case-insensitive matching
        self.id = self.id.lower()

        # Set bundle path from the bundle object if provided
        if self.bundle is not None and self._bundle_path is None:
            self._bundle_path = self.bundle.path

    def __eq__(self, other: object) -> bool:
        """Compare recipes based on id only."""
        if not isinstance(other, Recipe):
            return False
        return self.id == other.id

    def __lt__(self, other: object) -> bool:
        """Enable sorting of recipes by id."""
        if not isinstance(other, Recipe):
            return NotImplemented
        return self.id < other.id

    def __hash__(self) -> int:
        """Enable recipes to be used in sets and as dict keys."""
        return hash(self.id)

    @property
    def type(self) -> RecipeTypes | None:
        """Get the type of this recipe, loading it if needed."""
        if self._type is None and self._bundle_path is not None:
            self._load_recipe_data()
        return self._type

    @property
    def inputs(self) -> list[Dataset]:
        """Get the inputs for this recipe, loading them if needed."""
        if self._inputs is None and self._bundle_path is not None:
            self._load_recipe_data()
        return self._inputs or []

    @property
    def outputs(self) -> list[Dataset]:
        """Get the outputs for this recipe, loading them if needed."""
        if self._outputs is None and self._bundle_path is not None:
            self._load_recipe_data()
        return self._outputs or []

    @property
    def sql(self) -> str | None:
        """Get the SQL for this recipe, loading it if needed."""
        if self._sql is None and self._bundle_path is not None:
            self._load_recipe_data()
        return self._sql

    @property
    def sql_query(self) -> str | None:
        """Alias for sql property. Get the SQL for this recipe, loading it if needed."""
        return self.sql

    @property
    def code(self) -> str | None:
        """Get the Python code for this recipe, loading it if needed."""
        if self._code is None and self._bundle_path is not None:
            self._load_recipe_data()
        return self._code

    @property
    def algorithm(self) -> str | None:
        """Get the algorithm for this recipe (for prediction_training recipes), loading it if needed."""
        if self._algorithm is None and self._bundle_path is not None:
            self._load_recipe_data()
        return self._algorithm

    def _load_recipe_data(self) -> None:
        """Load recipe data from the recipe JSON file."""
        # Import here to avoid circular dependency
        from .datasets import Dataset

        if self._bundle_path is None:
            raise ValueError(f"Bundle path not set for recipe: {self.id}")

        recipes_dir = self._bundle_path / "project_config" / "recipes"
        recipe_file = find_file_case_insensitive(recipes_dir, f"{self.id}.json")

        if recipe_file is None:
            raise FileNotFoundError(f"Recipe file not found for ID: {self.id}")

        with open(recipe_file) as f:
            recipe_data = json.load(f)

        # Get recipe type
        raw_type = recipe_data.get("type", "")
        self._type = self._convert_recipe_type(raw_type)

        # Get list of managed folder IDs to exclude from dataset inputs
        managed_folder_ids = set()
        if self.bundle:
            from .managed_folders import get_managed_folders_list

            managed_folders = get_managed_folders_list(self.bundle)
            managed_folder_ids = {f.id for f in managed_folders}

        # Get input datasets (excluding managed folders)
        self._inputs = []
        inputs_data = recipe_data.get("inputs", {}).get("main", {}).get("items", [])

        for input_item in inputs_data:
            dataset_id = input_item.get("ref")
            if dataset_id and dataset_id not in managed_folder_ids:
                # Dataset constructor will normalize the ID to lowercase
                self._inputs.append(Dataset(id=dataset_id, bundle=self.bundle))

        # Get output datasets from all output sections
        self._outputs = []
        outputs_sections = recipe_data.get("outputs", {})

        # Iterate through all output sections (main, rejected, etc.)
        for _section_name, section_data in outputs_sections.items():
            if isinstance(section_data, dict) and "items" in section_data:
                for output_item in section_data.get("items", []):
                    output_ref = output_item.get("ref")
                    if output_ref:
                        # Check if this is a saved model (for prediction_training recipes)
                        if self._type == RecipeTypes.prediction_training:
                            # For prediction training recipes, outputs are Predict models
                            self._outputs.append(
                                Predict(id=output_ref, bundle=self.bundle)
                            )
                        else:
                            # For other recipes, outputs are Datasets
                            # Dataset constructor will normalize the ID to lowercase
                            self._outputs.append(
                                Dataset(id=output_ref, bundle=self.bundle)
                            )

        # Get algorithm for prediction_training recipes
        if self._type == RecipeTypes.prediction_training:
            # Algorithm is stored in the .prediction_training file, not the .json file
            pred_training_file = find_file_case_insensitive(
                recipes_dir, f"{self.id}.prediction_training"
            )
            if pred_training_file is not None:
                with open(pred_training_file) as f:
                    pred_training_data = json.load(f)
                self._algorithm = pred_training_data.get("modeling", {}).get(
                    "algorithm"
                )

        # Get SQL for the recipe
        self._sql = self._load_sql()

        # Get Python code for the recipe
        self._code = self._load_code()

    def _load_sql(self) -> str | None:
        """Load SQL for this recipe from .sql or .join files."""
        if self._bundle_path is None:
            return None

        recipes_dir = self._bundle_path / "project_config" / "recipes"

        # Check for .sql file first (for SQL recipes)
        sql_file = find_file_case_insensitive(recipes_dir, f"{self.id}.sql")
        if sql_file is not None:
            with open(sql_file) as f:
                return f.read()

        # Check for .join file (for join recipes) and generate SQL
        join_file = find_file_case_insensitive(recipes_dir, f"{self.id}.join")
        if join_file is not None:
            return self._generate_sql_from_join(join_file)

        return None

    def _generate_sql_from_join(self, join_file: Path) -> str:
        """
        Generate SQL SELECT statement from a join recipe configuration.

        Join recipes in Dataiku store their configuration in .join files (JSON).
        This method reconstructs the SQL that would be executed by the join.

        Args:
            join_file: Path to the .join configuration file

        Returns:
            SQL SELECT statement representing the join operation
        """
        if self._bundle_path is None:
            return "SELECT *"  # Fallback if bundle path not available

        # Delegate to the sql_generator module
        from .sql_generator import generate_sql_from_join

        return generate_sql_from_join(self, join_file, self._bundle_path)

    def _load_code(self) -> str | None:
        """Load Python code for this recipe from .py files."""
        if self._bundle_path is None:
            return None

        recipes_dir = self._bundle_path / "project_config" / "recipes"

        # Check for .py file (for Python recipes)
        py_file = find_file_case_insensitive(recipes_dir, f"{self.id}.py")
        if py_file is not None:
            with open(py_file) as f:
                return f.read()
        return None

    @staticmethod
    def _convert_recipe_type(raw_type: str) -> RecipeTypes | None:
        """Convert raw recipe type to RecipeTypes enum."""
        try:
            # Try to match by value first
            for recipe_type in RecipeTypes:
                if recipe_type.value == raw_type:
                    return recipe_type
            # If no match found, try direct name match
            return RecipeTypes[raw_type]
        except (KeyError, ValueError):
            # If type is unknown, return None or create a generic type
            return None


def get_recipe_list(bundle: DataikuBundle) -> list[Recipe]:
    """
    Get a list of all recipes in the Dataiku bundle.

    Each recipe in a Dataiku bundle has a corresponding .json file that defines
    its configuration. This function discovers all recipes by scanning for JSON files.
    Recipe metadata (inputs, outputs, code) is lazy-loaded when accessed.

    Args:
        bundle: DataikuBundle instance to extract recipes from

    Returns:
        List of Recipe objects, sorted by recipe ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> recipes = get_recipe_list(bundle)
        >>> len(recipes)
        91
        >>> recipes[0].id
        'compute_capacity_with_expected_size'
    """
    recipes: list[Recipe] = []
    recipes_path = bundle.path / "project_config" / "recipes"

    # Return empty list if recipes directory doesn't exist
    if not recipes_path.exists():
        return recipes

    # Find all recipe configuration files (.json files)
    # Each .json file corresponds to one recipe in the project
    for recipe_file in sorted(recipes_path.glob("*.json")):
        # Recipe ID is the filename without extension
        recipe_id = recipe_file.stem

        # Create Recipe object with lazy-loading enabled
        recipe = Recipe(id=recipe_id, bundle=bundle)
        recipes.append(recipe)

    return recipes
