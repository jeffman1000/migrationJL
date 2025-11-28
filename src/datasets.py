"""
Dataset management for Dataiku bundles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import Column, datatype
from .utils import DatasetModes, find_file_case_insensitive

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle
    from .recipes import Recipe


@dataclass
class Dataset:
    """
    Represents a dataset in a Dataiku bundle.

    Datasets are the primary data storage objects in Dataiku. They can be
    stored in various backends (Snowflake, PostgreSQL, files, etc.) and have
    different modes (managed by recipes, read from tables, defined by queries).

    This class uses lazy loading: metadata is only loaded from JSON files when
    properties are first accessed, improving performance when working with
    large bundles.

    Attributes:
        id: Dataset identifier (normalized to lowercase for case-insensitive matching)
        bundle: DataikuBundle instance that this dataset belongs to
        _type: Internal storage for dataset type (Snowflake, PostgreSQL, etc.) - lazy-loaded
        _recipe: Recipe that produces this dataset (lazy-loaded)
        _recipes: All recipes that reference this dataset (lazy-loaded)
        _mode: Dataset mode - managed, table, view, etc. (lazy-loaded)
        _columns: List of columns in the dataset (lazy-loaded)
        _source_table: Source database table name if applicable (lazy-loaded)
        _source_database: Source database/catalog name if applicable (lazy-loaded)
        _source_schema: Source database schema name if applicable (lazy-loaded)
        _bundle_path: Path to bundle root (derived from bundle parameter)
        _data_loaded: Flag to prevent infinite loops during lazy loading
    """

    id: str
    bundle: DataikuBundle | str | Path | None = field(default=None, repr=False)
    _type: str | None = field(default=None, repr=False)
    _recipe: Recipe | None = field(default=None, repr=False)
    _recipes: list[Recipe] | None = field(default=None, repr=False)
    _mode: DatasetModes | None = field(default=None, repr=False)
    _columns: list[Column] | None = field(default=None, repr=False)
    _source_table: str | None = field(default=None, repr=False)
    _source_database: str | None = field(default=None, repr=False)
    _source_schema: str | None = field(default=None, repr=False)
    _bundle_path: Path | None = field(default=None, repr=False)
    _data_loaded: bool = field(default=False, repr=False, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to handle bundle path resolution and ID normalization."""
        # Normalize ID to lowercase for case-insensitive matching
        self.id = self.id.lower()

        # Set bundle path from the bundle object/string/path if provided
        if self.bundle is not None and self._bundle_path is None:
            if isinstance(self.bundle, (str, Path)):
                self._bundle_path = Path(self.bundle)
            else:
                self._bundle_path = self.bundle.path

    @property
    def type(self) -> str | None:
        """Get the dataset type, loading it if needed."""
        if (
            self._type is None
            and self._bundle_path is not None
            and not self._data_loaded
        ):
            self._load_dataset_data()
        return self._type

    @type.setter
    def type(self, value: str | None) -> None:
        """Set the dataset type."""
        self._type = value

    def __eq__(self, other: object) -> bool:
        """Compare datasets based on id only."""
        if not isinstance(other, Dataset):
            return False
        return self.id == other.id

    @property
    def recipe(self) -> Recipe | None:
        """Get the recipe that produces this dataset, loading it if needed."""
        if self._recipe is None and self._bundle_path is not None:
            self._load_dataset_data()
        return self._recipe

    @property
    def recipes(self) -> list[Recipe]:
        """Get all recipes that reference this dataset (as input or output), loading them if needed."""
        if self._recipes is None and self._bundle_path is not None:
            self._load_dataset_data()
        return self._recipes or []

    @property
    def mode(self) -> DatasetModes | None:
        """Get the mode for this dataset, loading it if needed."""
        if self._mode is None and self._bundle_path is not None:
            self._load_dataset_data()
        return self._mode

    @property
    def columns(self) -> list[Column]:
        """Get the columns for this dataset, loading them if needed. Returns sorted list."""
        if self._columns is None and self._bundle_path is not None:
            self._load_dataset_data()
        return sorted(self._columns or [])

    @property
    def source_table(self) -> str | None:
        """Get the source table for this dataset, loading it if needed."""
        if self._source_table is None and self._bundle_path is not None:
            self._load_dataset_data()
        return self._source_table

    @property
    def parent_recipe(self) -> Recipe | None:
        """Alias for recipe property - get the recipe that produces this dataset."""
        return self.recipe

    @property
    def table(self) -> str | None:
        """Alias for source_table property - get the source table for this dataset."""
        return self.source_table

    @property
    def schema(self) -> str | None:
        """Get the source schema for this dataset, loading it if needed."""
        if self._source_schema is None and self._bundle_path is not None:
            self._load_dataset_data()
        return self._source_schema

    @property
    def database(self) -> str | None:
        """Get the source database for this dataset, loading it if needed."""
        if self._source_database is None and self._bundle_path is not None:
            self._load_dataset_data()
        return self._source_database

    def _load_dataset_data(self) -> None:
        """
        Load dataset metadata from the dataset JSON file.

        This method is called when any lazy-loaded property is first accessed.
        It reads the dataset configuration file and extracts all metadata.
        """
        # Mark as loaded immediately to prevent infinite loops
        self._data_loaded = True

        if self._bundle_path is None:
            return

        dataset_data = self._load_dataset_config()
        if dataset_data is None:
            return

        self._extract_basic_metadata(dataset_data)
        self._extract_schema(dataset_data)
        self._extract_source_info(dataset_data)
        self._extract_recipe_relationships()

    def _load_dataset_config(self) -> dict[str, Any] | None:
        """Load and parse the dataset JSON configuration file."""
        if self._bundle_path is None:
            return None

        datasets_dir = self._bundle_path / "project_config" / "datasets"
        dataset_file = find_file_case_insensitive(datasets_dir, f"{self.id}.json")

        if dataset_file is None:
            return None

        with open(dataset_file) as f:
            return json.load(f)

    def _extract_basic_metadata(self, dataset_data: dict[str, Any]) -> None:
        """Extract basic dataset metadata (type and mode)."""
        # Extract dataset type (storage backend)
        if self._type is None:
            self._type = dataset_data.get("type", "")

        # Determine dataset mode
        is_managed = dataset_data.get("managed", False)
        mode_str = dataset_data.get("params", {}).get("mode", "")

        if is_managed:
            self._mode = DatasetModes.recipe_created
        elif mode_str == "table":
            self._mode = DatasetModes.read_a_database_table
        else:
            self._mode = self._convert_dataset_mode(mode_str)

    def _extract_schema(self, dataset_data: dict[str, Any]) -> None:
        """Extract column schema from dataset configuration."""
        self._columns = []
        schema_columns = dataset_data.get("schema", {}).get("columns", [])

        for col in schema_columns:
            col_name = col.get("name", "")
            col_type_str = col.get("type", "")
            if col_name and col_type_str:
                col_type = getattr(datatype, col_type_str)
                self._columns.append(Column(name=col_name, type=col_type))

    def _extract_source_info(self, dataset_data: dict[str, Any]) -> None:
        """Extract source table, database, and schema information."""
        params = dataset_data.get("params", {})
        self._source_table = params.get("table")
        self._source_database = params.get("catalog")
        self._source_schema = params.get("schema")

    def _extract_recipe_relationships(self) -> None:
        """Extract recipe relationships (producing recipe and all related recipes)."""
        self._recipe = self._find_producing_recipe()
        self._recipes = self._find_all_recipes()

    def _find_producing_recipe(self) -> Recipe | None:
        """Find the recipe that produces this dataset (has it as output)."""
        from .recipes import Recipe

        if self._bundle_path is None:
            return None

        recipes_path = self._bundle_path / "project_config" / "recipes"
        if not recipes_path.exists():
            return None

        # Search for a recipe that outputs this dataset
        for recipe_file in sorted(recipes_path.glob("*.json")):
            recipe_data = self._load_recipe_json(recipe_file)
            if self._dataset_in_recipe_outputs(recipe_data):
                return Recipe(id=recipe_file.stem, bundle=self.bundle)

        return None

    def _find_all_recipes(self) -> list[Recipe]:
        """Find all recipes that reference this dataset (either as input or output)."""
        from .recipes import Recipe

        if self._bundle_path is None:
            return []

        recipes_path = self._bundle_path / "project_config" / "recipes"
        if not recipes_path.exists():
            return []

        found_recipes = []

        # Search for all recipes that reference this dataset
        for recipe_file in sorted(recipes_path.glob("*.json")):
            recipe_data = self._load_recipe_json(recipe_file)
            recipe_id = recipe_file.stem

            if self._dataset_in_recipe_outputs(recipe_data):
                found_recipes.append(Recipe(id=recipe_id, bundle=self.bundle))
            elif self._dataset_in_recipe_inputs(recipe_data):
                # Only add if not already in the list
                if not any(r.id == recipe_id.lower() for r in found_recipes):
                    found_recipes.append(Recipe(id=recipe_id, bundle=self.bundle))

        return found_recipes

    @staticmethod
    def _load_recipe_json(recipe_file: Path) -> dict[str, Any]:
        """Load and parse a recipe JSON file."""
        with open(recipe_file) as f:
            return json.load(f)

    def _dataset_in_recipe_outputs(self, recipe_data: dict[str, Any]) -> bool:
        """Check if this dataset is in the recipe's outputs."""
        outputs_sections = recipe_data.get("outputs", {})
        for section_data in outputs_sections.values():
            if isinstance(section_data, dict) and "items" in section_data:
                for output_item in section_data.get("items", []):
                    if output_item.get("ref", "").lower() == self.id:
                        return True
        return False

    def _dataset_in_recipe_inputs(self, recipe_data: dict[str, Any]) -> bool:
        """Check if this dataset is in the recipe's inputs."""
        inputs_data = recipe_data.get("inputs", {}).get("main", {}).get("items", [])
        for input_item in inputs_data:
            if input_item.get("ref", "").lower() == self.id:
                return True
        return False

    @staticmethod
    def _convert_dataset_mode(mode_str: str) -> DatasetModes | None:
        """Convert dataset mode string to DatasetModes enum."""
        try:
            # Try to match by value first
            for mode in DatasetModes:
                if mode.value == mode_str:
                    return mode
            # If no match found, try direct name match
            return DatasetModes[mode_str]
        except (KeyError, ValueError):
            return None


def get_datasets_list(bundle: DataikuBundle) -> list[Dataset]:
    """
    Get a list of all datasets in the Dataiku bundle.

    Datasets are the core data objects in Dataiku. Each dataset has a
    corresponding JSON configuration file. This function discovers all datasets
    and creates Dataset objects with lazy-loaded metadata.

    Args:
        bundle: DataikuBundle instance to extract datasets from

    Returns:
        List of Dataset objects, sorted by dataset ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> datasets = get_datasets_list(bundle)
        >>> len(datasets)
        139
        >>> datasets[0].id
        'budget_and_delivery'
        >>> datasets[0].type  # Triggers lazy loading
        'Snowflake'
    """
    datasets: list[Dataset] = []
    datasets_path = bundle.path / "project_config" / "datasets"

    # Return empty list if datasets directory doesn't exist
    if not datasets_path.exists():
        return datasets

    # Find all dataset configuration files (.json files)
    # Each .json file corresponds to one dataset in the project
    for dataset_file in sorted(datasets_path.glob("*.json")):
        # Dataset ID is the filename without extension
        dataset_id = dataset_file.stem

        # Pre-load the type for efficiency (avoids repeated file reads)
        # Other metadata will be lazy-loaded when accessed
        with open(dataset_file) as f:
            dataset_data = json.load(f)

        dataset_type = dataset_data.get("type", "")

        # Create Dataset object with type pre-loaded
        dataset = Dataset(id=dataset_id, bundle=bundle)
        dataset.type = dataset_type  # Use property setter
        datasets.append(dataset)

    return datasets
