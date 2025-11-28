"""
Recipe Analyzer for Dataiku Bundles

Scans a Dataiku bundle and extracts comprehensive information about a given recipe.
This information can be used to enhance comments in generated component recipes.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    name: str
    dataset_type: str | None = None
    description: str | None = None
    columns: list[dict[str, Any]] = field(default_factory=list)
    connection: str | None = None
    table_name: str | None = None
    created_by: str | None = None
    created_on: datetime | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class RecipeInfo:
    """Comprehensive information about a Dataiku recipe."""

    name: str
    recipe_type: str | None = None
    description: str | None = None

    # Input/Output datasets
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    # Metadata
    created_by: str | None = None
    created_on: datetime | None = None
    modified_by: str | None = None
    modified_on: datetime | None = None

    # Code content
    code: str | None = None
    code_type: str | None = None  # 'python', 'sql', or None

    # Additional metadata
    labels: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    custom_fields: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    # Related dataset information
    input_datasets_info: list[DatasetInfo] = field(default_factory=list)
    output_datasets_info: list[DatasetInfo] = field(default_factory=list)

    # Raw JSON for advanced use cases
    raw_json: dict[str, Any] | None = None


class RecipeAnalyzer:
    """Analyzes Dataiku bundle recipes and extracts comprehensive information."""

    def __init__(self, bundle: "DataikuBundle"):
        """
        Initialize the analyzer with a DataikuBundle instance.

        Args:
            bundle: DataikuBundle instance to analyze
        """
        self.bundle = bundle
        self.bundle_path = bundle.path
        self.recipes_path = self.bundle_path / "project_config" / "recipes"
        self.datasets_path = self.bundle_path / "project_config" / "datasets"

        if not self.bundle_path.exists():
            raise ValueError(f"Bundle path does not exist: {self.bundle_path}")
        if not self.recipes_path.exists():
            raise ValueError(f"Recipes path does not exist: {self.recipes_path}")

    def analyze_recipe(self, recipe_name: str) -> RecipeInfo:
        """
        Analyze a recipe and extract all available information.

        Args:
            recipe_name: Name of the recipe (without extension)

        Returns:
            RecipeInfo object containing all extracted information
        """
        from .utils import find_file_case_insensitive

        info = RecipeInfo(name=recipe_name)

        # Read recipe JSON metadata (case-insensitive)
        json_file = find_file_case_insensitive(self.recipes_path, f"{recipe_name}.json")
        if json_file is not None:
            with open(json_file) as f:
                recipe_json = json.load(f)
                info.raw_json = recipe_json
                self._extract_from_json(info, recipe_json)

        # Read recipe code (Python or SQL)
        info.code, info.code_type = self._read_recipe_code(recipe_name)

        # Get detailed information about input datasets
        for input_name in info.inputs:
            dataset_info = self._get_dataset_info(input_name)
            if dataset_info:
                info.input_datasets_info.append(dataset_info)

        # Get detailed information about output datasets
        for output_name in info.outputs:
            dataset_info = self._get_dataset_info(output_name)
            if dataset_info:
                info.output_datasets_info.append(dataset_info)

        return info

    def _extract_from_json(self, info: RecipeInfo, recipe_json: dict[str, Any]):
        """Extract information from recipe JSON metadata."""
        # Basic metadata
        info.recipe_type = recipe_json.get("type")
        info.description = recipe_json.get("shortDesc")
        info.labels = recipe_json.get("labels", [])
        info.tags = recipe_json.get("tags", [])
        info.custom_fields = recipe_json.get("customFields", {})
        info.variables = recipe_json.get("variables", {})
        info.params = recipe_json.get("params", {})

        # Creation metadata
        creation_tag = recipe_json.get("creationTag", {})
        if "lastModifiedBy" in creation_tag:
            info.modified_by = creation_tag["lastModifiedBy"].get("login")
        if "lastModifiedOn" in creation_tag:
            timestamp = (
                creation_tag["lastModifiedOn"] / 1000
            )  # Convert from milliseconds
            info.modified_on = datetime.fromtimestamp(timestamp)

        # Extract inputs
        inputs_section = recipe_json.get("inputs", {}).get("main", {}).get("items", [])
        info.inputs = [item["ref"] for item in inputs_section]

        # Extract outputs
        outputs_section = (
            recipe_json.get("outputs", {}).get("main", {}).get("items", [])
        )
        info.outputs = [item["ref"] for item in outputs_section]

    def _read_recipe_code(self, recipe_name: str) -> tuple[str | None, str | None]:
        """
        Read recipe code file (Python or SQL).

        Returns:
            Tuple of (code_content, code_type)
        """
        from .utils import find_file_case_insensitive

        # Try Python file (case-insensitive)
        py_file = find_file_case_insensitive(self.recipes_path, f"{recipe_name}.py")
        if py_file is not None:
            with open(py_file) as f:
                return f.read(), "python"

        # Try SQL file (case-insensitive)
        sql_file = find_file_case_insensitive(self.recipes_path, f"{recipe_name}.sql")
        if sql_file is not None:
            with open(sql_file) as f:
                return f.read(), "sql"

        return None, None

    def _get_dataset_info(self, dataset_name: str) -> DatasetInfo | None:
        """Get detailed information about a dataset."""
        from .utils import find_file_case_insensitive

        # Find dataset file (case-insensitive)
        dataset_file = find_file_case_insensitive(
            self.datasets_path, f"{dataset_name}.json"
        )

        if dataset_file is None:
            return None

        try:
            with open(dataset_file) as f:
                dataset_json = json.load(f)

            info = DatasetInfo(name=dataset_name)
            info.dataset_type = dataset_json.get("type")
            info.description = dataset_json.get("shortDesc")
            info.tags = dataset_json.get("tags", [])

            # Schema information
            schema = dataset_json.get("schema", {})
            info.columns = schema.get("columns", [])

            # Connection information
            params = dataset_json.get("params", {})
            info.connection = params.get("connection")
            info.table_name = params.get("table")

            # Creation metadata
            creation_tag = dataset_json.get("creationTag", {})
            if "lastModifiedBy" in creation_tag:
                info.created_by = creation_tag["lastModifiedBy"].get("login")
            if "lastModifiedOn" in creation_tag:
                timestamp = creation_tag["lastModifiedOn"] / 1000
                info.created_on = datetime.fromtimestamp(timestamp)

            return info
        except Exception as e:
            print(f"Warning: Could not read dataset {dataset_name}: {e}")
            return None

    def _format_header(self, recipe_info: RecipeInfo, max_width: int) -> list[str]:
        """Format the header section."""
        return [
            "=" * max_width,
            f"RECIPE: {recipe_info.name}",
            "=" * max_width,
            "",
        ]

    def _format_basic_info(self, recipe_info: RecipeInfo) -> list[str]:
        """Format basic recipe information (type, description)."""
        lines = []
        if recipe_info.recipe_type:
            lines.append(f"Type: {recipe_info.recipe_type}")
        if recipe_info.description:
            lines.append(f"Description: {recipe_info.description}")
        if lines:
            lines.append("")
        return lines

    def _format_dataset_with_details(
        self, dataset_name: str, datasets_info: list[DatasetInfo]
    ) -> list[str]:
        """Format a single dataset with its details."""
        lines = [f"  - {dataset_name}"]
        ds_info = next((ds for ds in datasets_info if ds.name == dataset_name), None)
        if ds_info:
            if ds_info.description:
                lines.append(f"    Description: {ds_info.description}")
            if ds_info.columns:
                lines.append(f"    Columns: {len(ds_info.columns)}")
        return lines

    def _format_inputs_outputs(self, recipe_info: RecipeInfo) -> list[str]:
        """Format inputs and outputs sections."""
        lines = []

        if recipe_info.inputs:
            lines.append("Inputs:")
            for inp in recipe_info.inputs:
                lines.extend(
                    self._format_dataset_with_details(
                        inp, recipe_info.input_datasets_info
                    )
                )

        if recipe_info.outputs:
            if lines:
                lines.append("")
            lines.append("Outputs:")
            for out in recipe_info.outputs:
                lines.extend(
                    self._format_dataset_with_details(
                        out, recipe_info.output_datasets_info
                    )
                )

        if lines:
            lines.append("")
        return lines

    def _format_metadata(self, recipe_info: RecipeInfo) -> list[str]:
        """Format metadata section (modified_by, modified_on)."""
        lines = []
        if recipe_info.modified_by or recipe_info.modified_on:
            lines.append("Metadata:")
            if recipe_info.modified_by:
                lines.append(f"  Modified by: {recipe_info.modified_by}")
            if recipe_info.modified_on:
                lines.append(
                    f"  Modified on: {recipe_info.modified_on.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            lines.append("")
        return lines

    def _format_tags_labels(self, recipe_info: RecipeInfo) -> list[str]:
        """Format tags and labels section."""
        lines = []
        if recipe_info.tags or recipe_info.labels:
            if recipe_info.tags:
                lines.append(f"Tags: {', '.join(recipe_info.tags)}")
            if recipe_info.labels:
                lines.append(f"Labels: {', '.join(recipe_info.labels)}")
            lines.append("")
        return lines

    def _format_variables(self, recipe_info: RecipeInfo) -> list[str]:
        """Format variables section."""
        lines = []
        if recipe_info.variables:
            lines.append("Variables:")
            for key, value in recipe_info.variables.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        return lines

    def _format_code_section(
        self, recipe_info: RecipeInfo, max_width: int
    ) -> list[str]:
        """Format the code section with header and footer."""
        lines = []

        # Code type indicator
        if recipe_info.code_type:
            lines.append(f"Implementation: {recipe_info.code_type.upper()}")
            lines.append("")

        # Actual code content
        if recipe_info.code:
            code_type = (
                recipe_info.code_type.upper() if recipe_info.code_type else "UNKNOWN"
            )
            lines.extend(
                [
                    "=" * max_width,
                    f"RECIPE CODE ({code_type})",
                    "=" * max_width,
                    "",
                ]
            )
            # Add each line of code
            lines.extend(recipe_info.code.split("\n"))
            lines.extend(
                [
                    "",
                    "=" * max_width,
                    "END RECIPE CODE",
                    "=" * max_width,
                    "",
                ]
            )

        return lines

    def format_as_comment(self, recipe_info: RecipeInfo, max_width: int = 100) -> str:
        """
        Format recipe information as a comprehensive comment block.

        Args:
            recipe_info: RecipeInfo object to format
            max_width: Maximum width of comment lines

        Returns:
            Formatted comment string
        """
        lines = []

        # Build comment sections using helper methods
        lines.extend(self._format_header(recipe_info, max_width))
        lines.extend(self._format_basic_info(recipe_info))
        lines.extend(self._format_inputs_outputs(recipe_info))
        lines.extend(self._format_metadata(recipe_info))
        lines.extend(self._format_tags_labels(recipe_info))
        lines.extend(self._format_variables(recipe_info))
        lines.extend(self._format_code_section(recipe_info, max_width))

        # Footer
        lines.append("=" * max_width)

        return "\n".join(lines)

    def get_summary_dict(self, recipe_info: RecipeInfo) -> dict[str, Any]:
        """
        Get recipe information as a dictionary for programmatic use.

        Args:
            recipe_info: RecipeInfo object to convert

        Returns:
            Dictionary with recipe information
        """
        return {
            "name": recipe_info.name,
            "type": recipe_info.recipe_type,
            "description": recipe_info.description,
            "inputs": recipe_info.inputs,
            "outputs": recipe_info.outputs,
            "code_type": recipe_info.code_type,
            "modified_by": recipe_info.modified_by,
            "modified_on": recipe_info.modified_on.isoformat()
            if recipe_info.modified_on
            else None,
            "tags": recipe_info.tags,
            "labels": recipe_info.labels,
            "variables": recipe_info.variables,
            "input_datasets": [
                {
                    "name": ds.name,
                    "type": ds.dataset_type,
                    "description": ds.description,
                    "column_count": len(ds.columns),
                }
                for ds in recipe_info.input_datasets_info
            ],
            "output_datasets": [
                {
                    "name": ds.name,
                    "type": ds.dataset_type,
                    "description": ds.description,
                    "column_count": len(ds.columns),
                }
                for ds in recipe_info.output_datasets_info
            ],
        }


def main():
    """Example usage of the RecipeAnalyzer."""
    import sys

    from .dataiku_bundle import DataikuBundle

    if len(sys.argv) < 3:
        print("Usage: uv run python src/recipe_analyzer.py <bundle_path> <recipe_name>")
        print("\nExample:")
        print(
            "  uv run python src/recipe_analyzer.py dss-bundle-SDKEXPLORATION-2025-10-15 compute_budget_and_delivery_distinct"
        )
        sys.exit(1)

    bundle_path = sys.argv[1]
    recipe_name = sys.argv[2]

    # Create DataikuBundle instance
    bundle = DataikuBundle(bundle_path)
    analyzer = RecipeAnalyzer(bundle)
    recipe_info = analyzer.analyze_recipe(recipe_name)

    # Print formatted comment
    print(analyzer.format_as_comment(recipe_info))

    # Optionally print as JSON
    if len(sys.argv) > 3 and sys.argv[3] == "--json":
        import json

        print("\n" + "=" * 100)
        print("JSON FORMAT:")
        print("=" * 100)
        print(json.dumps(analyzer.get_summary_dict(recipe_info), indent=2))


if __name__ == "__main__":
    main()
