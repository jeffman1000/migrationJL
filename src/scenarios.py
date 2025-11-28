"""
Scenario and Step management for Dataiku bundles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .utils import StepsList, StepTypes, find_file_case_insensitive

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle


@dataclass
class SnowflakeDataset:
    """
    Represents a Snowflake dataset in a Dataiku bundle.

    This is a legacy class used for backward compatibility with scenario steps
    that explicitly reference Snowflake datasets. For most use cases, use the
    more general Dataset class instead.

    Attributes:
        id: Dataset identifier (normalized to lowercase for case-insensitive matching)
    """

    id: str

    def __post_init__(self) -> None:
        """Post-initialization to normalize ID to lowercase for case-insensitive matching."""
        self.id = self.id.lower()

    def __eq__(self, other: object) -> bool:
        """Compare datasets based on id only."""
        if not isinstance(other, SnowflakeDataset):
            return False
        return self.id == other.id


@dataclass
class Step:
    """
    Represents a step in a Dataiku scenario.

    Scenarios are composed of multiple steps that execute sequentially.
    Each step has a specific type and may contain different kinds of metadata
    depending on that type (e.g., Python scripts, datasets to build, scenarios to run).

    Attributes:
        id: Step number (1-indexed, matching Dataiku's numbering convention)
        type: Type of step (custom_python, build, compute_metrics, etc.)
        desc: Human-readable description of the step
        script: Python code (for custom_python steps)
        tables: List of Snowflake datasets (legacy attribute for build steps)
        items: List of items to process - can be Dataset or Predict objects
        scenario: Scenario to run (for run_scenario steps)
    """

    id: int
    type: StepTypes
    desc: str
    script: str | None = None
    tables: list[SnowflakeDataset] | None = None
    items: list[Any] | None = None  # Can contain Dataset or Predict objects
    scenario: Scenario | None = None  # For run_scenario steps

    def __eq__(self, other: object) -> bool:
        """Compare steps based on id, type, and desc (ignore script, tables, items, and scenario for equality)."""
        if not isinstance(other, Step):
            return False
        return (
            self.id == other.id and self.type == other.type and self.desc == other.desc
        )


@dataclass
class Scenario:
    """
    Represents a Dataiku scenario (workflow automation).

    Scenarios in Dataiku are automated workflows that orchestrate data processing.
    They consist of multiple steps that execute in sequence to build datasets,
    run metrics, execute custom code, or trigger other scenarios.

    Attributes:
        id: Scenario identifier (normalized to lowercase for case-insensitive matching)
        desc: Human-readable description/name of the scenario
        _steps: List of steps in this scenario (lazy-loaded from JSON)
        _bundle_path: Path to bundle root (used for loading step metadata)
    """

    id: str
    desc: str
    _steps: StepsList | None = field(default=None, repr=False)
    _bundle_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to normalize ID to lowercase."""
        self.id = self.id.lower()

    def __eq__(self, other: object) -> bool:
        """Compare scenarios based on id and desc only."""
        if not isinstance(other, Scenario):
            return False
        return self.id == other.id and self.desc == other.desc

    @property
    def steps(self) -> StepsList:
        """Get the steps for this scenario, loading them if needed."""
        if self._steps is None and self._bundle_path is not None:
            self._load_steps()
        return self._steps or StepsList()

    def _load_steps(self) -> None:
        """
        Load steps from the scenario JSON file.

        Parses the scenario configuration to extract all steps and their metadata.
        Each step type may have different attributes that need to be extracted.

        Handles both step-based scenarios (with multiple steps) and custom_python
        scenarios (which are treated as a single step).
        """
        if self._bundle_path is None:
            return

        scenario_data = self._load_scenario_file()
        if scenario_data is None:
            return

        self._steps = StepsList()

        # Check if this is a custom_python scenario (not step-based)
        scenario_type = scenario_data.get("type", "")
        if scenario_type == "custom_python":
            # Treat the entire scenario as a single custom_python step
            # Load the Python script from the scenario's .py file
            script = self._load_custom_python_script()
            step = Step(
                id=1,
                type=StepTypes.custom_python,
                desc=scenario_data.get("name", "Run custom Python script"),
                script=script,
                tables=None,
                items=None,
                scenario=None,
            )
            self._steps.append(step)
            return

        # Otherwise, process step-based scenarios
        steps_data = scenario_data.get("params", {}).get("steps", [])

        # Process each step in the scenario
        # Steps are numbered starting from 1 (matching Dataiku's convention)
        for idx, step_data in enumerate(steps_data, start=1):
            step = self._create_step_from_data(idx, step_data)
            self._steps.append(step)

    def _load_scenario_file(self) -> dict[str, Any] | None:
        """Load and parse the scenario JSON file."""
        if self._bundle_path is None:
            return None

        scenarios_dir = self._bundle_path / "project_config" / "scenarios"
        scenario_file = find_file_case_insensitive(scenarios_dir, f"{self.id}.json")

        if scenario_file is None:
            return None

        with open(scenario_file) as f:
            return json.load(f)

    def _load_custom_python_script(self) -> str | None:
        """Load the Python script for a custom_python scenario."""
        if self._bundle_path is None:
            return None

        scenarios_dir = self._bundle_path / "project_config" / "scenarios"
        # Look for the .py file with the same name as the scenario (case-insensitive)
        script_file = find_file_case_insensitive(scenarios_dir, f"{self.id}.py")

        if script_file is None:
            return None

        with open(script_file) as f:
            return f.read()

    def _create_step_from_data(self, idx: int, step_data: dict[str, Any]) -> Step:
        """Create a Step object from raw step data."""
        step_type = self._convert_step_type(step_data.get("type", ""))
        step_name = step_data.get("name", "")

        # Extract step-type-specific metadata
        script = self._extract_script(step_data)
        tables = self._extract_legacy_tables(step_data)
        items = self._extract_items(step_data)
        scenario = self._extract_scenario(step_data)

        return Step(
            id=idx,
            type=step_type,
            desc=step_name,
            script=script,
            tables=tables,
            items=items,
            scenario=scenario,
        )

    def _extract_script(self, step_data: dict[str, Any]) -> str | None:
        """Extract Python script from custom_python steps."""
        if step_data.get("type") == "custom_python":
            return step_data.get("params", {}).get("script")
        return None

    def _extract_legacy_tables(
        self, step_data: dict[str, Any]
    ) -> list[SnowflakeDataset] | None:
        """Extract legacy Snowflake tables from build_flowitem steps."""
        if step_data.get("type") == "build_flowitem":
            return self._parse_build_datasets(step_data)
        return None

    def _extract_items(self, step_data: dict[str, Any]) -> list[Any] | None:
        """Extract items (datasets/models) based on step type."""
        step_type = step_data.get("type")

        if step_type == "build_flowitem":
            return self._parse_build_items(step_data)
        elif step_type == "compute_metrics":
            return self._parse_compute_metrics_items(step_data)
        elif step_type == "check_dataset":
            return self._parse_check_dataset_items(step_data)

        return None

    def _extract_scenario(self, step_data: dict[str, Any]) -> Scenario | None:
        """Extract scenario reference from run_scenario steps."""
        if step_data.get("type") == "run_scenario":
            return self._parse_run_scenario(step_data)
        return None

    def _parse_build_datasets(
        self, step_data: dict[str, Any]
    ) -> list[SnowflakeDataset]:
        """Parse datasets from a build_flowitem step (legacy Snowflake-only)."""
        tables: list[SnowflakeDataset] = []
        builds = step_data.get("params", {}).get("builds", [])

        if self._bundle_path is None:
            return tables

        for build_item in builds:
            if build_item.get("type") == "DATASET":
                dataset_id = build_item.get("itemId")
                if dataset_id:
                    # Check if this is a Snowflake dataset
                    datasets_dir = self._bundle_path / "project_config" / "datasets"
                    dataset_file = find_file_case_insensitive(
                        datasets_dir, f"{dataset_id}.json"
                    )
                    if dataset_file is not None:
                        with open(dataset_file) as f:
                            dataset_data = json.load(f)

                        # Check if the dataset type is Snowflake
                        if dataset_data.get("type") == "Snowflake":
                            tables.append(SnowflakeDataset(id=dataset_id))

        return tables

    def _parse_build_items(self, step_data: dict[str, Any]) -> list[Any]:
        """Parse all items (datasets and models) from a build_flowitem step."""
        # Import here to avoid circular dependency
        from .datasets import Dataset
        from .recipes import Predict

        items: list[Any] = []
        builds = step_data.get("params", {}).get("builds", [])

        for build_item in builds:
            item_type = build_item.get("type")
            item_id = build_item.get("itemId")

            if item_id:
                if item_type == "DATASET":
                    # This is a dataset
                    items.append(Dataset(id=item_id, _bundle_path=self._bundle_path))
                elif item_type == "SAVED_MODEL":
                    # This is a saved model (Predict)
                    items.append(Predict(id=item_id, _bundle_path=self._bundle_path))

        return items

    def _parse_compute_metrics_items(self, step_data: dict[str, Any]) -> list[Any]:
        """Parse datasets from a compute_metrics step."""
        # Import here to avoid circular dependency
        from .datasets import Dataset

        items: list[Any] = []
        computes = step_data.get("params", {}).get("computes", [])

        for compute_item in computes:
            if compute_item.get("type") == "DATASET":
                dataset_id = compute_item.get("itemId")
                if dataset_id:
                    items.append(Dataset(id=dataset_id, _bundle_path=self._bundle_path))

        return items

    def _parse_check_dataset_items(self, step_data: dict[str, Any]) -> list[Any]:
        """Parse datasets from a check_dataset step."""
        # Import here to avoid circular dependency
        from .datasets import Dataset

        items: list[Any] = []
        checks = step_data.get("params", {}).get("checks", [])

        for check_item in checks:
            if check_item.get("type") == "DATASET":
                dataset_id = check_item.get("itemId")
                if dataset_id:
                    items.append(Dataset(id=dataset_id, _bundle_path=self._bundle_path))

        return items

    def _parse_run_scenario(self, step_data: dict[str, Any]) -> Scenario | None:
        """Parse the scenario to run from a run_scenario step."""
        if self._bundle_path is None:
            return None

        scenario_id = step_data.get("params", {}).get("scenarioId")
        if scenario_id:
            # Look up the scenario in the bundle
            scenario_file = find_file_case_insensitive(
                self._bundle_path / "project_config" / "scenarios",
                f"{scenario_id}.json",
            )
            if scenario_file is not None:
                with open(scenario_file) as f:
                    scenario_data = json.load(f)
                scenario_desc = scenario_data.get("name", "")
                return Scenario(
                    id=scenario_id, desc=scenario_desc, _bundle_path=self._bundle_path
                )
        return None

    @staticmethod
    def _convert_step_type(raw_type: str) -> StepTypes:
        """Convert raw step type to StepTypes enum."""
        try:
            # Try to match by value first
            for step_type in StepTypes:
                if step_type.value == raw_type:
                    return step_type
            # If no match found, try direct name match
            return StepTypes[raw_type]
        except (KeyError, ValueError):
            # If type is unknown, default to custom_python
            # This ensures the code is generic and works with any bundle
            return StepTypes.custom_python


def get_scenario_list(bundle: DataikuBundle) -> list[Scenario]:
    """
    Get a list of all scenarios in the Dataiku bundle.

    Scenarios are workflows that automate data processing tasks. Each scenario
    has a configuration file in the scenarios directory. This function discovers
    all scenarios and creates Scenario objects (without loading steps).

    Args:
        bundle: DataikuBundle instance to extract scenarios from

    Returns:
        List of Scenario objects, sorted by scenario ID

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> scenarios = get_scenario_list(bundle)
        >>> len(scenarios)
        9
        >>> scenarios[0].id
        'build_features'
    """
    scenarios: list[Scenario] = []

    # Find all scenario configuration files (.json files)
    # Each .json file corresponds to one scenario in the project
    for scenario_file in sorted(bundle.scenarios_path.glob("*.json")):
        # Scenario ID is the filename without extension
        scenario_id = scenario_file.stem

        # Read the scenario JSON to extract the human-readable name
        with open(scenario_file) as f:
            scenario_data = json.load(f)

        scenario_desc = scenario_data.get("name", "")

        # Create Scenario object (steps will be lazy-loaded when accessed)
        scenario = Scenario(
            id=scenario_id, desc=scenario_desc, _bundle_path=bundle.path
        )
        scenarios.append(scenario)

    return scenarios


def get_scenario(id: str, bundle: DataikuBundle) -> Scenario:
    """
    Get a specific scenario by ID.

    Args:
        bundle: DataikuBundle instance
        id: Scenario ID

    Returns:
        Scenario object with steps loaded
    """
    # Normalize the ID to lowercase for case-insensitive matching
    id_lower = id.lower()
    scenario_file = find_file_case_insensitive(
        bundle.scenarios_path, f"{id_lower}.json"
    )

    if scenario_file is None:
        raise FileNotFoundError(f"Scenario not found: {id}")

    # Read the scenario JSON to get the description
    with open(scenario_file) as f:
        scenario_data = json.load(f)

    scenario_desc = scenario_data.get("name", "")

    scenario = Scenario(id=id, desc=scenario_desc, _bundle_path=bundle.path)

    # Trigger loading of steps
    _ = scenario.steps

    return scenario


def get_scenario_steps(scenario: Scenario) -> StepsList:
    """
    Get the steps for a scenario.

    Args:
        scenario: Scenario object

    Returns:
        StepsList of Step objects (supports 1-indexed access)
    """
    return scenario.steps


def get_scenario_yaml(bundle: DataikuBundle) -> dict[str, Any]:
    """
    Generate a YAML-compatible dictionary representation of all scenarios in the bundle.

    Args:
        bundle: DataikuBundle instance to extract scenarios from

    Returns:
        Dictionary with bundle metadata and scenario details, formatted for YAML export

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> yaml_data = get_scenario_yaml(bundle)
        >>> yaml_data['bundle']['project_key']
        'MYPROJECT'
        >>> len(yaml_data['scenarios'])
        12
    """
    # Get all scenarios
    scenarios = get_scenario_list(bundle)

    # Build the output structure (without path)
    result = {
        "bundle": {
            "project_key": bundle.project_key,
            "total_scenarios": len(scenarios),
        },
        "scenarios": [],
    }

    # Process each scenario
    for scenario in scenarios:
        scenario_dict = {"name": scenario.id, "description": scenario.desc, "steps": []}

        # Get steps for this scenario
        steps = scenario.steps

        # Process all steps
        for step in steps:
            step_dict = {
                "step_number": step.id,  # Use original step ID from the scenario
                "type": step.type.value,
                "description": step.desc,
            }

            # Add type-specific attributes
            if step.type == StepTypes.build and step.items:
                # Add list of datasets/models being built (only if non-empty)
                builds = [item.id for item in step.items]
                if builds:  # Only add if there are actual builds
                    step_dict["builds"] = builds

            elif step.type == StepTypes.custom_python and step.script:
                # Add script metadata (only if script exists and is non-empty)
                step_dict["has_script"] = True
                step_dict["script_length"] = len(step.script)

            elif step.type == StepTypes.run_scenario and step.scenario:
                # Add scenario reference
                step_dict["calls_scenario"] = step.scenario.id

            scenario_dict["steps"].append(step_dict)

        # Only add scenario if it has steps
        if scenario_dict["steps"]:
            result["scenarios"].append(scenario_dict)

    return result
