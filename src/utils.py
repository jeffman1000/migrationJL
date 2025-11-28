"""
Utility functions and enums shared across Dataiku SDK modules.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, SupportsIndex, overload

if TYPE_CHECKING:
    from .scenarios import Step


def find_file_case_insensitive(directory: Path, filename: str) -> Path | None:
    """
    Find a file in a directory using case-insensitive matching.

    This is essential for cross-platform compatibility, as Dataiku bundles
    may use different casing conventions depending on the export source.

    Args:
        directory: Directory to search in
        filename: Filename to search for (case-insensitive)

    Returns:
        Path to the found file, or None if not found

    Example:
        >>> find_file_case_insensitive(Path("recipes"), "MyRecipe.json")
        Path("recipes/myrecipe.json")  # Finds file regardless of casing
    """
    # Return None early if directory doesn't exist
    if not directory.exists():
        return None

    # Normalize the search filename to lowercase for comparison
    filename_lower = filename.lower()

    # Iterate through all files in the directory
    for file in directory.iterdir():
        # Compare lowercased filenames for case-insensitive match
        if file.name.lower() == filename_lower:
            return file

    return None


def extract_managed_folder_ids_from_code(code: str) -> list[str]:
    """
    Extract managed folder IDs from Dataiku recipe code.

    Scans Python code for dataiku.Folder() calls and extracts the folder ID.
    Supports both direct string literals and variable references.

    Args:
        code: Python code string to scan

    Returns:
        List of managed folder IDs found in the code

    Example:
        >>> code = 'folder = dataiku.Folder("48kCywFK")'
        >>> extract_managed_folder_ids_from_code(code)
        ['48kCywFK']
    """
    if not code:
        return []

    # Pattern to match: dataiku.Folder("FOLDER_ID") or dataiku.Folder('FOLDER_ID')
    # This captures the folder ID within quotes
    pattern = r'dataiku\.Folder\(["\']([^"\']+)["\']\)'

    matches = re.findall(pattern, code)

    # Return unique IDs (convert to set then back to list)
    return list(set(matches))


class StepTypes(str, Enum):
    """
    Enumeration of Dataiku scenario step types.

    Each scenario step in Dataiku has a specific type that determines its behavior.
    This enum provides a type-safe way to work with step types.

    Attributes:
        custom_python: Execute custom Python code
        build: Build datasets or models (corresponds to "build_flowitem" in JSON)
        compute_metrics: Compute metrics for datasets
        check_dataset: Check dataset quality/rules
        verify_rules_or_run_checks: Alias for check_dataset (alternative naming)
        run_scenario: Trigger execution of another scenario
        pull_git_refs: Pull updates from Git references
        invalidate_cache: Invalidate cache for specific items
        exec_sql: Execute SQL queries
        step: Generic step (for custom_python scenarios without explicit type)
    """

    custom_python = "custom_python"
    build = "build_flowitem"
    compute_metrics = "compute_metrics"
    check_dataset = "check_dataset"
    verify_rules_or_run_checks = "check_dataset"  # Alias for check_dataset
    run_scenario = "run_scenario"
    pull_git_refs = "pull_git_refs"
    invalidate_cache = "invalidate_cache"
    exec_sql = "exec_sql"
    step = "step"  # Generic operational step

    def __str__(self) -> str:
        """Return the enum name (not the value) for display."""
        return self.name


class RecipeTypes(str, Enum):  # type: ignore[assignment]
    """
    Enumeration of Dataiku recipe types.

    Recipes are the core data transformation components in Dataiku.
    Each recipe type represents a different transformation pattern.

    Attributes:
        join: Join/merge multiple datasets
        grouping: Aggregate data by groups
        window: Apply window functions
        sort: Sort dataset rows
        distinct: Remove duplicate rows
        prepare: Data preparation steps
        python: Custom Python transformation
        sql: SQL transformation
        sync: Synchronize datasets
        sample_filter: Sample or filter rows (corresponds to "sampling" in JSON)
        sql_query: SQL query recipe
        data_preparation: Visual data preparation (corresponds to "shaker" in JSON)
        stack: Vertically stack datasets (union, corresponds to "vstack" in JSON)
        prediction_training: Train ML prediction models
    """

    join = "join"
    grouping = "grouping"
    window = "window"
    sort = "sort"
    distinct = "distinct"
    prepare = "prepare"
    python = "python"
    sql = "sql"
    sync = "sync"
    sample_filter = "sampling"
    sql_query = "sql_query"
    data_preparation = "shaker"
    stack = "vstack"
    prediction_training = "prediction_training"

    def __str__(self) -> str:
        """Return the enum name (not the value) for display."""
        return self.name


class DatasetTypes(str, Enum):
    """
    Enumeration of Dataiku dataset types (storage backends).

    Dataiku supports multiple storage backends for datasets.
    This enum represents the most common types.

    Attributes:
        snowflake: Snowflake data warehouse
        postgresql: PostgreSQL database
        filesystem: Local or network file system
        hdfs: Hadoop Distributed File System
        s3: Amazon S3 cloud storage
    """

    snowflake = "Snowflake"
    postgresql = "PostgreSQL"
    filesystem = "Filesystem"
    hdfs = "HDFS"
    s3 = "S3"

    def __str__(self) -> str:
        """Return the enum name (not the value) for display."""
        return self.name


class DatasetModes(str, Enum):
    """
    Enumeration of Dataiku dataset modes (how data is accessed/stored).

    Dataset modes determine how Dataiku interacts with the underlying data.

    Attributes:
        table: Read from or write to a database table
        database_table: Alias for table mode
        view: Dataset is a database view
        query: Dataset is defined by a SQL query
        recipe_created: Dataset is managed and created by a recipe
        read_a_database_table: Alias for table mode (read-only perspective)
    """

    table = "table"
    database_table = "table"  # Alias for table mode
    view = "view"
    query = "query"
    recipe_created = "recipe_created"
    read_a_database_table = "table"

    def __str__(self) -> str:
        """Return the enum name (not the value) for display."""
        return self.name


class StepsList(list["Step"]):
    """
    A specialized list for scenario steps that supports both 0-indexed and 1-indexed access.

    In Dataiku scenarios, steps are traditionally numbered starting from 1 (not 0).
    This class allows natural 1-indexed access (steps[1] returns first step) while
    maintaining compatibility with standard Python list operations.

    Example:
        >>> steps = StepsList([step1, step2, step3])
        >>> steps[1]  # Returns step1 (first step, 1-indexed)
        >>> steps[0]  # Returns step3 (last step, 0-indexed like normal Python)
        >>> len(steps) == 3  # Standard comparison
        >>> steps == 3  # Special: compares length to integer
    """

    @overload
    def __getitem__(self, index: SupportsIndex) -> Step: ...

    @overload
    def __getitem__(self, index: slice) -> list[Step]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> Step | list[Step]:
        """
        Get a step by index. Supports 1-indexed access (steps[1] returns first step).

        Args:
            index: Integer index or slice. Positive integers use 1-based indexing,
                   0 and negative integers use standard Python indexing.

        Returns:
            Step object or list of Step objects

        Behavior:
            - steps[1]: First step (1-indexed, Dataiku-style)
            - steps[2]: Second step (1-indexed)
            - steps[0]: Last step (0-indexed, Python-style)
            - steps[-1]: Last step (negative indexing works as normal)
        """
        if isinstance(index, int) and index > 0:
            # 1-indexed access: steps[1] returns the first step
            # Subtract 1 to convert to 0-based list indexing
            return super().__getitem__(index - 1)
        else:
            # 0-indexed or negative indices work as normal Python lists
            return super().__getitem__(index)

    def __eq__(self, other: object) -> bool:
        """
        Allow comparison with integers (compares length) or lists (compares contents).

        Args:
            other: Integer (compares to length) or list (compares items)

        Returns:
            True if equal, False otherwise

        Example:
            >>> steps == 3  # Compares length
            >>> steps == [step1, step2]  # Compares contents
        """
        if isinstance(other, int):
            # Special behavior: compare length to integer
            return len(self) == other
        # Standard behavior: compare list contents
        return super().__eq__(other)
