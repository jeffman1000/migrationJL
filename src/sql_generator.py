"""
SQL generation utilities for Dataiku recipes.

This module handles SQL generation for various recipe types,
particularly join recipes which need SQL reconstruction from configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import find_file_case_insensitive

if TYPE_CHECKING:
    from .recipes import Recipe


class JoinSqlBuilder:
    """
    Generates SQL SELECT statements from Dataiku join recipe configurations.

    Join recipes in Dataiku store their configuration in .join files (JSON).
    This class reconstructs the SQL that would be executed by the join.
    """

    def __init__(self, recipe: Recipe, join_file: Path, bundle_path: Path):
        """
        Initialize the SQL builder.

        Args:
            recipe: Recipe instance (used to access input datasets)
            join_file: Path to the .join configuration file
            bundle_path: Path to the bundle root (for loading dataset schemas)
        """
        self.recipe = recipe
        self.join_file = join_file
        self.bundle_path = bundle_path

    def generate_sql(self) -> str:
        """
        Generate SQL SELECT statement from a join recipe configuration.

        Returns:
            SQL SELECT statement representing the join operation
        """
        # Load the join configuration from JSON
        with open(self.join_file) as f:
            join_data = json.load(f)

        # Get the input dataset IDs from the recipe's inputs
        input_tables = [inp.id for inp in self.recipe.inputs]

        # Build mapping from table index to input
        table_to_input, table_selection_mode = self._build_table_mappings(join_data)

        # Build SELECT clause parts
        select_parts = self._build_select_columns(
            join_data, input_tables, table_to_input, table_selection_mode
        )

        # Build the final SQL SELECT statement
        return "SELECT\n  " + ",\n  ".join(select_parts)

    def _build_table_mappings(
        self, join_data: dict
    ) -> tuple[dict[int, int], dict[int, str]]:
        """
        Build mapping from table index to input index using virtualInputs.

        Returns:
            Tuple of (table_to_input mapping, table_selection_mode mapping)
        """
        virtual_inputs = join_data.get("virtualInputs", [])
        table_to_input = {}
        table_selection_mode = {}

        for input_idx, vi in enumerate(virtual_inputs):
            table_idx = vi.get("index")
            if table_idx is not None:
                table_to_input[table_idx] = input_idx
                table_selection_mode[table_idx] = vi.get(
                    "outputColumnsSelectionMode", "MANUAL"
                )

        return table_to_input, table_selection_mode

    def _build_select_columns(
        self,
        join_data: dict,
        input_tables: list[str],
        table_to_input: dict[int, int],
        table_selection_mode: dict[int, str],
    ) -> list[str]:
        """
        Build the list of SELECT clause column references.

        Processes both manually selected columns and auto-selected columns.
        """
        selected_column_names = set()
        select_parts = []

        # Add manually selected columns
        selected_columns = join_data.get("selectedColumns", [])
        for col in selected_columns:
            col_name = col.get("name", "")
            table_index = col.get("table", 0)
            selected_column_names.add(col_name)

            select_parts.append(
                self._format_column_reference(
                    col_name, table_index, input_tables, table_to_input
                )
            )

        # Add AUTO columns from tables with AUTO_NON_CONFLICTING mode
        self._add_auto_columns(
            select_parts,
            selected_column_names,
            table_selection_mode,
            table_to_input,
            input_tables,
        )

        return select_parts

    def _format_column_reference(
        self,
        col_name: str,
        table_index: int,
        input_tables: list[str],
        table_to_input: dict[int, int],
    ) -> str:
        """Format a column reference with table qualification."""
        input_index = table_to_input.get(table_index, table_index)
        if input_index < len(input_tables):
            table_name = input_tables[input_index]
            return f'"{table_name}"."{col_name}" AS "{col_name}"'
        else:
            return f'"{col_name}"'

    def _add_auto_columns(
        self,
        select_parts: list[str],
        selected_column_names: set[str],
        table_selection_mode: dict[int, str],
        table_to_input: dict[int, int],
        input_tables: list[str],
    ) -> None:
        """Add automatically selected columns from tables with AUTO_NON_CONFLICTING mode."""
        for table_idx, mode in table_selection_mode.items():
            if mode == "AUTO_NON_CONFLICTING":
                input_idx = table_to_input.get(table_idx, table_idx)
                if input_idx < len(input_tables):
                    self._add_auto_columns_from_table(
                        select_parts, selected_column_names, input_tables[input_idx]
                    )

    def _add_auto_columns_from_table(
        self, select_parts: list[str], selected_column_names: set[str], dataset_id: str
    ) -> None:
        """Add all non-conflicting columns from a dataset schema."""
        datasets_dir = self.bundle_path / "project_config" / "datasets"
        dataset_file = find_file_case_insensitive(datasets_dir, f"{dataset_id}.json")

        if dataset_file is None:
            return

        with open(dataset_file) as f:
            dataset_data = json.load(f)

        schema_columns = dataset_data.get("schema", {}).get("columns", [])
        for col in schema_columns:
            col_name = col.get("name", "")
            if col_name and col_name not in selected_column_names:
                selected_column_names.add(col_name)
                select_parts.append(f'"{dataset_id}"."{col_name}" AS "{col_name}"')


def generate_sql_from_join(recipe: Recipe, join_file: Path, bundle_path: Path) -> str:
    """
    Generate SQL SELECT statement from a join recipe configuration.

    This is a convenience function that creates a JoinSqlBuilder and generates the SQL.

    Args:
        recipe: Recipe instance (used to access input datasets)
        join_file: Path to the .join configuration file
        bundle_path: Path to the bundle root (for loading dataset schemas)

    Returns:
        SQL SELECT statement representing the join operation
    """
    builder = JoinSqlBuilder(recipe, join_file, bundle_path)
    return builder.generate_sql()
