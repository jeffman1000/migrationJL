"""
Schema components for Dataiku datasets.

This module provides type-safe representations of dataset schemas,
including data types and columns.
"""

from __future__ import annotations

from dataclasses import dataclass


class DataType:
    """
    Represents a column data type in a dataset schema.

    Dataiku supports various data types (string, bigint, double, etc.).
    This class provides a type-safe way to represent these types.
    """

    def __init__(self, name: str):
        """
        Initialize a DataType.

        Args:
            name: The name of the data type (e.g., "string", "bigint", "double")
        """
        self.name = name

    def __repr__(self) -> str:
        """Return a string representation in the format datatype.{name}."""
        return f"datatype.{self.name}"

    def __eq__(self, other: object) -> bool:
        """Compare data types by name."""
        if isinstance(other, DataType):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        """Enable data types to be used in sets and as dict keys."""
        return hash(self.name)


class DataTypeRegistry:
    """
    Registry of available data types.

    This registry allows accessing data types via attribute syntax:
    datatype.string, datatype.bigint, etc.

    Data types are created on-demand when first accessed and cached
    for subsequent use.
    """

    def __init__(self) -> None:
        """Initialize the data type registry with an empty cache."""
        # Cache for created data types
        self._types: dict[str, DataType] = {}

    def __getattr__(self, name: str) -> DataType:
        """
        Get or create a data type by name.

        Args:
            name: Name of the data type (e.g., "string", "bigint")

        Returns:
            DataType object

        Example:
            >>> datatype.string
            datatype.string
            >>> datatype.bigint
            datatype.bigint
        """
        # Don't allow access to private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Create and cache data type on first access
        if name not in self._types:
            self._types[name] = DataType(name)
        return self._types[name]


# Global datatype registry instance
# Use this to access data types: datatype.string, datatype.bigint, etc.
datatype = DataTypeRegistry()


@dataclass
class Column:
    """
    Represents a column in a dataset schema.

    Columns have a name and a data type. This class provides sorting
    and comparison functionality for working with dataset schemas.

    Attributes:
        name: Column name
        type: Column data type (DataType instance)
    """

    name: str
    type: DataType

    def __lt__(self, other: object) -> bool:
        """Enable sorting of columns by name."""
        if not isinstance(other, Column):
            return NotImplemented
        return self.name < other.name

    def __eq__(self, other: object) -> bool:
        """Compare columns by name and type."""
        if not isinstance(other, Column):
            return False
        return self.name == other.name and self.type == other.type

    def __hash__(self) -> int:
        """Enable columns to be used in sets and as dict keys."""
        return hash((self.name, self.type))
