"""
Dataiku Bundle SDK - Programmatically explore and interact with Dataiku bundles.

This package provides a Python SDK for working with Dataiku DSS project bundles.
"""

# Import and re-export all public API from dataiku_bundle
from .dataiku_bundle import (
    Column,
    DataikuBundle,
    Dataset,
    DatasetModes,
    Predict,
    Recipe,
    RecipeTypes,
    Scenario,
    SnowflakeDataset,
    Step,
    StepsList,
    StepTypes,
    datatype,
    get_datasets_list,
    get_recipe_list,
    get_scenario,
    get_scenario_list,
    get_scenario_steps,
)

# Import flow compression utilities
from .flow_compression import can_compress_recipe
from .vertical_section import find_vertical_section

__all__ = [
    # Main class
    "DataikuBundle",
    # Enums
    "StepTypes",
    "RecipeTypes",
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
    # Data types
    "datatype",
    # Functions
    "get_scenario_list",
    "get_scenario",
    "get_scenario_steps",
    "get_datasets_list",
    "get_recipe_list",
    # Flow compression functions
    "can_compress_recipe",
    "find_vertical_section",
]
