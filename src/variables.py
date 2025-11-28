"""
Variables module for Dataiku Bundle SDK.

This module provides classes for accessing project-level and local variables
from Dataiku bundles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Variables:
    """
    Represents variables in a Dataiku bundle.

    Variables in Dataiku can be defined at two levels:
    - Global variables (project-level): Stored in variables.json
    - Local variables: Stored in localvariables.json

    These variables can be used throughout the project in recipes, scenarios, etc.
    """

    def __init__(self, bundle_path: Path):
        """
        Initialize Variables from a bundle path.

        Args:
            bundle_path: Path to the root of the Dataiku bundle
        """
        self._bundle_path = bundle_path
        self._global_vars: dict[str, Any] | None = None
        self._local_vars: dict[str, Any] | None = None

    @property
    def project(self) -> dict[str, Any]:
        """
        Get project-level (global) variables.

        Returns:
            Dictionary of global variables from variables.json
        """
        if self._global_vars is None:
            variables_file = self._bundle_path / "project_config" / "variables.json"
            if variables_file.exists():
                with open(variables_file) as f:
                    self._global_vars = json.load(f)
            else:
                self._global_vars = {}
        return self._global_vars

    @property
    def local(self) -> dict[str, Any]:
        """
        Get local variables.

        Returns:
            Dictionary of local variables from localvariables.json
        """
        if self._local_vars is None:
            local_variables_file = (
                self._bundle_path / "project_config" / "localvariables.json"
            )
            if local_variables_file.exists():
                with open(local_variables_file) as f:
                    self._local_vars = json.load(f)
            else:
                self._local_vars = {}
        return self._local_vars
