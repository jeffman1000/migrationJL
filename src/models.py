"""
Machine learning models for Dataiku bundles.

This module handles prediction models (saved models) that are created
by prediction_training recipes in Dataiku.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import find_file_case_insensitive

if TYPE_CHECKING:
    from .dataiku_bundle import DataikuBundle


@dataclass
class Predict:
    """
    Represents a Dataiku prediction model (saved model).

    In Dataiku, prediction models are created by prediction_training recipes
    and can be used to score datasets. This class handles model identification
    and name resolution from bundle metadata.

    Attributes:
        id: Model identifier (file ID or human-readable name)
        bundle: DataikuBundle instance that this model belongs to
        _bundle_path: Path to the bundle root (derived from bundle parameter)
        _resolved_name: Cached human-readable model name from JSON metadata
    """

    id: str
    bundle: "DataikuBundle | None" = field(default=None, repr=False)
    _bundle_path: Path | None = field(default=None, repr=False)
    _resolved_name: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to set bundle path and resolve model name.
        """
        # Set bundle path from the bundle object if provided
        if self.bundle is not None and self._bundle_path is None:
            self._bundle_path = self.bundle.path

        # Load the human-readable model name from the JSON file
        self._resolve_model_name()

    def _resolve_model_name(self) -> None:
        """Resolve the human-readable model name from the saved model file."""
        if self._bundle_path is None:
            # Can't resolve without bundle path
            self._resolved_name = self.id.lower()
            self.id = self._resolved_name
            return

        saved_models_dir = self._bundle_path / "project_config" / "saved_models"

        # First, try to find by the ID as a filename (in case it's the file ID like hU9KQFd4)
        model_file = find_file_case_insensitive(saved_models_dir, f"{self.id}.json")

        if model_file is not None:
            # Found the file, read the name
            with open(model_file) as f:
                model_data = json.load(f)
            model_name = model_data.get("name", "")
            if model_name:
                self._resolved_name = model_name.lower()
                self.id = self._resolved_name
            else:
                self._resolved_name = self.id.lower()
                self.id = self._resolved_name
        else:
            # The ID might already be a name, search all model files to find a match
            if saved_models_dir.exists():
                id_lower = self.id.lower()
                for model_file in saved_models_dir.glob("*.json"):
                    with open(model_file) as f:
                        model_data = json.load(f)
                    model_name = model_data.get("name", "")
                    if model_name.lower() == id_lower:
                        self._resolved_name = model_name.lower()
                        self.id = self._resolved_name
                        return

                # No match found, just normalize to lowercase
                self._resolved_name = self.id.lower()
                self.id = self._resolved_name

    def __eq__(self, other: object) -> bool:
        """Compare prediction models based on id only."""
        if not isinstance(other, Predict):
            return False
        return self.id == other.id

    def __lt__(self, other: object) -> bool:
        """Enable sorting of prediction models by id."""
        if not isinstance(other, Predict):
            return NotImplemented
        return self.id < other.id

    def __hash__(self) -> int:
        """Enable prediction models to be used in sets and as dict keys."""
        return hash(self.id)
