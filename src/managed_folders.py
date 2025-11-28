"""
Managed folder management for Dataiku bundles.

This module provides functionality for working with managed folders,
which are file storage locations in Dataiku projects.
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
class ManagedFolder:
    """
    Represents a managed folder in a Dataiku bundle.

    Managed folders are file storage locations in Dataiku that can store
    files on various backends (GCS, S3, local filesystem, etc.).

    This class uses lazy loading: metadata is only loaded from JSON files when
    properties are first accessed.

    Attributes:
        id: Managed folder identifier (the internal ID like "48kCywFK")
        name: Human-readable name of the managed folder
        bundle: DataikuBundle instance that this managed folder belongs to
        _bundle_path: Path to bundle root (derived from bundle parameter)
        _data_loaded: Flag to prevent infinite loops during lazy loading
        _type: Storage type (e.g., "GCS", "S3", "Filesystem")
        _connection: Connection name used for storage
        _bucket: Storage bucket name (for cloud storage)
        _path: Path within the bucket/storage
        _files: List of files in the managed folder (if exported with bundle)
    """

    id: str
    name: str | None = field(default=None, repr=True)
    bundle: DataikuBundle | str | Path | None = field(default=None, repr=False)
    _bundle_path: Path | None = field(default=None, repr=False)
    _data_loaded: bool = field(default=False, repr=False)
    _type: str | None = field(default=None, repr=False)
    _connection: str | None = field(default=None, repr=False)
    _bucket: str | None = field(default=None, repr=False)
    _path: str | None = field(default=None, repr=False)
    _files: list[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize bundle path from bundle parameter."""
        if self.bundle:
            if isinstance(self.bundle, (str, Path)):
                self._bundle_path = Path(self.bundle)
            else:
                self._bundle_path = self.bundle.path

    def _load_data(self) -> None:
        """
        Lazy load managed folder metadata from JSON file.

        This method reads the managed folder's JSON configuration file
        and extracts key properties like type, connection, bucket, and path.
        """
        if self._data_loaded or not self._bundle_path:
            return

        # Find the managed folder JSON file (case-insensitive)
        managed_folders_path = self._bundle_path / "project_config" / "managed_folders"
        json_file = find_file_case_insensitive(managed_folders_path, f"{self.id}.json")

        if json_file and json_file.exists():
            with open(json_file) as f:
                data = json.load(f)

            # Extract basic properties
            self.name = data.get("name")
            self._type = data.get("type")

            # Extract params
            params = data.get("params", {})
            self._connection = params.get("connection")
            self._bucket = params.get("bucket")
            self._path = params.get("path")

            # Check if files were exported with the bundle
            managed_folder_dir = self._bundle_path / "managed_folders" / self.id
            if managed_folder_dir.exists():
                self._files = [
                    f.name for f in managed_folder_dir.iterdir() if f.is_file()
                ]
            else:
                self._files = []

        self._data_loaded = True

    @property
    def type(self) -> str | None:
        """
        Get the storage type of the managed folder.

        Returns:
            Storage type (e.g., "GCS", "S3", "Filesystem") or None if not found
        """
        self._load_data()
        return self._type

    @property
    def storage_bucket(self) -> str | None:
        """
        Get the storage bucket name (for cloud storage types).

        Returns:
            Bucket name or None if not applicable
        """
        self._load_data()
        return self._connection

    @property
    def bucket(self) -> str | None:
        """
        Get the actual bucket name from params.

        Returns:
            Bucket name or None if not applicable
        """
        self._load_data()
        return self._bucket

    @property
    def path(self) -> str | None:
        """
        Get the fully resolved path for the managed folder.

        This combines the bucket name with the path template (with variables expanded)
        to create the complete storage path.

        Returns:
            Full path string with bucket and expanded variables, or None if not found
        """
        self._load_data()
        if not self._path or not self._bucket:
            return None

        # Expand variables in the path template
        expanded_path = self._path
        if self.bundle:
            expanded_path = expanded_path.replace(
                "${projectKey}", self.bundle.project_key
            )
        expanded_path = expanded_path.replace("${odbId}", self.id)

        # Remove leading slash if present
        expanded_path = expanded_path.lstrip("/")

        # Construct full path with bucket and trailing slash
        full_path = f"{self._bucket}/{expanded_path}/"

        return full_path

    @property
    def files(self) -> list[str]:
        """
        Get the list of files in the managed folder (if exported).

        Returns:
            List of filenames
        """
        self._load_data()
        return self._files or []

    @property
    def length(self) -> int:
        """
        Get the number of files in the managed folder.

        Returns:
            Number of files
        """
        return len(self.files)


def get_managed_folders_list(bundle: DataikuBundle) -> list[ManagedFolder]:
    """
    Get all managed folders from a Dataiku bundle.

    Args:
        bundle: DataikuBundle instance to extract managed folders from

    Returns:
        List of ManagedFolder objects

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> folders = get_managed_folders_list(bundle)
        >>> for folder in folders:
        ...     print(f"{folder.name}: {folder.type}")
    """
    managed_folders_path = bundle.path / "project_config" / "managed_folders"

    if not managed_folders_path.exists():
        return []

    managed_folders = []
    for json_file in managed_folders_path.glob("*.json"):
        folder_id = json_file.stem
        managed_folder = ManagedFolder(id=folder_id, bundle=bundle)
        # Eagerly load data so that name is available for consumers
        managed_folder._load_data()
        managed_folders.append(managed_folder)

    return managed_folders


def get_managed_folder(bundle: DataikuBundle, name: str) -> ManagedFolder | None:
    """
    Get a specific managed folder by name.

    Args:
        bundle: DataikuBundle instance to search in
        name: Name of the managed folder to find

    Returns:
        ManagedFolder object if found, None otherwise

    Example:
        >>> bundle = DataikuBundle("dss-bundle-MYPROJECT-2025-10-15")
        >>> folder = get_managed_folder(bundle, "model-store")
        >>> if folder:
        ...     print(f"Bucket: {folder.bucket}")
    """
    folders = get_managed_folders_list(bundle)

    for folder in folders:
        # Force lazy loading to get the name
        folder._load_data()
        if folder.name == name:
            return folder

    return None
