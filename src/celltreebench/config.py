"""
Configuration module for CellTreeBench package.

This module provides centralized configuration and path management for the package.
It offers multiple strategies to find data directories and package resources.
"""

import os
from pathlib import Path
import logging

# Try to use the modern importlib.resources, fall back to importlib_resources for older Python
try:
    from importlib import resources
except ImportError:
    try:
        import importlib_resources as resources
    except ImportError:
        resources = None

logger = logging.getLogger(__name__)


class CellTreeBenchConfig:
    """Configuration class for CellTreeBench package."""
    
    def __init__(self):
        self._package_root = None
        self._data_root = None
        
    @property
    def package_root(self) -> Path:
        """Get the package root directory."""
        if self._package_root is None:
            self._package_root = self._find_package_root()
        return self._package_root
    
    @property 
    def data_root(self) -> Path:
        """Get the data directory."""
        if self._data_root is None:
            self._data_root = self._find_data_directory()
        return self._data_root
    
    def _find_package_root(self) -> Path:
        """
        Find the package root directory using multiple strategies.
        
        Returns:
            Path: The root directory of the package installation
        """
        # Method 1: Try to find the package root using importlib.resources
        if resources is not None:
            try:
                # Get the path to the celltreebench package
                with resources.path('celltreebench', '__init__.py') as package_path:
                    # Go up one level from src/celltreebench to get to the project root
                    # This works whether installed via pip or in development mode
                    return package_path.parent.parent.parent
            except (FileNotFoundError, AttributeError, ModuleNotFoundError):
                pass
        
        # Method 2: Use __file__ path with pathlib (more robust than the original)
        try:
            # Get the path to this current file
            current_file = Path(__file__).resolve()
            # Navigate up to find the project root by looking for pyproject.toml or setup.py
            current_dir = current_file.parent
            while current_dir != current_dir.parent:  # Stop at filesystem root
                if (current_dir / 'pyproject.toml').exists() or (current_dir / 'setup.py').exists():
                    return current_dir
                current_dir = current_dir.parent
        except NameError:
            pass  # __file__ might not be available in some contexts
        
        # Method 3: Environment variable fallback
        if 'CELLTREEBENCH_ROOT' in os.environ:
            return Path(os.environ['CELLTREEBENCH_ROOT'])
        
        # Method 4: Working directory fallback (least reliable)
        cwd = Path.cwd()
        if (cwd / 'pyproject.toml').exists() and 'celltreebench' in cwd.name.lower():
            return cwd
        
        # Final fallback: use the original method but with pathlib
        try:
            return Path(__file__).resolve().parent.parent.parent
        except NameError:
            raise RuntimeError(
                "Could not determine package root directory. "
                "Please set the CELLTREEBENCH_ROOT environment variable or "
                "ensure the package is properly installed."
            )

    def _find_data_directory(self) -> Path:
        """
        Find the data directory with multiple fallback strategies.
        
        Returns:
            Path: The data directory path
        """
        # Method 1: Environment variable (allows user override)
        if 'CELLTREEBENCH_DATA_DIR' in os.environ:
            data_dir = Path(os.environ['CELLTREEBENCH_DATA_DIR'])
            if data_dir.exists():
                return data_dir
            else:
                logger.warning(f"CELLTREEBENCH_DATA_DIR points to non-existent directory: {data_dir}")
        
        # Method 2: Data directory relative to package root
        try:
            package_root = self.package_root
            data_dir = package_root / 'data'
            if data_dir.exists():
                return data_dir
        except Exception as e:
            logger.warning(f"Could not determine package root: {e}")
        
        # Method 3: Look for data directory in common locations
        search_paths = [
            Path.cwd() / 'data',
            Path.cwd().parent / 'data',
            Path.home() / '.celltreebench' / 'data',
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found data directory at: {path}")
                return path
        
        raise FileNotFoundError(
            "Could not find data directory. Please either:\n"
            "1. Set CELLTREEBENCH_DATA_DIR environment variable\n"
            "2. Ensure 'data' directory exists in the package root\n"
            "3. Run from the project root directory"
        )
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get the path to a specific dataset."""
        return self.data_root / dataset_name
    
    def get_output_path(self, output_dir: str = None) -> Path:
        """Get a standardized output path."""
        if output_dir:
            return Path(output_dir)
        return self.package_root / 'output'


# Global configuration instance
config = CellTreeBenchConfig()

# Convenience functions for backward compatibility
def get_package_root() -> Path:
    """Get the package root directory."""
    return config.package_root

def get_data_root() -> Path:
    """Get the data root directory."""
    return config.data_root

def get_dataset_path(dataset_name: str) -> Path:
    """Get the path to a specific dataset."""
    return config.get_dataset_path(dataset_name) 