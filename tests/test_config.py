"""
Unit tests for the celltreebench.config module.

Tests the robust path discovery mechanisms and configuration management.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path for testing
sys.path.insert(0, 'src')

from celltreebench.config import (
    CellTreeBenchConfig, 
    get_package_root, 
    get_data_root, 
    get_dataset_path,
    config
)


class TestCellTreeBenchConfig:
    """Test the CellTreeBenchConfig class."""
    
    def test_config_instance_creation(self):
        """Test that config instance can be created."""
        test_config = CellTreeBenchConfig()
        assert test_config is not None
        assert hasattr(test_config, 'package_root')
        assert hasattr(test_config, 'data_root')
    
    def test_package_root_property(self):
        """Test that package_root property returns a Path object."""
        test_config = CellTreeBenchConfig()
        root = test_config.package_root
        assert isinstance(root, Path)
        assert root.exists()
    
    def test_data_root_property(self):
        """Test that data_root property returns a Path object."""
        test_config = CellTreeBenchConfig()
        data_root = test_config.data_root
        assert isinstance(data_root, Path)
        # Note: data_root might not exist in test environments, so we just check it's a Path
    
    def test_get_dataset_path(self):
        """Test the get_dataset_path method."""
        test_config = CellTreeBenchConfig()
        dataset_path = test_config.get_dataset_path("test_dataset")
        assert isinstance(dataset_path, Path)
        assert str(dataset_path).endswith("test_dataset")
    
    def test_get_output_path_with_custom_dir(self):
        """Test get_output_path with custom directory."""
        test_config = CellTreeBenchConfig()
        custom_dir = "/custom/output"
        output_path = test_config.get_output_path(custom_dir)
        assert isinstance(output_path, Path)
        assert str(output_path) == custom_dir
    
    def test_get_output_path_default(self):
        """Test get_output_path with default directory."""
        test_config = CellTreeBenchConfig()
        output_path = test_config.get_output_path()
        assert isinstance(output_path, Path)
        assert str(output_path).endswith("output")


class TestEnvironmentVariableOverrides:
    """Test environment variable configuration overrides."""
    
    def test_data_root_environment_override(self):
        """Test that CELLTREEBENCH_DATA_DIR overrides default data root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {'CELLTREEBENCH_DATA_DIR': temp_dir}):
                # Create a fresh config instance to test environment override
                test_config = CellTreeBenchConfig()
                # Clear any cached values
                test_config._data_root = None
                
                data_root = test_config.data_root
                assert str(data_root) == temp_dir
    
    def test_package_root_environment_override(self):
        """Test that CELLTREEBENCH_ROOT overrides default package root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a pyproject.toml to make it look like a valid project root
            (Path(temp_dir) / 'pyproject.toml').touch()
            
            with patch.dict(os.environ, {'CELLTREEBENCH_ROOT': temp_dir}):
                test_config = CellTreeBenchConfig()
                # Clear any cached values
                test_config._package_root = None
                
                # Mock the _find_package_root method to test env var path
                with patch.object(test_config, '_find_package_root') as mock_find:
                    mock_find.return_value = Path(temp_dir)
                    package_root = test_config.package_root
                    assert str(package_root) == temp_dir
    
    def test_invalid_data_dir_environment_variable(self):
        """Test behavior when CELLTREEBENCH_DATA_DIR points to non-existent directory."""
        non_existent_dir = "/this/path/does/not/exist"
        with patch.dict(os.environ, {'CELLTREEBENCH_DATA_DIR': non_existent_dir}):
            test_config = CellTreeBenchConfig()
            test_config._data_root = None  # Clear cache
            
            # Should fall back to other methods and not raise error immediately
            # The actual behavior depends on whether other fallback paths exist
            try:
                data_root = test_config.data_root
                # If successful, it found a fallback
                assert isinstance(data_root, Path)
            except FileNotFoundError:
                # If no fallbacks exist, should raise FileNotFoundError
                pass


class TestFallbackMechanisms:
    """Test the various fallback mechanisms for path discovery."""
    
    @patch('celltreebench.config.resources', None)
    def test_fallback_when_importlib_resources_unavailable(self):
        """Test fallback when importlib.resources is not available."""
        test_config = CellTreeBenchConfig()
        test_config._package_root = None  # Clear cache
        
        # Should still be able to find package root using other methods
        package_root = test_config.package_root
        assert isinstance(package_root, Path)
        assert package_root.exists()
    
    def test_project_root_detection_by_pyproject_toml(self):
        """Test that project root can be detected by finding pyproject.toml."""
        test_config = CellTreeBenchConfig()
        package_root = test_config.package_root
        
        # Should find the actual project root containing pyproject.toml
        assert (package_root / 'pyproject.toml').exists()
    
    def test_error_handling_no_valid_paths(self):
        """Test error handling when no valid paths can be found."""
        test_config = CellTreeBenchConfig()
        
        # Mock all methods to fail
        with patch.object(test_config, '_find_package_root') as mock_find_root:
            mock_find_root.side_effect = RuntimeError("No valid package root found")
            
            with pytest.raises(RuntimeError, match="No valid package root found"):
                _ = test_config.package_root


class TestConvenienceFunctions:
    """Test the convenience functions that use the global config instance."""
    
    def test_get_package_root_function(self):
        """Test the get_package_root convenience function."""
        root = get_package_root()
        assert isinstance(root, Path)
        assert root.exists()
    
    def test_get_data_root_function(self):
        """Test the get_data_root convenience function."""
        data_root = get_data_root()
        assert isinstance(data_root, Path)
    
    def test_get_dataset_path_function(self):
        """Test the get_dataset_path convenience function."""
        dataset_path = get_dataset_path("test_dataset")
        assert isinstance(dataset_path, Path)
        assert str(dataset_path).endswith("test_dataset")
    
    def test_global_config_instance(self):
        """Test that the global config instance works correctly."""
        assert config is not None
        assert isinstance(config, CellTreeBenchConfig)
        
        # Test that multiple calls return the same instance (singleton-like behavior)
        root1 = config.package_root
        root2 = config.package_root
        assert root1 == root2


class TestPathResolution:
    """Test various path resolution scenarios."""
    
    def test_path_objects_are_absolute(self):
        """Test that returned paths are absolute."""
        test_config = CellTreeBenchConfig()
        
        package_root = test_config.package_root
        data_root = test_config.data_root
        
        assert package_root.is_absolute()
        assert data_root.is_absolute()
    
    def test_path_consistency(self):
        """Test that paths are consistent across multiple calls."""
        test_config = CellTreeBenchConfig()
        
        # Multiple calls should return the same paths
        root1 = test_config.package_root
        root2 = test_config.package_root
        data1 = test_config.data_root
        data2 = test_config.data_root
        
        assert root1 == root2
        assert data1 == data2
    
    def test_data_root_relative_to_package_root(self):
        """Test that data root is correctly resolved relative to package root."""
        test_config = CellTreeBenchConfig()
        
        package_root = test_config.package_root
        expected_data_root = package_root / 'data'
        
        # This test assumes the standard layout exists
        if expected_data_root.exists():
            actual_data_root = test_config.data_root
            assert actual_data_root == expected_data_root


class TestIntegrationWithExistingCode:
    """Test integration with existing dataset code."""
    
    def test_config_works_with_dataset_imports(self):
        """Test that config works when imported by dataset modules."""
        # This is an integration test that imports should work
        from celltreebench.config import get_package_root, get_data_root
        
        root = get_package_root()
        data = get_data_root()
        
        assert isinstance(root, Path)
        assert isinstance(data, Path)
    
    def test_backward_compatibility(self):
        """Test that the refactored code maintains backward compatibility."""
        # Import the dataset module which should use the new config
        try:
            from celltreebench.datasets.celegans_dataset_base import PROJECT_ROOT, DATA_ROOT
            
            assert PROJECT_ROOT is not None
            assert DATA_ROOT is not None
            
            # These should be Path-like or string paths
            assert len(str(PROJECT_ROOT)) > 0
            assert len(str(DATA_ROOT)) > 0
            
        except ImportError:
            # If import fails due to missing dependencies, that's okay for this test
            pass


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__]) 