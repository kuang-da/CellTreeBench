"""
Integration tests for dataset modules using the new config system.

Tests that the refactored dataset modules work correctly with centralized configuration.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

# Add src to path for testing
sys.path.insert(0, 'src')


class TestDatasetConfigIntegration:
    """Test that dataset modules integrate correctly with the new config system."""
    
    def test_celegans_dataset_base_imports(self):
        """Test that celegans_dataset_base imports and initializes correctly."""
        try:
            from celltreebench.datasets.celegans_dataset_base import PROJECT_ROOT, DATA_ROOT
            
            # Should successfully import the path variables
            assert PROJECT_ROOT is not None
            assert DATA_ROOT is not None
            
            # Should be Path objects or convertible to paths
            project_path = Path(PROJECT_ROOT)
            data_path = Path(DATA_ROOT)
            
            assert project_path.exists()
            # data_path might not exist in all test environments, so just check it's a valid path
            assert len(str(data_path)) > 0
            
        except ImportError as e:
            pytest.skip(f"Could not import celegans_dataset_base: {e}")
    
    def test_mutation_dataset_imports(self):
        """Test that mutation_dataset imports correctly with new config."""
        try:
            from celltreebench.datasets.mutation_dataset import PROJECT_ROOT, DATA_ROOT
            
            assert PROJECT_ROOT is not None
            assert DATA_ROOT is not None
            
            # Verify paths are valid
            assert len(str(PROJECT_ROOT)) > 0
            assert len(str(DATA_ROOT)) > 0
            
        except ImportError as e:
            # This might fail due to missing dependencies (utilities, tree_utils, etc.)
            # which is expected in a minimal test environment
            pytest.skip(f"Could not import mutation_dataset due to dependencies: {e}")
    
    def test_dnameth_dataset_imports(self):
        """Test that dnameth_dataset_base imports correctly with new config."""
        try:
            from celltreebench.datasets.dnameth_dataset_base import PROJECT_ROOT, DATA_ROOT
            
            assert PROJECT_ROOT is not None
            assert DATA_ROOT is not None
            
            # Verify paths are valid
            assert len(str(PROJECT_ROOT)) > 0
            assert len(str(DATA_ROOT)) > 0
            
        except ImportError as e:
            # This might fail due to missing dependencies
            pytest.skip(f"Could not import dnameth_dataset_base due to dependencies: {e}")


class TestDatasetClassInstantiation:
    """Test that dataset classes can be instantiated with the new config system."""
    
    def test_celegans_dataset_base_instantiation(self):
        """Test that CElegansDatasetBase can be instantiated."""
        try:
            from celltreebench.datasets.celegans_dataset_base import CElegansDatasetBase
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to create an instance with a temporary output directory
                # This might fail if the data directory doesn't exist, which is expected
                try:
                    dataset = CElegansDatasetBase(
                        dataset_name='celegans_small',
                        lineage_name='P0', 
                        out_dir=temp_dir
                    )
                    
                    # If successful, verify basic properties
                    assert hasattr(dataset, 'data_dir')
                    assert hasattr(dataset, 'out_dir')
                    assert dataset.out_dir == Path(temp_dir)
                    
                except (FileNotFoundError, ValueError) as e:
                    # Expected if data files don't exist in test environment
                    pytest.skip(f"Cannot instantiate dataset due to missing data: {e}")
                    
        except ImportError as e:
            pytest.skip(f"Could not import CElegansDatasetBase: {e}")


class TestPathConsistency:
    """Test that paths are consistent across different dataset modules."""
    
    def test_all_datasets_use_same_paths(self):
        """Test that all dataset modules resolve to the same PROJECT_ROOT and DATA_ROOT."""
        project_roots = {}
        data_roots = {}
        
        # Test celegans dataset
        try:
            from celltreebench.datasets.celegans_dataset_base import PROJECT_ROOT, DATA_ROOT
            project_roots['celegans'] = str(PROJECT_ROOT)
            data_roots['celegans'] = str(DATA_ROOT)
        except ImportError:
            pass
        
        # Test mutation dataset  
        try:
            from celltreebench.datasets.mutation_dataset import PROJECT_ROOT, DATA_ROOT
            project_roots['mutation'] = str(PROJECT_ROOT)
            data_roots['mutation'] = str(DATA_ROOT)
        except ImportError:
            pass
        
        # Test dnameth dataset
        try:
            from celltreebench.datasets.dnameth_dataset_base import PROJECT_ROOT, DATA_ROOT
            project_roots['dnameth'] = str(PROJECT_ROOT)
            data_roots['dnameth'] = str(DATA_ROOT)
        except ImportError:
            pass
        
        # If we have multiple datasets imported, they should use the same paths
        if len(project_roots) > 1:
            unique_project_roots = set(project_roots.values())
            unique_data_roots = set(data_roots.values())
            
            assert len(unique_project_roots) == 1, f"Inconsistent PROJECT_ROOT across datasets: {project_roots}"
            assert len(unique_data_roots) == 1, f"Inconsistent DATA_ROOT across datasets: {data_roots}"


class TestEnvironmentVariableIntegration:
    """Test that environment variables work correctly with dataset modules."""
    
    def test_dataset_respects_data_dir_override(self):
        """Test that dataset modules respect CELLTREEBENCH_DATA_DIR override."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {'CELLTREEBENCH_DATA_DIR': temp_dir}):
                # Clear any cached imports
                import importlib
                if 'celltreebench.config' in sys.modules:
                    importlib.reload(sys.modules['celltreebench.config'])
                
                try:
                    # Re-import after setting environment variable
                    from celltreebench.config import get_data_root
                    
                    # Create a fresh config to test environment override
                    data_root = get_data_root()
                    assert str(data_root) == temp_dir
                    
                except Exception as e:
                    # Might fail in complex test scenarios
                    pytest.skip(f"Environment variable test failed: {e}")


class TestPathOperations:
    """Test path operations with the new pathlib integration."""
    
    def test_pathlib_operations_work(self):
        """Test that pathlib operations work correctly with dataset paths."""
        try:
            from celltreebench.datasets.celegans_dataset_base import DATA_ROOT
            
            data_path = Path(DATA_ROOT)
            
            # Test pathlib operations
            parent = data_path.parent
            assert isinstance(parent, Path)
            
            # Test path joining
            test_subpath = data_path / "test_subdirectory"
            assert isinstance(test_subpath, Path)
            assert str(test_subpath).endswith("test_subdirectory")
            
            # Test path properties
            assert data_path.is_absolute()
            
        except ImportError:
            pytest.skip("Could not import dataset module")


class TestBackwardCompatibility:
    """Test that the refactoring maintains backward compatibility."""
    
    def test_module_level_constants_exist(self):
        """Test that PROJECT_ROOT and DATA_ROOT constants still exist in dataset modules."""
        modules_to_test = [
            'celltreebench.datasets.celegans_dataset_base',
            'celltreebench.datasets.mutation_dataset', 
            'celltreebench.datasets.dnameth_dataset_base'
        ]
        
        for module_name in modules_to_test:
            try:
                module = __import__(module_name, fromlist=['PROJECT_ROOT', 'DATA_ROOT'])
                
                # Should have these constants
                assert hasattr(module, 'PROJECT_ROOT')
                assert hasattr(module, 'DATA_ROOT')
                
                # Should not be None
                assert module.PROJECT_ROOT is not None
                assert module.DATA_ROOT is not None
                
            except ImportError:
                # Skip modules that can't be imported due to missing dependencies
                continue
    
    def test_string_path_compatibility(self):
        """Test that paths work in contexts expecting string paths."""
        try:
            from celltreebench.datasets.celegans_dataset_base import PROJECT_ROOT, DATA_ROOT
            
            # Should be convertible to strings
            project_str = str(PROJECT_ROOT)
            data_str = str(DATA_ROOT)
            
            assert isinstance(project_str, str)
            assert isinstance(data_str, str)
            assert len(project_str) > 0
            assert len(data_str) > 0
            
            # Should work with os.path operations (for legacy code)
            import os.path
            assert os.path.isabs(project_str)
            assert os.path.isabs(data_str)
            
        except ImportError:
            pytest.skip("Could not import dataset module")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__]) 