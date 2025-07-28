# CellTreeBench Test Suite

This directory contains unit and integration tests for the CellTreeBench package, with a focus on the refactored configuration system.

## Test Structure

- **`test_config.py`**: Unit tests for the `celltreebench.config` module
  - Tests the robust path discovery mechanisms
  - Tests environment variable overrides
  - Tests fallback strategies
  - Tests error handling

- **`test_datasets_integration.py`**: Integration tests for dataset modules
  - Tests that refactored dataset modules work correctly
  - Tests backward compatibility
  - Tests path consistency across modules
  - Tests pathlib integration

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-mock
# Or from the test requirements:
pip install -r tests/requirements.txt
```

### Run All Tests

```bash
# From the project root
cd tests
python -m pytest

# Or from the project root
python -m pytest tests/
```

### Run Specific Test Files

```bash
# Config tests only
python -m pytest test_config.py

# Integration tests only  
python -m pytest test_datasets_integration.py
```

### Run with Verbose Output

```bash
python -m pytest -v
```

### Run Tests with Coverage (if pytest-cov is installed)

```bash
pip install pytest-cov
python -m pytest --cov=celltreebench --cov-report=html
```

## Test Categories

Tests are organized by functionality:

### Unit Tests (`test_config.py`)

- **TestCellTreeBenchConfig**: Basic config class functionality
- **TestEnvironmentVariableOverrides**: Environment variable configuration
- **TestFallbackMechanisms**: Path discovery fallback strategies
- **TestConvenienceFunctions**: Global convenience functions
- **TestPathResolution**: Path resolution and consistency
- **TestIntegrationWithExistingCode**: Integration with existing codebase

### Integration Tests (`test_datasets_integration.py`)

- **TestDatasetConfigIntegration**: Dataset module imports and initialization
- **TestDatasetClassInstantiation**: Actual dataset class creation
- **TestPathConsistency**: Consistency across different dataset modules
- **TestEnvironmentVariableIntegration**: Environment variable effects
- **TestPathOperations**: Pathlib operations with dataset paths
- **TestBackwardCompatibility**: Ensures refactoring doesn't break existing code

## Expected Results

### Successful Test Run

```
================================ 28 passed, 2 skipped, X warnings ================================
```

- **28 passed**: All core functionality tests pass
- **2 skipped**: Some dataset modules may be skipped due to missing dependencies (expected)
- **X warnings**: Deprecation warnings from external libraries (expected)

### Common Issues

1. **Missing Dependencies**: Some dataset modules require dependencies that may not be installed in test environments. These tests will be gracefully skipped.

2. **Missing Data Files**: Dataset instantiation tests may fail if data files don't exist. These are handled gracefully with appropriate skip messages.

3. **Environment Variables**: Tests that modify environment variables are isolated and should not affect other tests.

## Testing the Refactored Configuration

The test suite specifically validates:

✅ **Path Discovery**: Multiple fallback strategies work correctly  
✅ **Environment Overrides**: `CELLTREEBENCH_ROOT` and `CELLTREEBENCH_DATA_DIR` work  
✅ **Backward Compatibility**: Existing code continues to work  
✅ **Pathlib Integration**: Modern path operations work correctly  
✅ **Error Handling**: Graceful failure with helpful error messages  
✅ **Cross-Module Consistency**: All dataset modules use the same paths  

## Adding New Tests

When adding new functionality:

1. Add unit tests to `test_config.py` for core config functionality
2. Add integration tests to `test_datasets_integration.py` for dataset-related features
3. Use appropriate pytest fixtures and mocking for isolation
4. Include both positive and negative test cases
5. Test environment variable overrides where applicable

## Continuous Integration

These tests are designed to run in various environments:
- Development environments
- CI/CD pipelines  
- Docker containers
- Different Python versions

The test suite uses robust mocking and fallback strategies to work even when data files or dependencies are missing. 