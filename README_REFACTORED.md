# Magpie Workflow - Refactored Version

A modern, modular implementation of the Magpie hydrological modeling workflow for Raven.

## Features

### ✅ Improved Code Structure
- **Modular Design**: Separated into logical components (data collection, basin processing, model building)
- **Configuration Management**: Centralized configuration with validation and templates
- **Error Handling**: Comprehensive error handling and logging
- **Type Hints**: Full type annotations for better code maintainability

### ✅ Comprehensive Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Validation**: Data integrity and model file validation
- **Mocking**: Synthetic data generation for testing

### ✅ Better Documentation
- **Docstrings**: Comprehensive documentation for all functions and classes
- **Examples**: Working examples and tutorials
- **API Reference**: Clear API documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/UrbanlyticsSolutions/raven.git
cd raven

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

### Basic Usage

```python
from src.core.config import WorkflowConfig
from src.core.workflow import MagpieWorkflow

# Create configuration
config = WorkflowConfig(
    model_name="my_watershed",
    workspace_dir="./workspace",
    study_area_method="coordinates",
    lat=45.0,
    lon=-75.0,
    start_year=2000,
    end_year=2010
)

# Initialize workflow
workflow = MagpieWorkflow(config)

# Run complete workflow
results = workflow.run_complete_workflow()
```

## Project Structure

```
raven/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   └── workflow.py        # Main workflow orchestration
│   ├── data_collection.py     # Geospatial data processing
│   ├── basin_processing.py    # Basin discretization
│   ├── forcing_data.py        # Climate data processing
│   ├── model_builder.py       # Raven model file generation
│   └── visualization.py       # Plotting and visualization
├── tests/
│   └── test_magpie_workflow.py # Comprehensive test suite
├── examples/
│   └── sample_workflow.ipynb   # Example notebook
└── requirements.txt
```

## Key Improvements

### 1. Configuration Management
- **Centralized settings** with validation
- **Template configurations** for different regions
- **Version control** for configurations
- **JSON serialization** for reproducibility

### 2. Modular Architecture
- **Separation of concerns**: Each module has a single responsibility
- **Lazy loading**: Components are loaded only when needed
- **Dependency injection**: Easy to swap implementations
- **Plugin architecture**: Easy to extend functionality

### 3. Error Handling & Logging
- **Structured logging** throughout the workflow
- **Graceful degradation**: Workflow continues when possible
- **Detailed error messages** with context
- **Validation at each step**

### 4. Testing Framework
- **Unit tests** for individual components
- **Integration tests** for complete workflows
- **Mock data generation** for testing without external dependencies
- **Continuous integration** ready

### 5. Data Validation
- **Input validation**: Check data quality and completeness
- **Output validation**: Verify generated model files
- **Consistency checks**: Ensure data consistency across components
- **Quality metrics**: Generate data quality reports

## Usage Examples

### Example 1: Canadian Watershed
```python
config = WorkflowConfig.create_template("canadian_basin")
config.update(
    model_name="ottawa_river",
    lat=45.4215,
    lon=-75.6972,
    start_year=1990,
    end_year=2020
)
```

### Example 2: US Watershed
```python
config = WorkflowConfig.create_template("us_basin")
config.update(
    model_name="potomac_river",
    lat=38.9072,
    lon=-77.0369,
    climate_source="daymet"
)
```

### Example 3: Custom Configuration
```python
config = WorkflowConfig(
    model_name="custom_watershed",
    model_template="HMETS",
    dem_source="aster",
    landcover_source="nalcms",
    climate_source="era5",
    min_drainage_area=25.0,
    use_lakes=True
)
```

## Testing

Run the complete test suite:

```bash
# Run all tests
python tests/test_magpie_workflow.py

# Run specific test classes
python -m unittest tests.test_magpie_workflow.TestWorkflowConfig
python -m unittest tests.test_magpie_workflow.TestIntegration
```

## API Reference

### Core Classes

#### `WorkflowConfig`
Central configuration management for the workflow.

**Key Methods:**
- `save(filepath)`: Save configuration to JSON
- `load(filepath)`: Load configuration from JSON
- `validate()`: Validate configuration parameters
- `get_paths()`: Get all relevant file paths

#### `MagpieWorkflow`
Main workflow orchestration class.

**Key Methods:**
- `run_complete_workflow()`: Execute full workflow
- `run_setup()`: Initialize workspace
- `run_study_area_processing()`: Process study area
- `run_data_collection()`: Collect geospatial data
- `run_basin_discretization()`: Create subbasins and HRUs
- `run_model_setup()`: Generate Raven input files

#### `DataCollector`
Handles geospatial data collection and processing.

**Key Methods:**
- `collect_dem()`: Download DEM data
- `collect_landcover()`: Download landcover data
- `collect_soil_data()`: Download soil data
- `validate_data_integrity()`: Validate collected data

#### `BasinProcessor`
Basin discretization and HRU creation.

**Key Methods:**
- `create_subbasins()`: Generate subbasins
- `create_hrus()`: Create hydrologic response units
- `validate_basin_structure()`: Validate basin connectivity

#### `RavenModelBuilder`
Generate Raven model input files.

**Key Methods:**
- `create_rvi_file()`: Generate RVI file
- `create_rvh_file()`: Generate RVH file
- `create_rvp_file()`: Generate RVP file
- `create_rvc_file()`: Generate RVC file

## Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Ensure** all tests pass
5. **Submit** a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support:
- Email: hburdett@uwaterloo.ca
- GitHub Issues: [Create an issue](https://github.com/UrbanlyticsSolutions/raven/issues)

## Acknowledgments

- Original Magpie Workflow developers
- Raven hydrological modeling framework
- BasinMaker watershed discretization tool
- DayMet climate data service
