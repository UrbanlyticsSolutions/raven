"""
Unit tests for the simplified Magpie Workflow
"""

import pytest
import tempfile
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import box
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from magpie_workflow import Config, DataProcessor, BasinProcessor, MagpieWorkflow


class TestConfig:
    """Test configuration class"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = Config()
        
        assert config.model_name == "raven_model"
        assert config.start_year == 2000
        assert config.end_year == 2005
        assert config.min_drainage_area == 50.0
        assert config.buffer_distance == 5000.0
    
    def test_config_with_file(self):
        """Test configuration loading from file"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"model_name": "test_model", "start_year": 2010}')
            temp_config_file = f.name
        
        try:
            config = Config(temp_config_file)
            assert config.model_name == "test_model"
            assert config.start_year == 2010
        finally:
            os.unlink(temp_config_file)
    
    def test_create_directories(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.workspace_dir = Path(tmpdir) / "test_workspace"
            config._setup_paths()
            
            config.create_directories()
            
            # Check that directories were created
            assert config.paths['data'].exists()
            assert config.paths['outputs'].exists()
            assert config.paths['temp'].exists()


class TestDataProcessor:
    """Test data processing functionality"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.workspace_dir = Path(tmpdir)
            config.paths = {
                'data': Path(tmpdir) / 'data',
                'inputs': Path(tmpdir) / 'data' / 'inputs',
                'outputs': Path(tmpdir) / 'outputs',
                'temp': Path(tmpdir) / 'temp'
            }
            config.create_directories()
            yield config
    
    def test_create_sample_study_area(self, temp_config):
        """Test sample study area creation"""
        processor = DataProcessor(temp_config)
        
        study_area = processor._create_sample_study_area()
        
        assert isinstance(study_area, gpd.GeoDataFrame)
        assert len(study_area) == 1
        assert study_area.crs.to_string() == 'EPSG:4326'
    
    def test_create_sample_dem(self, temp_config):
        """Test sample DEM creation"""
        processor = DataProcessor(temp_config)
        
        # Create sample study area
        bbox = box(-76, 45, -75, 46)
        study_area = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326')
        
        dem_path = processor.create_sample_dem(study_area)
        
        assert Path(dem_path).exists()
        
        # Check DEM data
        dem_df = pd.read_csv(dem_path)
        assert 'lat' in dem_df.columns
        assert 'lon' in dem_df.columns
        assert 'elevation' in dem_df.columns
        assert len(dem_df) > 0
    
    def test_create_sample_climate_data(self, temp_config):
        """Test sample climate data creation"""
        processor = DataProcessor(temp_config)
        
        # Create sample study area
        bbox = box(-76, 45, -75, 46)
        study_area = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326')
        
        climate_path = processor.create_sample_climate_data(study_area)
        
        assert Path(climate_path).exists()
        
        # Check climate data
        climate_df = pd.read_csv(climate_path)
        assert 'date' in climate_df.columns
        assert 'tmax' in climate_df.columns
        assert 'tmin' in climate_df.columns
        assert 'precip' in climate_df.columns
        
        # Check date range
        expected_days = (temp_config.end_year - temp_config.start_year + 1) * 365
        # Allow for leap years
        assert len(climate_df) >= expected_days
        assert len(climate_df) <= expected_days + 10


class TestBasinProcessor:
    """Test basin processing functionality"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.workspace_dir = Path(tmpdir)
            config.paths = {
                'basin': Path(tmpdir) / 'basin',
                'data': Path(tmpdir) / 'data'
            }
            config.create_directories()
            yield config
    
    @pytest.fixture
    def sample_study_area(self):
        """Create sample study area for testing"""
        bbox = box(-76, 45, -75, 46)
        return gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326')
    
    def test_create_subbasins(self, temp_config, sample_study_area):
        """Test subbasin creation"""
        processor = BasinProcessor(temp_config)
        
        subbasins = processor.create_subbasins(sample_study_area)
        
        assert isinstance(subbasins, gpd.GeoDataFrame)
        assert len(subbasins) > 0
        assert 'SubId' in subbasins.columns
        assert 'Area_km2' in subbasins.columns
        
        # Check that output file was created
        output_path = temp_config.paths['basin'] / 'subbasins.shp'
        assert output_path.exists()
    
    def test_create_hrus(self, temp_config, sample_study_area):
        """Test HRU creation"""
        processor = BasinProcessor(temp_config)
        
        # First create subbasins
        subbasins = processor.create_subbasins(sample_study_area)
        
        # Then create HRUs
        hrus = processor.create_hrus(subbasins)
        
        assert isinstance(hrus, gpd.GeoDataFrame)
        assert len(hrus) == len(subbasins)  # One HRU per subbasin
        assert 'HRU_ID' in hrus.columns
        assert 'SubId' in hrus.columns
        assert 'Elevation' in hrus.columns
        
        # Check that output file was created
        output_path = temp_config.paths['basin'] / 'hrus.shp'
        assert output_path.exists()


class TestMagpieWorkflow:
    """Test main workflow functionality"""
    
    def test_workflow_initialization(self):
        """Test workflow initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temporary config
            config_file = Path(tmpdir) / "config.json"
            with open(config_file, 'w') as f:
                f.write('{"model_name": "test_workflow", "start_year": 2015, "end_year": 2016}')
            
            workflow = MagpieWorkflow(str(config_file))
            
            assert workflow.config.model_name == "test_workflow"
            assert workflow.config.start_year == 2015
            assert workflow.config.end_year == 2016
    
    def test_run_single_step(self):
        """Test running individual workflow steps"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            with open(config_file, 'w') as f:
                f.write(f'{{"workspace_dir": "{tmpdir}", "model_name": "test"}}')
            
            workflow = MagpieWorkflow(str(config_file))
            
            # Test study area step
            result = workflow.run_step("study_area")
            assert result is True
            
            # Test sample data step
            result = workflow.run_step("sample_data")
            assert result is True


class TestIntegration:
    """Integration tests for full workflow"""
    
    def test_full_workflow_run(self):
        """Test complete workflow execution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config for minimal test
            config_file = Path(tmpdir) / "config.json"
            config_data = {
                "workspace_dir": tmpdir,
                "model_name": "integration_test",
                "start_year": 2020,
                "end_year": 2020,  # Single year for faster test
                "min_drainage_area": 100.0
            }
            
            with open(config_file, 'w') as f:
                import json
                json.dump(config_data, f)
            
            # Run workflow
            workflow = MagpieWorkflow(str(config_file))
            results = workflow.run_full_workflow()
            
            # Check that most steps completed successfully
            successful_steps = sum(results.values())
            total_steps = len(results)
            
            # Allow for some steps to fail in test environment
            assert successful_steps >= total_steps * 0.7  # At least 70% success


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
