"""
Secure file operation utilities with comprehensive validation and error handling.
Replaces direct file I/O operations with validated, explicit error handling.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, BinaryIO, TextIO
import logging
import rasterio
import geopandas as gpd
from .path_manager import AbsolutePathManager, PathResolutionError, FileAccessError

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails"""
    def __init__(self, file_path: str, validation_type: str, reason: str):
        self.file_path = file_path
        self.validation_type = validation_type
        self.reason = reason
        super().__init__(f"File validation failed for '{file_path}' ({validation_type}): {reason}")


class SecureFileOperations:
    """
    Secure file operations with comprehensive validation and error handling.
    All operations use absolute paths and validate file integrity.
    """
    
    def __init__(self, path_manager: AbsolutePathManager):
        """
        Initialize with a path manager for consistent path handling.
        
        Args:
            path_manager: Configured AbsolutePathManager instance
        """
        self.path_manager = path_manager
        
    def safe_read_text(self, file_path: Union[str, Path], encoding: str = 'utf-8', 
                      validate_size: bool = True, max_size_mb: int = 100) -> str:
        """
        Safely read text file with validation.
        
        Args:
            file_path: Path to text file
            encoding: Text encoding (default: utf-8)
            validate_size: Whether to validate file size
            max_size_mb: Maximum file size in MB
            
        Returns:
            File content as string
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be accessed
            FileValidationError: If file validation fails
        """
        abs_path = self.path_manager.resolve_path(file_path)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        if validate_size:
            max_size_bytes = max_size_mb * 1024 * 1024
            self.path_manager.validate_file_integrity(abs_path, min_size=1, max_size=max_size_bytes)
        
        try:
            with open(abs_path, 'r', encoding=encoding) as f:
                content = f.read()
            logger.info(f"Successfully read text file: {abs_path}")
            return content
            
        except (OSError, UnicodeDecodeError) as e:
            raise FileAccessError(str(abs_path), "read", f"Text file read failed: {str(e)}")
    
    def safe_write_text(self, file_path: Union[str, Path], content: str, 
                       encoding: str = 'utf-8', validate_write: bool = True) -> Path:
        """
        Safely write text file with validation.
        
        Args:
            file_path: Path to write file
            content: Text content to write
            encoding: Text encoding (default: utf-8)
            validate_write: Whether to validate written file
            
        Returns:
            Absolute path of written file
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be written
            FileValidationError: If validation fails
        """
        abs_path = self.path_manager.ensure_file_writable(file_path)
        
        try:
            with open(abs_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            if validate_write:
                # Validate the written file
                if not abs_path.exists():
                    raise FileValidationError(str(abs_path), "write_validation", "File not created")
                
                # Quick content validation
                try:
                    with open(abs_path, 'r', encoding=encoding) as f:
                        written_content = f.read()
                    if len(written_content) != len(content):
                        raise FileValidationError(
                            str(abs_path), 
                            "write_validation", 
                            f"Content length mismatch: expected {len(content)}, got {len(written_content)}"
                        )
                except (OSError, UnicodeDecodeError) as e:
                    raise FileValidationError(str(abs_path), "write_validation", f"Cannot read back written file: {e}")
            
            logger.info(f"Successfully wrote text file: {abs_path}")
            return abs_path
            
        except (OSError, UnicodeEncodeError) as e:
            raise FileAccessError(str(abs_path), "write", f"Text file write failed: {str(e)}")
    
    def safe_read_json(self, file_path: Union[str, Path], validate_schema: bool = True) -> Dict[str, Any]:
        """
        Safely read JSON file with validation.
        
        Args:
            file_path: Path to JSON file
            validate_schema: Whether to perform basic JSON validation
            
        Returns:
            Parsed JSON data
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be accessed
            FileValidationError: If JSON validation fails
        """
        content = self.safe_read_text(file_path)
        
        try:
            data = json.loads(content)
            
            if validate_schema:
                # Basic JSON structure validation
                if not isinstance(data, (dict, list)):
                    raise FileValidationError(
                        str(file_path), 
                        "json_validation", 
                        f"Invalid JSON root type: {type(data)}"
                    )
            
            logger.info(f"Successfully read JSON file: {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            raise FileValidationError(str(file_path), "json_parsing", f"Invalid JSON: {str(e)}")
    
    def safe_write_json(self, file_path: Union[str, Path], data: Dict[str, Any], 
                       indent: int = 2, validate_write: bool = True) -> Path:
        """
        Safely write JSON file with validation.
        
        Args:
            file_path: Path to write JSON file
            data: Data to write as JSON
            indent: JSON indentation level
            validate_write: Whether to validate written JSON
            
        Returns:
            Absolute path of written file
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be written
            FileValidationError: If validation fails
        """
        try:
            json_content = json.dumps(data, indent=indent, default=str)
        except (TypeError, ValueError) as e:
            raise FileValidationError(str(file_path), "json_serialization", f"Cannot serialize to JSON: {str(e)}")
        
        abs_path = self.safe_write_text(file_path, json_content, validate_write=validate_write)
        
        if validate_write:
            # Validate by reading back the JSON
            try:
                self.safe_read_json(abs_path)
            except FileValidationError as e:
                raise FileValidationError(str(abs_path), "json_write_validation", f"Written JSON is invalid: {e.reason}")
        
        return abs_path
    
    def safe_read_shapefile(self, file_path: Union[str, Path], validate_geometry: bool = True) -> gpd.GeoDataFrame:
        """
        Safely read shapefile with validation.
        
        Args:
            file_path: Path to shapefile
            validate_geometry: Whether to validate geometries
            
        Returns:
            GeoDataFrame with shapefile data
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be accessed
            FileValidationError: If shapefile validation fails
        """
        abs_path = self.path_manager.resolve_path(file_path)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        # Check for required shapefile components
        shp_path = abs_path.with_suffix('.shp')
        shx_path = abs_path.with_suffix('.shx')
        dbf_path = abs_path.with_suffix('.dbf')
        
        for required_file in [shp_path, shx_path, dbf_path]:
            if not required_file.exists():
                raise FileValidationError(
                    str(abs_path), 
                    "shapefile_validation", 
                    f"Missing required shapefile component: {required_file.name}"
                )
        
        try:
            gdf = gpd.read_file(shp_path)
            
            if validate_geometry:
                # Basic geometry validation
                if len(gdf) == 0:
                    raise FileValidationError(str(abs_path), "shapefile_validation", "Shapefile is empty")
                
                # Check for invalid geometries
                invalid_geoms = (~gdf.geometry.is_valid).sum()
                if invalid_geoms > 0:
                    logger.warning(f"Found {invalid_geoms} invalid geometries in {abs_path}")
                    # Don't raise error, but log warning
                
                # Check CRS
                if gdf.crs is None:
                    logger.warning(f"Shapefile has no CRS defined: {abs_path}")
            
            logger.info(f"Successfully read shapefile: {abs_path} ({len(gdf)} features)")
            return gdf
            
        except Exception as e:
            raise FileAccessError(str(abs_path), "read", f"Shapefile read failed: {str(e)}")
    
    def safe_write_shapefile(self, gdf: gpd.GeoDataFrame, file_path: Union[str, Path], 
                           validate_write: bool = True) -> Path:
        """
        Safely write shapefile with validation.
        
        Args:
            gdf: GeoDataFrame to write
            file_path: Path to write shapefile
            validate_write: Whether to validate written shapefile
            
        Returns:
            Absolute path of written shapefile
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be written
            FileValidationError: If validation fails
        """
        abs_path = self.path_manager.ensure_file_writable(file_path)
        
        # Basic input validation
        if len(gdf) == 0:
            raise FileValidationError(str(file_path), "input_validation", "Cannot write empty GeoDataFrame")
        
        if gdf.crs is None:
            logger.warning(f"Writing shapefile without CRS: {abs_path}")
        
        try:
            # Ensure .shp extension
            shp_path = abs_path.with_suffix('.shp')
            gdf.to_file(shp_path)
            
            if validate_write:
                # Validate by reading back
                try:
                    validation_gdf = self.safe_read_shapefile(shp_path, validate_geometry=False)
                    if len(validation_gdf) != len(gdf):
                        raise FileValidationError(
                            str(shp_path), 
                            "write_validation", 
                            f"Feature count mismatch: expected {len(gdf)}, got {len(validation_gdf)}"
                        )
                except Exception as e:
                    raise FileValidationError(str(shp_path), "write_validation", f"Cannot read back written shapefile: {e}")
            
            logger.info(f"Successfully wrote shapefile: {shp_path} ({len(gdf)} features)")
            return shp_path
            
        except Exception as e:
            raise FileAccessError(str(abs_path), "write", f"Shapefile write failed: {str(e)}")
    
    def safe_read_raster(self, file_path: Union[str, Path], validate_data: bool = True) -> rasterio.io.DatasetReader:
        """
        Safely read raster file with validation.
        
        Args:
            file_path: Path to raster file
            validate_data: Whether to validate raster data
            
        Returns:
            Open rasterio dataset (caller must close)
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be accessed
            FileValidationError: If raster validation fails
        """
        abs_path = self.path_manager.resolve_path(file_path)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        try:
            dataset = rasterio.open(abs_path)
            
            if validate_data:
                # Basic raster validation
                if dataset.width <= 0 or dataset.height <= 0:
                    dataset.close()
                    raise FileValidationError(str(abs_path), "raster_validation", f"Invalid raster dimensions: {dataset.width}x{dataset.height}")
                
                if dataset.crs is None:
                    logger.warning(f"Raster has no CRS defined: {abs_path}")
                
                # Try to read a small sample to validate data
                try:
                    sample_size = min(100, dataset.width, dataset.height)
                    sample = dataset.read(1, window=((0, sample_size), (0, sample_size)))
                    if sample.size == 0:
                        raise FileValidationError(str(abs_path), "raster_validation", "Cannot read raster data")
                except Exception as e:
                    dataset.close()
                    raise FileValidationError(str(abs_path), "raster_validation", f"Cannot read raster data: {e}")
            
            logger.info(f"Successfully opened raster: {abs_path} ({dataset.width}x{dataset.height})")
            return dataset
            
        except rasterio.errors.RasterioIOError as e:
            raise FileAccessError(str(abs_path), "read", f"Raster read failed: {str(e)}")
    
    def safe_copy_file(self, source: Union[str, Path], destination: Union[str, Path], 
                      validate_copy: bool = True) -> Path:
        """
        Safely copy file with validation using enhanced path manager.
        
        Args:
            source: Source file path
            destination: Destination file path  
            validate_copy: Whether to validate copied file
            
        Returns:
            Absolute path of destination file
        """
        return self.path_manager.safe_copy_file(source, destination, 
                                              validate_source=True, 
                                              validate_destination=validate_copy)
    
    def safe_move_file(self, source: Union[str, Path], destination: Union[str, Path], 
                      validate_move: bool = True) -> Path:
        """
        Safely move file with validation.
        
        Args:
            source: Source file path
            destination: Destination file path
            validate_move: Whether to validate moved file
            
        Returns:
            Absolute path of destination file
            
        Raises:
            PathResolutionError: If paths cannot be resolved
            FileAccessError: If move operation fails
        """
        source_abs = self.path_manager.resolve_path(source)
        dest_abs = self.path_manager.ensure_file_writable(destination)
        
        # Validate source exists
        self.path_manager.validate_path(source_abs, must_exist=True, must_be_file=True)
        
        # Get source file size for validation
        source_size = source_abs.stat().st_size
        
        try:
            shutil.move(source_abs, dest_abs)
            
            if validate_move:
                # Validate destination exists and has correct size
                if not dest_abs.exists():
                    raise FileAccessError(str(dest_abs), "move", "Destination file not created")
                
                dest_size = dest_abs.stat().st_size
                if dest_size != source_size:
                    raise FileAccessError(
                        str(dest_abs), 
                        "move", 
                        f"File size mismatch: expected {source_size}, got {dest_size}"
                    )
                
                # Validate source is removed
                if source_abs.exists():
                    raise FileAccessError(str(source_abs), "move", "Source file not removed after move")
            
            logger.info(f"Successfully moved file: {source_abs} -> {dest_abs}")
            return dest_abs
            
        except (OSError, shutil.Error) as e:
            raise FileAccessError(str(dest_abs), "move", f"Move failed: {str(e)}")
    
    def safe_delete_file(self, file_path: Union[str, Path], must_exist: bool = True) -> bool:
        """
        Safely delete file with validation.
        
        Args:
            file_path: Path to file to delete
            must_exist: Whether file must exist (raises error if not)
            
        Returns:
            True if file was deleted, False if didn't exist and must_exist=False
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If delete operation fails
        """
        abs_path = self.path_manager.resolve_path(file_path)
        
        if not abs_path.exists():
            if must_exist:
                raise FileAccessError(str(abs_path), "delete", "File does not exist")
            return False
        
        if not abs_path.is_file():
            raise FileAccessError(str(abs_path), "delete", "Path is not a file")
        
        try:
            abs_path.unlink()
            
            # Validate deletion
            if abs_path.exists():
                raise FileAccessError(str(abs_path), "delete", "File still exists after deletion attempt")
            
            logger.info(f"Successfully deleted file: {abs_path}")
            return True
            
        except OSError as e:
            raise FileAccessError(str(abs_path), "delete", f"Delete failed: {str(e)}")