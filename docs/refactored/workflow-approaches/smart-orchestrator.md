# Smart Workflow Orchestrator

## Overview

The **Smart Workflow Orchestrator** automatically selects the optimal approach (Routing Product vs Full Delineation) based on data availability and user requirements. It provides a unified interface that delivers the best performance while maintaining flexibility.

## Intelligent Selection Logic

### **Automatic Approach Selection**
```python
class SmartRAVENWorkflow:
    def execute_workflow(self, latitude: float, longitude: float, 
                        approach: str = 'auto') -> Dict[str, Any]:
        """
        Execute RAVEN workflow with intelligent approach selection
        
        Parameters:
        -----------
        latitude, longitude : float
            Outlet coordinates
        approach : str
            'auto', 'routing_product', 'full_delineation', or 'hybrid'
        """
        
        if approach == 'auto':
            return self._auto_select_approach(latitude, longitude)
        elif approach == 'routing_product':
            return self._execute_routing_product_approach(latitude, longitude)
        elif approach == 'full_delineation':
            return self._execute_full_delineation_approach(latitude, longitude)
        elif approach == 'hybrid':
            return self._execute_hybrid_approach(latitude, longitude)
        else:
            raise ValueError(f"Unknown approach: {approach}")
```

### **Selection Decision Tree**
```
Input: Outlet Coordinates
    ↓
Check Routing Product Availability
    ↓
┌─────────────────┬─────────────────┐
│ Available       │ Not Available   │
│     ↓           │       ↓         │
│ Check Quality   │ Check DEM       │
│     ↓           │ Availability    │
│ ┌─────┬─────┐   │       ↓         │
│ │Good │Poor │   │ ┌─────┬─────┐   │
│ │  ↓  │  ↓  │   │ │ Yes │ No  │   │
│ │ A  │ B/H │   │ │  ↓  │  ↓  │   │
│ └────┴─────┘   │ │  B  │Error│   │
└─────────────────┴─┴─────┴─────┘
A = Routing Product Approach
B = Full Delineation Approach  
H = Hybrid Approach
```

## Implementation

### **Core Orchestrator Class**
```python
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from workflows.approaches.routing_product_workflow import RoutingProductWorkflow
from workflows.approaches.full_delineation_workflow import FullDelineationWorkflow
from workflows.approaches.hybrid_workflow import HybridWorkflow

class SmartRAVENWorkflow:
    """
    Smart workflow orchestrator with automatic approach selection
    """
    
    def __init__(self, workspace_dir: str = None, 
                 routing_product_paths: Dict[str, str] = None):
        """
        Initialize smart workflow orchestrator
        
        Parameters:
        -----------
        workspace_dir : str
            Base workspace directory for all processing
        routing_product_paths : Dict[str, str]
            Paths to available routing products by region
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "raven_workspace"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Default routing product paths
        self.routing_product_paths = routing_product_paths or {
            'canada': 'data/canadian/routing_product_v2.1/',
            'north_america': 'data/north_america/routing_product_v1.0/',
        }
        
        # Initialize workflow approaches
        self.routing_workflow = RoutingProductWorkflow(self.workspace_dir)
        self.delineation_workflow = FullDelineationWorkflow(self.workspace_dir)
        self.hybrid_workflow = HybridWorkflow(self.workspace_dir)
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def execute_workflow(self, latitude: float, longitude: float,
                        outlet_name: str = None,
                        approach: str = 'auto',
                        **kwargs) -> Dict[str, Any]:
        """
        Execute RAVEN workflow with smart approach selection
        
        Parameters:
        -----------
        latitude, longitude : float
            Outlet coordinates in decimal degrees
        outlet_name : str, optional
            Name for the outlet/watershed
        approach : str
            'auto', 'routing_product', 'full_delineation', or 'hybrid'
        **kwargs
            Additional parameters passed to specific workflows
            
        Returns:
        --------
        Dict with workflow results and metadata
        """
        
        start_time = datetime.now()
        outlet_name = outlet_name or f"outlet_{latitude:.4f}_{longitude:.4f}"
        
        self.logger.info(f"Starting RAVEN workflow for {outlet_name}")
        self.logger.info(f"Coordinates: {latitude:.6f}, {longitude:.6f}")
        self.logger.info(f"Approach: {approach}")
        
        try:
            # Execute workflow based on approach
            if approach == 'auto':
                result = self._auto_select_and_execute(latitude, longitude, outlet_name, **kwargs)
            elif approach == 'routing_product':
                result = self._execute_routing_product_approach(latitude, longitude, outlet_name, **kwargs)
            elif approach == 'full_delineation':
                result = self._execute_full_delineation_approach(latitude, longitude, outlet_name, **kwargs)
            elif approach == 'hybrid':
                result = self._execute_hybrid_approach(latitude, longitude, outlet_name, **kwargs)
            else:
                raise ValueError(f"Unknown approach: {approach}")
            
            # Add execution metadata
            execution_time = (datetime.now() - start_time).total_seconds() / 60.0
            result.update({
                'outlet_name': outlet_name,
                'execution_time_minutes': execution_time,
                'approach_used': result.get('approach_used', approach),
                'timestamp': datetime.now().isoformat()
            })
            
            if result.get('success', False):
                self.logger.info(f"Workflow completed successfully in {execution_time:.1f} minutes")
                self.logger.info(f"Approach used: {result['approach_used']}")
            else:
                self.logger.error(f"Workflow failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() / 60.0
            error_result = {
                'success': False,
                'error': str(e),
                'outlet_name': outlet_name,
                'execution_time_minutes': execution_time,
                'approach_used': approach,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.error(f"Workflow failed with exception: {str(e)}")
            return error_result
    
    def _auto_select_and_execute(self, latitude: float, longitude: float, 
                                outlet_name: str, **kwargs) -> Dict[str, Any]:
        """Automatically select and execute optimal approach"""
        
        self.logger.info("Auto-selecting optimal workflow approach...")
        
        # Step 1: Check routing product availability
        routing_product_info = self._check_routing_product_availability(latitude, longitude)
        
        if routing_product_info['available']:
            self.logger.info(f"Routing product available: {routing_product_info['product_path']}")
            
            # Step 2: Assess routing product quality
            quality_score = self._assess_routing_product_quality(routing_product_info, latitude, longitude)
            
            if quality_score >= 0.8:  # High quality
                self.logger.info(f"High quality routing product (score: {quality_score:.2f}) - using Approach A")
                result = self._execute_routing_product_approach(latitude, longitude, outlet_name, **kwargs)
                result['approach_used'] = 'routing_product'
                result['selection_reason'] = f'High quality routing product available (score: {quality_score:.2f})'
                
            elif quality_score >= 0.6:  # Medium quality
                self.logger.info(f"Medium quality routing product (score: {quality_score:.2f}) - using Hybrid approach")
                result = self._execute_hybrid_approach(latitude, longitude, outlet_name, **kwargs)
                result['approach_used'] = 'hybrid'
                result['selection_reason'] = f'Medium quality routing product, using hybrid approach (score: {quality_score:.2f})'
                
            else:  # Low quality
                self.logger.info(f"Low quality routing product (score: {quality_score:.2f}) - using Approach B")
                result = self._execute_full_delineation_approach(latitude, longitude, outlet_name, **kwargs)
                result['approach_used'] = 'full_delineation'
                result['selection_reason'] = f'Low quality routing product, using full delineation (score: {quality_score:.2f})'
        
        else:
            self.logger.info("No routing product available - checking DEM availability")
            
            # Step 3: Check DEM availability for full delineation
            dem_available = self._check_dem_availability(latitude, longitude)
            
            if dem_available:
                self.logger.info("DEM data available - using Approach B")
                result = self._execute_full_delineation_approach(latitude, longitude, outlet_name, **kwargs)
                result['approach_used'] = 'full_delineation'
                result['selection_reason'] = 'No routing product available, DEM-based delineation used'
            else:
                self.logger.error("Neither routing product nor DEM data available")
                return {
                    'success': False,
                    'error': 'No data sources available for watershed delineation',
                    'approach_used': 'none',
                    'selection_reason': 'No routing product or DEM data available'
                }
        
        return result
    
    def _check_routing_product_availability(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Check if routing product covers the given coordinates"""
        
        for region, product_path in self.routing_product_paths.items():
            if not Path(product_path).exists():
                continue
                
            # Check geographic bounds for region
            if region == 'canada' and self._is_in_canada(latitude, longitude):
                return {
                    'available': True,
                    'region': region,
                    'product_path': product_path,
                    'version': self._get_routing_product_version(product_path)
                }
            elif region == 'north_america' and self._is_in_north_america(latitude, longitude):
                return {
                    'available': True,
                    'region': region,
                    'product_path': product_path,
                    'version': self._get_routing_product_version(product_path)
                }
        
        return {'available': False}
    
    def _assess_routing_product_quality(self, routing_info: Dict, latitude: float, longitude: float) -> float:
        """Assess quality of routing product for given location"""
        
        quality_score = 0.0
        
        # Base score for having routing product
        quality_score += 0.5
        
        # Version score
        version = routing_info.get('version', 'unknown')
        if 'v2.1' in version:
            quality_score += 0.2
        elif 'v2.0' in version:
            quality_score += 0.15
        elif 'v1.0' in version:
            quality_score += 0.1
        
        # Region-specific quality
        region = routing_info.get('region', '')
        if region == 'canada':
            quality_score += 0.2  # High quality for Canada
        elif region == 'north_america':
            quality_score += 0.1  # Medium quality for North America
        
        # Distance from major population centers (proxy for data quality)
        if self._near_major_city(latitude, longitude):
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _check_dem_availability(self, latitude: float, longitude: float) -> bool:
        """Check if DEM data is available for coordinates"""
        
        # USGS 3DEP covers North America
        if self._is_in_north_america(latitude, longitude):
            return True
        
        # SRTM covers most of the world
        if -60 <= latitude <= 60:
            return True
        
        return False
    
    def _is_in_canada(self, latitude: float, longitude: float) -> bool:
        """Check if coordinates are within Canada"""
        return (41.0 <= latitude <= 84.0 and -141.0 <= longitude <= -52.0)
    
    def _is_in_north_america(self, latitude: float, longitude: float) -> bool:
        """Check if coordinates are within North America"""
        return (25.0 <= latitude <= 85.0 and -170.0 <= longitude <= -50.0)
    
    def _near_major_city(self, latitude: float, longitude: float) -> bool:
        """Check if coordinates are near major cities (better data quality)"""
        major_cities = [
            (45.5017, -73.5673),  # Montreal
            (43.6532, -79.3832),  # Toronto
            (49.2827, -123.1207), # Vancouver
            (51.0447, -114.0719), # Calgary
            (53.5461, -113.4938), # Edmonton
        ]
        
        for city_lat, city_lon in major_cities:
            distance = ((latitude - city_lat)**2 + (longitude - city_lon)**2)**0.5
            if distance < 2.0:  # Within ~200km
                return True
        
        return False
    
    def _get_routing_product_version(self, product_path: str) -> str:
        """Extract version from routing product path"""
        path_str = str(product_path).lower()
        if 'v2.1' in path_str:
            return 'v2.1'
        elif 'v2.0' in path_str:
            return 'v2.0'
        elif 'v1.0' in path_str:
            return 'v1.0'
        return 'unknown'
    
    def _execute_routing_product_approach(self, latitude: float, longitude: float, 
                                        outlet_name: str, **kwargs) -> Dict[str, Any]:
        """Execute routing product workflow"""
        return self.routing_workflow.execute_complete_workflow(
            latitude=latitude,
            longitude=longitude,
            outlet_name=outlet_name,
            **kwargs
        )
    
    def _execute_full_delineation_approach(self, latitude: float, longitude: float,
                                         outlet_name: str, **kwargs) -> Dict[str, Any]:
        """Execute full delineation workflow"""
        return self.delineation_workflow.execute_complete_workflow(
            latitude=latitude,
            longitude=longitude,
            outlet_name=outlet_name,
            **kwargs
        )
    
    def _execute_hybrid_approach(self, latitude: float, longitude: float,
                               outlet_name: str, **kwargs) -> Dict[str, Any]:
        """Execute hybrid workflow"""
        return self.hybrid_workflow.execute_complete_workflow(
            latitude=latitude,
            longitude=longitude,
            outlet_name=outlet_name,
            **kwargs
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for workflow orchestrator"""
        logger = logging.getLogger('SmartRAVENWorkflow')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
```

## Usage Examples

### **Basic Usage (Automatic Selection)**
```python
from workflows import SmartRAVENWorkflow

# Initialize orchestrator
workflow = SmartRAVENWorkflow(workspace_dir="my_watershed_project")

# Execute with automatic approach selection
result = workflow.execute_workflow(
    latitude=45.5017,
    longitude=-73.5673,
    outlet_name="Montreal_Smart_Test"
)

print(f"Approach used: {result['approach_used']}")
print(f"Execution time: {result['execution_time_minutes']:.1f} minutes")
print(f"Selection reason: {result['selection_reason']}")
```

### **Force Specific Approach**
```python
# Force routing product approach
result = workflow.execute_workflow(
    latitude=45.5017,
    longitude=-73.5673,
    approach='routing_product'
)

# Force full delineation approach
result = workflow.execute_workflow(
    latitude=45.5017,
    longitude=-73.5673,
    approach='full_delineation'
)

# Use hybrid approach
result = workflow.execute_workflow(
    latitude=45.5017,
    longitude=-73.5673,
    approach='hybrid'
)
```

### **Batch Processing**
```python
outlets = [
    {'lat': 45.5017, 'lon': -73.5673, 'name': 'Montreal'},
    {'lat': 43.6532, 'lon': -79.3832, 'name': 'Toronto'},
    {'lat': 49.2827, 'lon': -123.1207, 'name': 'Vancouver'}
]

results = []
for outlet in outlets:
    result = workflow.execute_workflow(
        latitude=outlet['lat'],
        longitude=outlet['lon'],
        outlet_name=outlet['name']
    )
    results.append(result)

# Analyze results
for result in results:
    print(f"{result['outlet_name']}: {result['approach_used']} "
          f"({result['execution_time_minutes']:.1f} min)")
```

## Performance Monitoring

### **Execution Metrics**
```python
class WorkflowMetrics:
    def __init__(self):
        self.executions = []
    
    def record_execution(self, result: Dict[str, Any]):
        """Record workflow execution metrics"""
        self.executions.append({
            'timestamp': result['timestamp'],
            'approach': result['approach_used'],
            'execution_time': result['execution_time_minutes'],
            'success': result['success'],
            'outlet_name': result['outlet_name']
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.executions:
            return {'message': 'No executions recorded'}
        
        successful = [e for e in self.executions if e['success']]
        
        return {
            'total_executions': len(self.executions),
            'successful_executions': len(successful),
            'success_rate': len(successful) / len(self.executions),
            'average_execution_time': sum(e['execution_time'] for e in successful) / len(successful) if successful else 0,
            'approach_distribution': self._get_approach_distribution(),
            'fastest_execution': min(e['execution_time'] for e in successful) if successful else None,
            'slowest_execution': max(e['execution_time'] for e in successful) if successful else None
        }
    
    def _get_approach_distribution(self) -> Dict[str, int]:
        """Get distribution of approaches used"""
        distribution = {}
        for execution in self.executions:
            approach = execution['approach']
            distribution[approach] = distribution.get(approach, 0) + 1
        return distribution
```

## Configuration Options

### **Orchestrator Configuration**
```python
config = {
    'routing_product_paths': {
        'canada': 'data/canadian/routing_product_v2.1/',
        'north_america': 'data/north_america/routing_product_v1.0/',
        'custom_region': 'data/custom/routing_product/'
    },
    'quality_thresholds': {
        'high_quality': 0.8,
        'medium_quality': 0.6,
        'low_quality': 0.4
    },
    'fallback_preferences': [
        'routing_product',
        'hybrid',
        'full_delineation'
    ],
    'performance_targets': {
        'max_execution_time_minutes': 30,
        'min_success_rate': 0.9
    }
}

workflow = SmartRAVENWorkflow(
    workspace_dir="my_project",
    routing_product_paths=config['routing_product_paths']
)
```

## Error Handling and Recovery

### **Automatic Fallback**
```python
def execute_with_fallback(self, latitude: float, longitude: float, **kwargs) -> Dict[str, Any]:
    """Execute workflow with automatic fallback on failure"""
    
    approaches = ['routing_product', 'hybrid', 'full_delineation']
    
    for approach in approaches:
        try:
            self.logger.info(f"Attempting {approach} approach...")
            result = self.execute_workflow(
                latitude=latitude,
                longitude=longitude,
                approach=approach,
                **kwargs
            )
            
            if result.get('success', False):
                result['fallback_used'] = approach != approaches[0]
                return result
            else:
                self.logger.warning(f"{approach} approach failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"{approach} approach failed with exception: {str(e)}")
            continue
    
    return {
        'success': False,
        'error': 'All workflow approaches failed',
        'approaches_tried': approaches
    }
```

## Benefits of Smart Orchestration

### **Performance Optimization**
- **Automatic Selection**: Always uses fastest available approach
- **Quality Assessment**: Balances speed vs accuracy based on data quality
- **Resource Management**: Optimizes CPU, memory, and network usage

### **User Experience**
- **Simplified Interface**: Single function call for all scenarios
- **Transparent Selection**: Clear reasoning for approach selection
- **Consistent Output**: Same result format regardless of approach

### **Operational Benefits**
- **Reliability**: Automatic fallback on failures
- **Monitoring**: Built-in performance tracking
- **Scalability**: Optimized for batch processing

## Related Documentation

- [Approach A: Routing Product](./approach-a-routing-product.md)
- [Approach B: Full Delineation](./approach-b-full-delineation.md)
- [Performance Optimization](./performance-guide.md)
- [Error Handling Guide](./error-handling.md)

---

**The Smart Orchestrator provides the optimal balance of performance, reliability, and ease of use for RAVEN workflow execution.**