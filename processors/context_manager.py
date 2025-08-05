#!/usr/bin/env python3
"""
Context Manager for Raven Hydrological Modeling System
Implements context engineering patterns for AI agent workflow understanding
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class WorkflowContext:
    """Represents the current state of a hydrological modeling workflow"""
    workflow_id: str
    created_at: str
    updated_at: str
    
    # Spatial context
    study_area_bounds: Optional[tuple] = None
    watershed_data: Optional[Dict] = None
    hru_count: Optional[int] = None
    
    # Temporal context  
    simulation_period: Optional[Dict] = None
    climate_data_period: Optional[Dict] = None
    
    # Data context
    data_sources: Dict[str, List[str]] = None
    data_quality: Dict[str, Dict] = None
    
    # Model context
    model_type: Optional[str] = None
    model_configuration: Optional[Dict] = None
    
    # Processing context
    completed_steps: List[str] = None
    failed_steps: List[str] = None
    current_step: Optional[str] = None
    
    # Output context
    generated_files: List[str] = None
    model_outputs: Optional[Dict] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = {}
        if self.data_quality is None:
            self.data_quality = {}
        if self.completed_steps is None:
            self.completed_steps = []
        if self.failed_steps is None:
            self.failed_steps = []
        if self.generated_files is None:
            self.generated_files = []


class ContextManager:
    """
    Context Engineering Manager for Raven Hydrological Modeling
    
    Provides:
    1. Workflow context persistence across AI sessions
    2. Data relationship tracking
    3. Intelligent context summarization
    4. Context-aware task planning
    """
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.context_dir = self.workspace_dir / ".context"
        self.context_dir.mkdir(exist_ok=True, parents=True)
        
        self.current_context: Optional[WorkflowContext] = None
        self.context_file = self.context_dir / "current_workflow.json"
        
        # Load existing context if available
        self._load_current_context()
    
    def create_new_workflow(self, workflow_name: str = None) -> str:
        """Create a new workflow context"""
        if workflow_name is None:
            workflow_name = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        workflow_id = self._generate_workflow_id(workflow_name)
        timestamp = datetime.now().isoformat()
        
        self.current_context = WorkflowContext(
            workflow_id=workflow_id,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        self._save_current_context()
        self._update_progress_tracking("workflow_created", workflow_id)
        
        return workflow_id
    
    def update_spatial_context(self, bounds: tuple, watershed_data: Dict = None):
        """Update spatial context information"""
        if not self.current_context:
            self.create_new_workflow()
        
        self.current_context.study_area_bounds = bounds
        self.current_context.watershed_data = watershed_data
        self.current_context.updated_at = datetime.now().isoformat()
        
        self._save_current_context()
        self._update_progress_tracking("spatial_context_updated", 
                                     f"bounds: {bounds}")
    
    def update_temporal_context(self, simulation_period: Dict, 
                              climate_period: Dict = None):
        """Update temporal context information"""
        if not self.current_context:
            self.create_new_workflow()
        
        self.current_context.simulation_period = simulation_period
        if climate_period:
            self.current_context.climate_data_period = climate_period
        self.current_context.updated_at = datetime.now().isoformat()
        
        self._save_current_context()
        self._update_progress_tracking("temporal_context_updated", 
                                     str(simulation_period))
    
    def register_data_source(self, source_type: str, source_details: List[str],
                           quality_metrics: Dict = None):
        """Register a data source and its quality metrics"""
        if not self.current_context:
            self.create_new_workflow()
        
        self.current_context.data_sources[source_type] = source_details
        if quality_metrics:
            self.current_context.data_quality[source_type] = quality_metrics
        
        self.current_context.updated_at = datetime.now().isoformat()
        self._save_current_context()
        self._update_progress_tracking("data_source_registered", 
                                     f"{source_type}: {len(source_details)} sources")
    
    def update_model_context(self, model_type: str, configuration: Dict = None):
        """Update model context information"""
        if not self.current_context:
            self.create_new_workflow()
        
        self.current_context.model_type = model_type
        if configuration:
            self.current_context.model_configuration = configuration
        
        self.current_context.updated_at = datetime.now().isoformat()
        self._save_current_context()
        self._update_progress_tracking("model_context_updated", model_type)
    
    def mark_step_completed(self, step_name: str, outputs: List[str] = None):
        """Mark a processing step as completed"""
        if not self.current_context:
            self.create_new_workflow()
        
        if step_name not in self.current_context.completed_steps:
            self.current_context.completed_steps.append(step_name)
        
        if step_name in self.current_context.failed_steps:
            self.current_context.failed_steps.remove(step_name)
        
        if outputs:
            self.current_context.generated_files.extend(outputs)
        
        self.current_context.updated_at = datetime.now().isoformat()
        self._save_current_context()
        self._update_progress_tracking("step_completed", step_name)
    
    def mark_step_failed(self, step_name: str, error_message: str):
        """Mark a processing step as failed"""
        if not self.current_context:
            self.create_new_workflow()
        
        if step_name not in self.current_context.failed_steps:
            self.current_context.failed_steps.append(step_name)
        
        if step_name in self.current_context.completed_steps:
            self.current_context.completed_steps.remove(step_name)
        
        self.current_context.updated_at = datetime.now().isoformat()
        self._save_current_context()
        self._update_progress_tracking("step_failed", f"{step_name}: {error_message}")
    
    def get_context_summary(self) -> Dict:
        """Get an intelligent summary of current context for AI agents"""
        if not self.current_context:
            return {"status": "no_active_workflow"}
        
        summary = {
            "workflow_id": self.current_context.workflow_id,
            "status": self._get_workflow_status(),
            "progress": {
                "completed_steps": len(self.current_context.completed_steps),
                "failed_steps": len(self.current_context.failed_steps),
                "completion_rate": self._calculate_completion_rate()
            },
            "spatial_context": {
                "has_study_area": self.current_context.study_area_bounds is not None,
                "has_watershed_data": self.current_context.watershed_data is not None,
                "hru_count": self.current_context.hru_count
            },
            "temporal_context": {
                "has_simulation_period": self.current_context.simulation_period is not None,
                "has_climate_data": self.current_context.climate_data_period is not None
            },
            "data_context": {
                "source_types": list(self.current_context.data_sources.keys()),
                "total_sources": sum(len(sources) for sources in self.current_context.data_sources.values()),
                "quality_assessed": list(self.current_context.data_quality.keys())
            },
            "model_context": {
                "model_type": self.current_context.model_type,
                "configured": self.current_context.model_configuration is not None
            },
            "outputs": {
                "files_generated": len(self.current_context.generated_files),
                "model_executed": self.current_context.model_outputs is not None
            }
        }
        
        return summary
    
    def get_next_recommended_steps(self) -> List[str]:
        """Get AI agent recommendations for next workflow steps"""
        if not self.current_context:
            return ["create_new_workflow"]
        
        completed = set(self.current_context.completed_steps)
        failed = set(self.current_context.failed_steps)
        
        # Define workflow step dependencies
        step_dependencies = {
            "spatial_data_acquisition": [],
            "watershed_delineation": ["spatial_data_acquisition"],
            "hru_generation": ["watershed_delineation"],
            "climate_data_acquisition": [],
            "climate_data_processing": ["climate_data_acquisition"],
            "streamflow_data_acquisition": [],
            "model_configuration": ["hru_generation", "climate_data_processing"],
            "model_execution": ["model_configuration"],
            "model_validation": ["model_execution", "streamflow_data_acquisition"]
        }
        
        recommendations = []
        
        for step, dependencies in step_dependencies.items():
            if step in completed:
                continue
            
            if step in failed:
                recommendations.append(f"retry_{step}")
                continue
            
            # Check if dependencies are met
            deps_met = all(dep in completed for dep in dependencies)
            if deps_met:
                recommendations.append(step)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _generate_workflow_id(self, workflow_name: str) -> str:
        """Generate unique workflow ID"""
        timestamp = datetime.now().isoformat()
        content = f"{workflow_name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_workflow_status(self) -> str:
        """Determine current workflow status"""
        if not self.current_context.completed_steps:
            return "initialized"
        
        if self.current_context.failed_steps:
            return "errors_present"
        
        if len(self.current_context.completed_steps) >= 7:  # Most steps completed
            return "near_completion"
        elif len(self.current_context.completed_steps) >= 3:
            return "in_progress"
        else:
            return "early_stage"
    
    def _calculate_completion_rate(self) -> float:
        """Calculate workflow completion percentage"""
        total_possible_steps = 9  # Typical hydrological modeling workflow
        completed = len(self.current_context.completed_steps)
        return min(completed / total_possible_steps, 1.0)
    
    def _save_current_context(self):
        """Save current context to file"""
        if self.current_context:
            with open(self.context_file, 'w') as f:
                json.dump(asdict(self.current_context), f, indent=2)
    
    def _load_current_context(self):
        """Load existing context from file"""
        if self.context_file.exists():
            try:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                self.current_context = WorkflowContext(**data)
            except Exception as e:
                print(f"Warning: Could not load existing context: {e}")
                self.current_context = None
    
    def _update_progress_tracking(self, action: str, details: str):
        """Update progress tracking in markdown file"""
        progress_file = Path("CONTEXT_ENGINEERING_PROGRESS.md")
        
        if progress_file.exists():
            with open(progress_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n### {timestamp} - {action}\n")
                f.write(f"- **Details**: {details}\n")
                if self.current_context:
                    f.write(f"- **Workflow**: {self.current_context.workflow_id}\n")


def test_context_manager():
    """Test the context manager functionality"""
    print("Testing Context Manager for Raven Hydrological Modeling...")
    
    # Initialize context manager
    ctx = ContextManager()
    
    # Create new workflow
    workflow_id = ctx.create_new_workflow("test_watershed_modeling")
    print(f"✓ Created workflow: {workflow_id}")
    
    # Update contexts
    ctx.update_spatial_context((-75.0, 45.0, -74.0, 46.0))
    ctx.register_data_source("climate", ["ECCC_station_123"], {"completeness": 0.95})
    ctx.update_model_context("GR4JCN")
    
    # Mark steps
    ctx.mark_step_completed("spatial_data_acquisition", ["dem.tif", "boundaries.shp"])
    ctx.mark_step_completed("climate_data_acquisition", ["climate.nc"])
    
    # Get summary
    summary = ctx.get_context_summary()
    print(f"✓ Context summary: {summary['status']}, {summary['progress']['completion_rate']:.1%} complete")
    
    # Get recommendations
    recommendations = ctx.get_next_recommended_steps()
    print(f"✓ Next steps: {recommendations}")
    
    print("✓ Context Manager test completed successfully")


if __name__ == "__main__":
    test_context_manager()