#!/usr/bin/env python3
"""
Complete Workflow Orchestrator: Steps 1-5 with Climate Data in Sequence
Runs the complete RAVEN workflow from data preparation to model generation
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_command(command, step_name, cwd=None):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"RUNNING {step_name}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n{step_name} COMPLETED SUCCESSFULLY in {duration:.1f}s")
            return True
        else:
            print(f"\n{step_name} FAILED with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n{step_name} FAILED with exception: {e}")
        return False

def main():
    """Run complete workflow steps 1-5 in sequence or individual steps"""
    
    # Configuration
    latitude = 49.5738
    longitude = -119.0368
    outlet_name = f"outlet_{latitude:.4f}_{longitude:.4f}"
    workspace_dir = outlet_name  # Use outlet name as workspace directory
    
    # Parse command line arguments for individual step execution
    start_step = 1
    end_step = 6
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "4":
            start_step = end_step = 4
            print("Running only Step 4: HRU Generation")
        elif arg == "5":
            start_step = end_step = 6
            print("Running only Step 5: RAVEN Model Generation (including climate data)")
        elif arg == "step5":
            start_step = end_step = 6
            print("Running only Step 5: RAVEN Model Generation (including climate data)")
        elif arg == "4-5":
            start_step = 4
            end_step = 6
            print("Running Steps 4-5 only (including climate data)")
        elif arg == "1-3":
            start_step = 1
            end_step = 3
            print("Running Steps 1-3 only")
    
    # Create workspace directory automatically
    workspace_path = Path(workspace_dir)
    if not workspace_path.exists():
        print(f"Creating workspace directory: {workspace_dir}")
        workspace_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Using existing workspace directory: {workspace_dir}")
    
    print(f"""
==============================================================
                 COMPLETE RAVEN WORKFLOW                     
                 Steps 1-5 + Climate Data Orchestrator                   
==============================================================
 Outlet Coordinates: {latitude:.4f}, {longitude:.4f}                    
 Workspace: {workspace_dir}                                      
 Outlet Name: {outlet_name}                      
 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                        
==============================================================
    """)
    
    # Step definitions
    steps = [
        {
            "name": "STEP 1: Data Preparation",
            "command": f"python workflows/project_steps/step1_data_preparation/step1_data_preparation.py {latitude} {longitude} --workspace-dir {workspace_dir}",
            "required": True
        },
        {
            "name": "STEP 2: Watershed Delineation", 
            "command": f"python workflows/project_steps/step2_watershed_delineation/step2_watershed_delineation.py {latitude} {longitude} --workspace-dir {workspace_dir}",
            "required": True
        },
        {
            "name": "STEP 3: Lake Processing",
            "command": f"python workflows/project_steps/step3_lake_processing/step3_lake_processing.py \"{latitude},{longitude}\" --workspace-dir {workspace_dir}",
            "required": True
        },
        {
            "name": "STEP 4: HRU Generation",
            "command": f"python workflows/project_steps/step4_hru_generation/step4_hru_generation.py {latitude} {longitude} --workspace-dir {workspace_dir}",
            "required": True
        },
        {
            "name": "STEP 4.5: Climate & Hydrometric Data",
            "command": f"python workflows/project_steps/climate_hydrometric_data/step_climate_hydrometric_data.py {latitude} {longitude} --workspace {workspace_dir}",
            "required": True
        },
        {
            "name": "STEP 5: RAVEN Model Generation",
            "command": f"python workflows/project_steps/step5_raven_model/step5_raven_model.py {latitude} {longitude} --workspace-dir {workspace_dir} --outlet-name {outlet_name}",
            "required": True
        }
    ]
    
    # Track results
    results = {}
    failed_steps = []
    
    # Run selected steps
    for i, step in enumerate(steps, 1):
        # Skip steps outside the specified range
        if i < start_step or i > end_step:
            continue
            
        step_name = step["name"]
        command = step["command"]
        required = step.get("required", True)
        
        print(f"\nStarting Step {i}/5: {step_name}")
        
        success = run_command(command, step_name)
        results[step_name] = success
        
        if not success:
            failed_steps.append(step_name)
            if required:
                print(f"\nCRITICAL FAILURE: {step_name} is required but failed!")
                print("Stopping workflow execution.")
                break
            else:
                print(f"\nWARNING: {step_name} failed but is not critical. Continuing...")
        
        # Small delay between steps
        time.sleep(1)
    
    # Final summary
    print(f"\n{'='*80}")
    print("WORKFLOW EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    successful_steps = [name for name, success in results.items() if success]
    
    print(f"Successful Steps ({len(successful_steps)}):")
    for step_name in successful_steps:
        print(f"   - {step_name}")
    
    if failed_steps:
        print(f"\nFailed Steps ({len(failed_steps)}):")
        for step_name in failed_steps:
            print(f"   - {step_name}")
    
    print(f"\nWorkspace Directory: {workspace_dir}")
    print(f"Outlet Coordinates: {latitude:.4f}, {longitude:.4f}")
    print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(successful_steps) == len(steps):
        print(f"\nALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"RAVEN model files should be available in: {workspace_dir}/models/files/{outlet_name}/")
        return 0
    else:
        print(f"\nWORKFLOW INCOMPLETE: {len(failed_steps)} step(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)