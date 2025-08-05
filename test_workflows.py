#!/usr/bin/env python3
"""
Test script for both multi-gauge and refactored full delineation workflows
"""

import sys
from pathlib import Path
import json
import traceback
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_refactored_full_delineation():
    """Test the refactored full delineation workflow"""
    print("=" * 60)
    print("TESTING REFACTORED FULL DELINEATION WORKFLOW")
    print("=" * 60)
    
    try:
        from workflows.refactored_full_delineation import RefactoredFullDelineation
        
        # Test coordinates - Montreal area
        test_coords = [
            (45.5017, -73.5673, "Montreal_Downtown"),
            (45.4215, -75.6972, "Ottawa_Downtown"),
            (46.8139, -71.2080, "Quebec_City")
        ]
        
        results = {}
        
        for lat, lon, name in test_coords:
            print(f"\nTesting outlet: {name} ({lat}, {lon})")
            print("-" * 40)
            
            try:
                # Create workflow instance
                workflow = RefactoredFullDelineation(
                    workspace_dir=f"test_outputs/refactored_{name.lower()}"
                )
                
                # Execute single delineation
                result = workflow.execute_single_delineation(
                    latitude=lat,
                    longitude=lon,
                    outlet_name=name,
                    buffer_km=3.0
                )
                
                results[name] = result
                
                if result['success']:
                    print(f"✅ SUCCESS: {name}")
                    print(f"   Watershed area: {result.get('watershed_area_km2', 0):.1f} km²")
                    print(f"   HRUs: {result.get('total_hru_count', 0)}")
                    print(f"   Subbasins: {result.get('subbasin_count', 0)}")
                    print(f"   Model: {result.get('selected_model', 'Unknown')}")
                    print(f"   Model valid: {result.get('model_valid', False)}")
                    
                    # List output files
                    workspace = Path(result.get('workspace', ''))
                    if workspace.exists():
                        files = list(workspace.rglob('*'))
                        print(f"   Output files: {len(files)}")
                        
                        # Show key files
                        key_files = ['watershed_boundary', 'final_hrus_file', 'rvh_file', 'rvp_file', 'rvi_file']
                        for key in key_files:
                            if key in result and result[key]:
                                file_path = Path(result[key])
                                if file_path.exists():
                                    size_mb = file_path.stat().st_size / (1024 * 1024)
                                    print(f"     {key}: {file_path.name} ({size_mb:.2f} MB)")
                else:
                    print(f"❌ FAILED: {name}")
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ EXCEPTION: {name}")
                print(f"   Error: {str(e)}")
                print(f"   Traceback: {traceback.format_exc()}")
                results[name] = {'success': False, 'error': str(e)}
        
        # Save results
        results_file = Path("test_outputs/refactored_results.json")
        results_file.parent.mkdir(exist_ok=True, parents=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nRefactored workflow results saved to: {results_file}")
        return results
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in refactored workflow test: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'error': str(e)}

def test_multi_gauge_delineation():
    """Test the multi-gauge delineation workflow"""
    print("\n" + "=" * 60)
    print("TESTING MULTI-GAUGE DELINEATION WORKFLOW")
    print("=" * 60)
    
    try:
        from workflows.multi_gauge_delineation import MultiGaugeDelineation
        
        # Test regions
        test_regions = [
            ((-74.0, 45.0, -73.0, 46.0), "Montreal_Region"),
            ((-76.0, 45.0, -75.0, 46.0), "Ottawa_Region")
        ]
        
        results = {}
        
        for bbox, region_name in test_regions:
            print(f"\nTesting region: {region_name} {bbox}")
            print("-" * 40)
            
            try:
                # Create workflow instance
                workflow = MultiGaugeDelineation(
                    workspace_dir=f"test_outputs/multi_gauge_{region_name.lower()}"
                )
                
                # Execute multi-gauge workflow
                result = workflow.execute_multi_gauge_workflow(
                    bbox=bbox,
                    buffer_km=1.0,
                    min_drainage_area_km2=20.0,
                    gauge_buffer_km=3.0
                )
                
                results[region_name] = result
                
                if result['success']:
                    print(f"✅ SUCCESS: {region_name}")
                    print(f"   Gauges discovered: {result.get('gauges_discovered', 0)}")
                    print(f"   Gauges processed: {result.get('gauges_processed', 0)}")
                    print(f"   Successful: {result['summary']['completed']}")
                    print(f"   Failed: {result['summary']['failed']}")
                    print(f"   Total area: {result['summary']['total_area_km2']:.1f} km²")
                    print(f"   Execution time: {result.get('execution_time_minutes', 0):.1f} min")
                    
                    # Show area validation
                    validation = result['summary']['area_validation']
                    print(f"   Area validation:")
                    print(f"     Within 10%: {validation['within_10_percent']}")
                    print(f"     Within 25%: {validation['within_25_percent']}")
                    print(f"     Over 25%: {validation['over_25_percent']}")
                    
                    # Show individual gauge results
                    print(f"   Individual gauges:")
                    for station_id, gauge_result in result['gauge_results'].items():
                        if gauge_result['success']:
                            area = gauge_result.get('watershed_area_km2', 0)
                            hrus = gauge_result.get('total_hru_count', 0)
                            validation_status = gauge_result.get('area_validation', {}).get('validation_status', 'Unknown')
                            print(f"     {station_id}: {area:.1f} km², {hrus} HRUs, {validation_status}")
                        else:
                            print(f"     {station_id}: FAILED - {gauge_result.get('error', 'Unknown')}")
                            
                    # Generate summary report
                    summary_report = workflow.generate_summary_report(result)
                    report_file = Path(f"test_outputs/multi_gauge_{region_name.lower()}/summary_report.txt")
                    report_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(report_file, 'w') as f:
                        f.write(summary_report)
                    print(f"   Summary report: {report_file}")
                    
                else:
                    print(f"❌ FAILED: {region_name}")
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ EXCEPTION: {region_name}")
                print(f"   Error: {str(e)}")
                print(f"   Traceback: {traceback.format_exc()}")
                results[region_name] = {'success': False, 'error': str(e)}
        
        # Save results
        results_file = Path("test_outputs/multi_gauge_results.json")
        results_file.parent.mkdir(exist_ok=True, parents=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nMulti-gauge workflow results saved to: {results_file}")
        return results
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in multi-gauge workflow test: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'error': str(e)}

def analyze_outputs():
    """Analyze and compare outputs from both workflows"""
    print("\n" + "=" * 60)
    print("ANALYZING WORKFLOW OUTPUTS")
    print("=" * 60)
    
    # Check for output directories
    test_outputs = Path("test_outputs")
    if not test_outputs.exists():
        print("❌ No test outputs found")
        return
    
    print(f"Test outputs directory: {test_outputs.absolute()}")
    
    # List all output directories
    output_dirs = [d for d in test_outputs.iterdir() if d.is_dir()]
    print(f"Output directories: {len(output_dirs)}")
    
    for output_dir in output_dirs:
        print(f"\n📁 {output_dir.name}")
        
        # Count files
        all_files = list(output_dir.rglob('*'))
        file_count = len([f for f in all_files if f.is_file()])
        dir_count = len([f for f in all_files if f.is_dir()])
        
        print(f"   Files: {file_count}, Directories: {dir_count}")
        
        # Show key files
        key_patterns = ['*.geojson', '*.tif', '*.rv*', '*.json', '*.log']
        for pattern in key_patterns:
            matching_files = list(output_dir.rglob(pattern))
            if matching_files:
                print(f"   {pattern}: {len(matching_files)} files")
                for f in matching_files[:3]:  # Show first 3
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"     {f.relative_to(output_dir)} ({size_mb:.2f} MB)")
                if len(matching_files) > 3:
                    print(f"     ... and {len(matching_files) - 3} more")
    
    # Load and compare results
    refactored_results_file = test_outputs / "refactored_results.json"
    multi_gauge_results_file = test_outputs / "multi_gauge_results.json"
    
    comparison = {}
    
    if refactored_results_file.exists():
        with open(refactored_results_file) as f:
            refactored_results = json.load(f)
        comparison['refactored'] = refactored_results
        
        successful_refactored = sum(1 for r in refactored_results.values() if isinstance(r, dict) and r.get('success'))
        print(f"\n📊 Refactored workflow: {successful_refactored}/{len(refactored_results)} successful")
    
    if multi_gauge_results_file.exists():
        with open(multi_gauge_results_file) as f:
            multi_gauge_results = json.load(f)
        comparison['multi_gauge'] = multi_gauge_results
        
        successful_multi = sum(1 for r in multi_gauge_results.values() if isinstance(r, dict) and r.get('success'))
        print(f"📊 Multi-gauge workflow: {successful_multi}/{len(multi_gauge_results)} successful")
    
    # Save comparison
    comparison_file = test_outputs / "workflow_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\n📋 Workflow comparison saved to: {comparison_file}")

def main():
    """Main test function"""
    print("🚀 STARTING WORKFLOW TESTS")
    print(f"Timestamp: {datetime.now()}")
    print(f"Working directory: {Path.cwd()}")
    
    # Test refactored full delineation
    refactored_results = test_refactored_full_delineation()
    
    # Test multi-gauge delineation
    multi_gauge_results = test_multi_gauge_delineation()
    
    # Analyze outputs
    analyze_outputs()
    
    print("\n🏁 WORKFLOW TESTS COMPLETED")
    print("Check the test_outputs directory for detailed results")

if __name__ == "__main__":
    main()