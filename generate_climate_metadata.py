#!/usr/bin/env python3
"""
Generate comprehensive climate period metadata
Updates the climate step to include detailed period analysis and metadata
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clients.data_clients.climate_client import ClimateDataClient

def generate_climate_metadata():
    """Generate and save comprehensive climate period metadata"""
    
    # Initialize climate client
    client = ClimateDataClient()
    
    # Define paths
    climate_csv_path = Path("outlet_49.5738_-119.0368/climate/climate_forcing.csv")
    metadata_output_path = Path("outlet_49.5738_-119.0368/climate/climate_period_metadata.json")
    
    print("=" * 60)
    print("GENERATING CLIMATE PERIOD METADATA")
    print("=" * 60)
    
    # Check if climate data exists
    if not climate_csv_path.exists():
        print(f"ERROR: Climate data file not found: {climate_csv_path}")
        print("Please run the climate data download step first.")
        return
    
    print(f"Source: {climate_csv_path}")
    print(f"Output: {metadata_output_path}")
    print()
    
    # Generate comprehensive metadata
    metadata = client.generate_climate_period_metadata(
        climate_csv_path=climate_csv_path,
        metadata_output_path=metadata_output_path
    )
    
    if "error" in metadata:
        print(f"FAILED: {metadata['error']}")
        return
    
    print()
    print("=" * 60)
    print("CLIMATE METADATA SUMMARY")
    print("=" * 60)
    
    # Display key metadata highlights
    period = metadata['period_coverage']
    quality = metadata['data_quality']
    temp_stats = metadata['temperature_statistics']
    precip_stats = metadata['precipitation_statistics']
    
    print(f"📅 PERIOD COVERAGE:")
    print(f"   Years: {period['years_covered']} ({period['total_years']} years)")
    print(f"   Days: {period['total_days']:,} total records")
    print()
    
    print(f"📊 DATA QUALITY:")
    print(f"   Temperature completeness: {quality['temp_max_completeness_percent']:.1f}%")
    print(f"   Precipitation completeness: {quality['precip_completeness_percent']:.1f}%")
    print(f"   Missing data: {quality['missing_temp_max_days']} temp, {quality['missing_precip_days']} precip days")
    print()
    
    print(f"🌡️ TEMPERATURE STATISTICS:")
    print(f"   Range: {temp_stats['min_temperature']:.1f}°C to {temp_stats['max_temperature']:.1f}°C")
    print(f"   Mean annual: {temp_stats['mean_annual_temp_min']:.1f}°C to {temp_stats['mean_annual_temp_max']:.1f}°C")
    print()
    
    print(f"🌧️ PRECIPITATION STATISTICS:")
    print(f"   Total: {precip_stats['total_precipitation_mm']:,.1f} mm over {period['total_years']} years")
    print(f"   Mean annual: {precip_stats['mean_annual_precipitation_mm']:.1f} mm/year")
    print(f"   Max daily: {precip_stats['max_daily_precipitation_mm']:.1f} mm")
    print(f"   Wet days: {precip_stats['days_with_precipitation']:,} ({precip_stats['precipitation_frequency_percent']:.1f}%)")
    print()
    
    print(f"❄️ EXTREME EVENTS:")
    extremes = metadata['extreme_events']
    print(f"   Hottest day: {extremes['hottest_day']['temperature']:.1f}°C on {extremes['hottest_day']['date']}")
    print(f"   Coldest day: {extremes['coldest_day']['temperature']:.1f}°C on {extremes['coldest_day']['date']}")
    print(f"   Wettest day: {extremes['wettest_day']['precipitation_mm']:.1f}mm on {extremes['wettest_day']['date']}")
    print(f"   Longest dry period: {extremes['longest_dry_period_days']} days")
    print(f"   Freeze-thaw cycles: {extremes['freeze_thaw_cycles']} days/year")
    print()
    
    print(f"🎯 CLIMATE SUITABILITY:")
    suitability = metadata['climate_suitability']
    print(f"   Hydrological modeling: {suitability['hydrological_modeling']}")
    print(f"   Period adequacy: {suitability['period_adequacy']}")
    print(f"   Data gaps: {suitability['data_gaps']}")
    print()
    print(f"   Recommended for:")
    for use in suitability['recommended_for']:
        print(f"     • {use}")
    
    print()
    print("=" * 60)
    print("METADATA GENERATION COMPLETE")
    print(f"Full metadata saved to: {metadata_output_path}")
    print("=" * 60)

if __name__ == "__main__":
    generate_climate_metadata()
