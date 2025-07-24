#!/usr/bin/env python3
"""
Get detailed capabilities for all working WCS endpoints
"""
import requests
from xml.etree import ElementTree as ET
import json

def get_detailed_capabilities(endpoint_name):
    """Get detailed capabilities for a WCS endpoint"""
    url = f"https://datacube.services.geo.ca/ows/{endpoint_name}"
    
    try:
        response = requests.get(url, 
                              params={"SERVICE": "WCS", "VERSION": "1.1.1", "REQUEST": "GetCapabilities"},
                              timeout=15)
        
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            
            # Extract service information
            service_info = {
                'endpoint': endpoint_name,
                'url': url,
                'title': '',
                'abstract': '',
                'identifiers': [],
                'supported_crs': [],
                'supported_formats': [],
                'coverage_details': []
            }
            
            # Service identification
            title_elem = root.find(".//{http://www.opengis.net/ows/1.1}Title")
            if title_elem is not None:
                service_info['title'] = title_elem.text
            
            abstract_elem = root.find(".//{http://www.opengis.net/ows/1.1}Abstract")
            if abstract_elem is not None:
                service_info['abstract'] = abstract_elem.text
            
            # Coverage summaries
            coverage_summaries = root.findall(".//{http://www.opengis.net/wcs/1.1.1}CoverageSummary")
            
            for coverage in coverage_summaries:
                coverage_detail = {}
                
                # Title
                title = coverage.find(".//{http://www.opengis.net/ows/1.1}Title")
                if title is not None:
                    coverage_detail['title'] = title.text
                
                # Identifier
                identifier = coverage.find(".//{http://www.opengis.net/wcs/1.1.1}Identifier")
                if identifier is not None:
                    coverage_detail['identifier'] = identifier.text
                    service_info['identifiers'].append(identifier.text)
                
                # Bounding box
                bbox = coverage.find(".//{http://www.opengis.net/ows/1.1}WGS84BoundingBox")
                if bbox is not None:
                    lower = bbox.find(".//{http://www.opengis.net/ows/1.1}LowerCorner")
                    upper = bbox.find(".//{http://www.opengis.net/ows/1.1}UpperCorner")
                    if lower is not None and upper is not None:
                        coverage_detail['bbox'] = {
                            'lower': lower.text,
                            'upper': upper.text
                        }
                
                # Supported CRS
                crs_elements = coverage.findall(".//{http://www.opengis.net/wcs/1.1.1}SupportedCRS")
                crs_list = [crs.text for crs in crs_elements if crs.text]
                coverage_detail['supported_crs'] = crs_list
                service_info['supported_crs'].extend(crs_list)
                
                # Supported formats
                format_elements = coverage.findall(".//{http://www.opengis.net/wcs/1.1.1}SupportedFormat")
                format_list = [fmt.text for fmt in format_elements if fmt.text]
                coverage_detail['supported_formats'] = format_list
                service_info['supported_formats'].extend(format_list)
                
                if coverage_detail:
                    service_info['coverage_details'].append(coverage_detail)
            
            # Remove duplicates
            service_info['supported_crs'] = list(set(service_info['supported_crs']))
            service_info['supported_formats'] = list(set(service_info['supported_formats']))
            
            return service_info
            
    except Exception as e:
        return {'endpoint': endpoint_name, 'error': str(e)}

# Test all working endpoints
working_endpoints = ['elevation', 'landcover', 'vegetation']

print("🔍 DETAILED WCS CAPABILITIES ANALYSIS")
print("=" * 80)

all_services = {}

for endpoint in working_endpoints:
    print(f"\n📋 Analyzing {endpoint.upper()} service...")
    
    capabilities = get_detailed_capabilities(endpoint)
    all_services[endpoint] = capabilities
    
    if 'error' in capabilities:
        print(f"❌ Error: {capabilities['error']}")
        continue
    
    print(f"✅ Service: {capabilities['title']}")
    if capabilities['abstract']:
        print(f"   Description: {capabilities['abstract'][:100]}...")
    
    print(f"   Available datasets: {len(capabilities['identifiers'])}")
    for identifier in capabilities['identifiers']:
        print(f"   - {identifier}")
    
    print(f"   Supported formats: {', '.join(capabilities['supported_formats'])}")
    print(f"   Supported CRS: {len(capabilities['supported_crs'])} coordinate systems")
    
    # Show coverage details
    if capabilities['coverage_details']:
        print(f"   Coverage details:")
        for coverage in capabilities['coverage_details']:
            print(f"     • {coverage.get('title', 'Unnamed')}")
            print(f"       ID: {coverage.get('identifier', 'N/A')}")
            if 'bbox' in coverage:
                print(f"       Extent: {coverage['bbox']['lower']} to {coverage['bbox']['upper']}")

print("\n" + "=" * 80)
print("📊 COMPLETE SERVICE SUMMARY")
print("=" * 80)

total_datasets = 0
for service_name, service_data in all_services.items():
    if 'identifiers' in service_data:
        total_datasets += len(service_data['identifiers'])

print(f"🎯 Total WCS services found: {len(all_services)}")
print(f"🎯 Total datasets available: {total_datasets}")

print(f"\n📋 SERVICES BREAKDOWN:")
for service_name, service_data in all_services.items():
    if 'error' not in service_data:
        dataset_count = len(service_data['identifiers'])
        print(f"   {service_name:12} -> {dataset_count:2d} datasets")

print(f"\n🔧 RECOMMENDED WCS CLIENT UPDATES:")
print(f"   1. Add support for vegetation endpoint")
print(f"   2. Update get_landcover() to use correct identifier")
print(f"   3. Create new get_vegetation() function")
print(f"   4. Test all identifier combinations")

# Save detailed results
with open('/workspaces/raven/wcs_capabilities_detailed.json', 'w') as f:
    json.dump(all_services, f, indent=2)

print(f"\n💾 Detailed capabilities saved to: wcs_capabilities_detailed.json")
