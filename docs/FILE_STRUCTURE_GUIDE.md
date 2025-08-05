# RAVEN Workflow System - File Management Structure Guide

## 📁 **Complete Project Organization**

This document explains the complete file and directory structure of the RAVEN Workflow System, including the purpose of each component and how they work together.

---

## 🏗️ **Root Directory Structure**

```
Raven/                                    # Main project root
├── 🚀 raven_workflow_driver.py          # Main entry point - CLI driver script
├── 🖥️ run_raven_workflow.bat            # Windows launcher script
├── 🐧 run_raven_workflow.sh             # Unix/Linux launcher script
├── 📋 requirements.txt                   # Python dependencies
├── 🧪 pytest.ini                        # Testing configuration
├── 📝 raven_workflow.log                # Runtime log file (auto-generated)
├── 🔍 analyze_geopackage.py             # Utility for analyzing spatial data
├── 📚 MAIN_DRIVER_COMPLETE.md           # Implementation documentation
├── 📖 CLAUDE.md                         # Development notes
├── 🧠 CONTEXT_MEMORY.md                 # System context documentation
└── 📂 [Multiple organized directories]   # See detailed breakdown below
```

---

## 📂 **Core System Directories**

### **1. `/workflows/` - Workflow Engine Core**
```
workflows/
├── 🔧 __init__.py                       # Package initialization
├── 📋 steps/                            # Modular workflow steps library
│   ├── 🏗️ __init__.py                   # Steps package init + registry
│   ├── 🧱 base_step.py                  # Abstract base class for all steps
│   ├── ✅ validation_steps.py           # Coordinate & model validation steps
│   ├── 🗺️ routing_product_steps.py      # Routing product extraction steps
│   ├── 🏔️ dem_processing_steps.py       # DEM download & processing steps
│   ├── 🌊 watershed_steps.py            # Watershed delineation steps
│   ├── 🏞️ lake_processing_steps.py      # Lake detection & classification steps
│   ├── 🏘️ hru_generation_steps.py       # HRU (Hydrologic Response Unit) generation
│   └── 🧠 raven_generation_steps.py     # RAVEN model file generation steps
├── 🎯 approaches/                       # Complete workflow implementations
│   ├── 🏗️ __init__.py                   # Approaches package initialization
│   ├── ⚡ routing_product_workflow.py   # Approach A: Fast routing product workflow
│   └── 🔬 full_delineation_workflow.py  # Approach B: Comprehensive delineation workflow
└── 🧪 test_complete_workflow.py         # Comprehensive system testing
```

**Purpose**: Contains the core workflow engine with modular steps and complete workflow implementations.

### **2. `/processors/` - Data Processing Modules**
```
processors/
├── 🔧 context_manager.py               # Execution context management
├── 📊 data_collector.py                # Data collection and aggregation
├── 🏘️ hru_attributes.py                # HRU attribute calculation
├── 🏗️ hru_generator.py                 # HRU geometry generation
├── 💧 hydraulic_attributes.py          # Hydraulic property calculation
├── 🏞️ lake_classifier.py               # Lake classification algorithms
├── 🔍 lake_detection.py                # Lake detection from imagery/DEM
├── 🚰 lake_filter.py                   # Lake filtering and validation
├── 🔗 lake_integrator.py               # Lake-watershed integration
├── 📐 manning_calculator.py            # Manning's roughness calculation
├── 🧠 model_builder.py                 # RAVEN model construction
├── 🌐 network_simplifier.py            # Stream network simplification
├── 📍 polygon_overlay.py               # Spatial overlay operations
├── 🚀 raven_executor.py                # RAVEN model execution
├── 🏗️ raven_generator.py               # RAVEN file generation
├── 🗺️ routing_product_processor.py     # Routing product data processing
├── 📄 rvh_generator.py                 # RVH file (watershed structure) generation
├── ⚙️ rvp_generator.py                 # RVP file (parameters) generation
├── 📊 rvt_generator.py                 # RVT file (time series) generation
├── 🏘️ subbasin_grouper.py              # Sub-basin grouping algorithms
├── 🗺️ subregion_extractor.py           # Geographic subregion extraction
├── 🧮 basic_attributes.py              # Basic geometric attribute calculation
└── 📁 __pycache__/                     # Python bytecode cache (auto-generated)
```

**Purpose**: Specialized data processing modules for specific hydrological calculations and transformations.

### **3. `/clients/` - External Data Interface**
```
clients/
├── 📊 data_clients/                     # Data download and access clients
│   └── [Various data source connectors]
├── 📈 visualization_clients/            # Visualization and plotting clients
│   └── [Mapping and charting tools]
└── 🌊 watershed_clients/                # Watershed-specific data clients
    └── [Hydrological data connectors]
```

**Purpose**: Interface modules for connecting to external data sources and services.

### **4. `/models/` - RAVEN Model Templates**
```
models/
├── 🇨🇦 GR4JCN_template.yaml           # GR4J Canadian model template
├── 🏔️ HBVEC_template.yaml             # HBV-EC model template (with lakes)
├── ❄️ HMETS_config.yaml               # HMETS configuration
├── ❄️ HMETS_template.yaml             # HMETS model template (cold regions)
└── 🌲 UBCWM_template.yaml             # UBC Watershed Model template
```

**Purpose**: Pre-configured RAVEN model templates for different hydrological conditions and regions.

### **5. `/docs/` - Documentation**
```
docs/
├── 📖 README.md                        # Main project documentation
├── 🔧 WORKFLOW_DOCUMENTATION.md        # Workflow system documentation
├── 📊 LOCAL_DATA_SOURCES_CATALOG.md    # Available data sources catalog
├── 🧠 RAVEN_DEEP_RESEARCH_REPORT.md    # RAVEN modeling research
├── 📋 RAVEN_WORKFLOW_USAGE.md          # Usage instructions and examples
├── 🏗️ FILE_STRUCTURE_GUIDE.md         # This file - complete structure guide
├── 📂 processors/                      # Processor-specific documentation
│   └── [Individual processor docs]
├── 📂 extras/                          # Additional documentation
│   └── [Supplementary guides]
└── 📂 refactored/                      # Legacy documentation
    └── [Historical documentation]
```

**Purpose**: Comprehensive documentation for users, developers, and system administrators.

### **6. `/data/` - Data Storage**
```
data/
├── 🇨🇦 canadian/                       # Canadian-specific datasets
│   └── [Canadian hydrological data]
└── [Other regional datasets]
```

**Purpose**: Local data storage for frequently used datasets and cached downloads.

---

## 🚀 **Execution Flow and File Interactions**

### **1. Main Entry Points**

#### **Command Line Interface**
```bash
# Primary entry point
python raven_workflow_driver.py --lat 45.5017 --lon -73.5673 --project "MyProject"

# Platform-specific launchers
run_raven_workflow.bat    # Windows
run_raven_workflow.sh     # Unix/Linux/macOS
```

#### **Programmatic Interface**
```python
# Direct workflow execution
from workflows.approaches import RoutingProductWorkflow, FullDelineationWorkflow

# Approach A: Fast routing product
workflow_a = RoutingProductWorkflow("project_name")
result_a = workflow_a.execute_complete_workflow(45.5017, -73.5673)

# Approach B: Comprehensive delineation
workflow_b = FullDelineationWorkflow("project_name")
result_b = workflow_b.execute_complete_workflow(45.5017, -73.5673)
```

### **2. Workflow Execution Path**

```
🚀 raven_workflow_driver.py
    ↓
📋 workflows/approaches/[routing_product|full_delineation]_workflow.py
    ↓
🔧 workflows/steps/[various]_steps.py
    ↓
⚙️ processors/[specific]_processor.py
    ↓
📊 Generated RAVEN Model Files
```

### **3. Output File Structure**

#### **Generated Workspace Structure**
```
workspaces/
└── {project_name}/                      # User-specified project name
    ├── 📊 model.rvh                     # RAVEN watershed structure file
    ├── ⚙️ model.rvp                     # RAVEN parameters file
    ├── 🚀 model.rvi                     # RAVEN instructions file
    ├── 📈 model.rvt                     # RAVEN time series template
    ├── 🔧 model.rvc                     # RAVEN initial conditions file
    ├── 📋 model_summary.json            # Model metadata and validation
    ├── 🗺️ final_hrus.shp               # Generated HRUs (shapefile)
    ├── 🌊 extracted_catchments.shp      # Watershed catchments
    ├── 🏞️ extracted_lakes.shp          # Detected lakes
    ├── 📊 dem_conditioned.tif           # Processed DEM
    └── 📝 execution_log.txt             # Detailed execution log
```

---

## 🔧 **Configuration and Settings**

### **1. System Configuration Files**
```
.kiro/                                   # Kiro IDE configuration (if present)
├── settings/
│   └── mcp.json                        # Model Context Protocol settings
└── steering/
    └── *.md                            # Steering files for AI assistance
```

### **2. Development Configuration**
```
.git/                                   # Git version control
pytest.ini                             # Testing configuration
requirements.txt                       # Python dependencies
```

---

## 📊 **Data Flow Architecture**

### **Input Data Sources**
1. **User Coordinates** → Validation Steps
2. **Routing Product Data** → Extraction Steps  
3. **DEM Data** → Processing Steps
4. **Climate Data** → Integration Steps

### **Processing Pipeline**
1. **Validation** → Coordinate checking and area calculation
2. **Data Acquisition** → Download/extract spatial data
3. **Spatial Processing** → Generate watersheds, HRUs, lakes
4. **Model Generation** → Create RAVEN model files
5. **Validation** → Verify model completeness and accuracy

### **Output Products**
1. **RAVEN Model Files** → Ready for simulation
2. **Spatial Data** → GIS-compatible formats
3. **Metadata** → Model documentation and validation
4. **Logs** → Execution tracking and debugging

---

## 🛠️ **Development and Maintenance**

### **Key Development Files**
- `workflows/steps/base_step.py` - Base class for all workflow steps
- `workflows/__init__.py` - Step registry and discovery
- `raven_workflow_driver.py` - Main CLI interface
- `workflows/test_complete_workflow.py` - Comprehensive testing

### **Testing Structure**
```
🧪 Testing Files:
├── workflows/test_complete_workflow.py  # End-to-end system tests
├── pytest.ini                         # Test configuration
└── [Individual step tests]            # Unit tests for each component
```

### **Logging and Debugging**
```
📝 Log Files:
├── raven_workflow.log                 # Main system log
└── workspaces/{project}/execution_log.txt  # Project-specific logs
```

---

## 🎯 **Usage Patterns**

### **1. Quick Start (Routing Product)**
```bash
python raven_workflow_driver.py --approach A --lat 45.5017 --lon -73.5673 --project "QuickTest"
```
**Generated Files**: 5 RAVEN files + spatial data in `workspaces/QuickTest/`

### **2. Comprehensive Analysis (Full Delineation)**
```bash
python raven_workflow_driver.py --approach B --lat 45.5017 --lon -73.5673 --project "FullAnalysis"
```
**Generated Files**: 5 RAVEN files + detailed spatial analysis in `workspaces/FullAnalysis/`

### **3. Automated Selection**
```bash
python raven_workflow_driver.py --lat 45.5017 --lon -73.5673 --project "AutoSelect"
```
**Behavior**: System automatically chooses best approach based on data availability

---

## 🔍 **File Management Best Practices**

### **1. Workspace Organization**
- Each project gets its own workspace directory
- All outputs are contained within the project workspace
- Intermediate files are preserved for debugging and analysis

### **2. Data Management**
- Input data is cached in `/data/` for reuse
- Large files (DEMs, routing products) are downloaded as needed
- Temporary files are cleaned up automatically

### **3. Version Control**
- Core system files are version controlled
- Generated outputs are excluded from version control
- Configuration files are tracked for reproducibility

### **4. Backup and Recovery**
- Project workspaces can be archived independently
- System logs provide complete execution history
- Model files are self-contained and portable

---

## 🎉 **Summary**

The RAVEN Workflow System uses a **modular, hierarchical file structure** that separates:

- **🚀 Execution** (main drivers and launchers)
- **🔧 Core Logic** (workflows and steps)
- **⚙️ Processing** (specialized algorithms)
- **📊 Data** (inputs, outputs, and caches)
- **📚 Documentation** (guides and references)
- **🧪 Testing** (validation and quality assurance)

This structure enables:
- **Easy maintenance** and updates
- **Modular development** and testing
- **Clear separation** of concerns
- **Scalable architecture** for future enhancements
- **User-friendly operation** with multiple interfaces

The system successfully transforms **outlet coordinates** into **complete RAVEN hydrological models** while maintaining a clean, organized, and maintainable codebase.