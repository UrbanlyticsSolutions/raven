# RAVEN Hydrological Modeling System - Context Engineering Implementation

## Core Principles

### **❌ NEVER USE:**
- Synthetic data
- Mock data  
- Placeholder data
- Simulated data
- Demo data
- Fake data
- Fallbacks
- DO NOT ADD CODE TO md files

### **✅ ALWAYS USE:**
- Real data from data clients
- Actual downloads from APIs
- Live data from services
- Genuine spatial data
- Authentic climate/hydrometric data
- Update local md to track all tasks and findings

## Context Engineering Integration

### Project Architecture Context
- **Data Clients**: `clients/data_clients/` - Real data acquisition from ECCC, USGS, NRCan
- **Processors**: `processors/` - Lake detection, HRU generation, model building
- **Models**: `models/` - YAML templates for GR4JCN, HMETS, HBVEC, UBCWM
- **BasinMaker Integration**: `basinmaker-extracted/` - Full watershed delineation toolkit

### Context Engineering Status
- **Implementation Started**: 2025-08-04
- **Implementation Completed**: 2025-08-04
- **Tracking Documentation**: This file and context tracking files
- **Integration Strategy**: MCP-based context management for hydrological workflows

## MEMORIZED RESEARCH RESULTS

### RAVEN Model Purpose (MEMORIZED)
RAVEN answers: **"Given this weather, how much water will flow where and when?"**
- Transforms meteorological data into actionable water information
- Simulates complete water cycle at watershed scale
- Provides flood forecasting, reservoir management, water supply planning

### 5-Phase Workflow (MEMORIZED)
1. **Data Collection**: USGS 3DEP 30m DEM acquisition
2. **Watershed Delineation**: WhiteboxTools depression filling → boundaries
3. **Lake Detection**: Depression analysis comparing elevation surfaces
4. **Lake Classification**: Connectivity analysis (connected vs isolated)
5. **Model Generation**: Complete RAVEN 5-file input set

### Real Data Sources (MEMORIZED - NO SYNTHETIC)
- USGS 3DEP, Environment Canada, Water Survey Canada
- WhiteboxTools, BasinMaker validated methods
- All research results saved in local markdown files
