# DEM Integration for Fluvial Inversion

This document explains how to use the `prepare_inversion_data()` wrapper functions to extract fluvial network data from DEM grids for use with the fluvial inversion tools.

## Overview

The fluvial inversion code requires specific 1D array inputs:
- `x, y`: Pixel coordinates [L]
- `z`: Elevations [L]
- `rec_array`: Flow network topology (receiver array)
- `area_array`: Drainage areas [L²]
- `slope_array`: Local slopes [dimensionless]

The `dem.py` toolkit stores these as 2D grids. The wrapper functions bridge the gap by:
1. Filtering pixels by drainage area threshold
2. Extracting connected fluvial network
3. Converting from grid format to 1D arrays
4. Building the receiver array from flow directions

## Quick Start

```python
from dem import Elevation, FilledElevation, FlowDirectionD8, Area
from fluvial_inversion import (
    prepare_inversion_data,
    findm_slope_area,
    calculate_chi,
    invert_block_uplift
)

# Load and process DEM
dem = Elevation(gdal_filename='my_dem.tif')
filled_dem = FilledElevation(elevation=dem)
flow_dir = FlowDirectionD8(flooded_dem=filled_dem)
area_grid = Area(flow_direction=flow_dir)

# Extract fluvial network
outlet = (523000.0, 4567000.0)  # UTM coordinates
min_area = 1e6  # 1 km² in m²

data = prepare_inversion_data(
    dem=filled_dem,
    area=area_grid,
    flow_direction=flow_dir,
    outlet_location=outlet,
    min_drainage_area=min_area
)

# Run inversion workflow
m, _, _, _ = findm_slope_area(data['slope_array'], data['area_array'])
chi = calculate_chi(data['x'], data['y'], data['rec_array'], data['area_array'], m)
Ustar, tstar, misfit = invert_block_uplift(chi, data['z'], gamma=1.0, q=10)
```

## Functions

### `prepare_inversion_data()`

Full-featured extraction with outlet-based network identification.

**Parameters:**
- `dem`: Elevation or GeographicElevation object
- `area`: Area or GeographicArea object
- `flow_direction`: FlowDirectionD8 object
- `outlet_location`: (x, y) tuple of outlet coordinates
- `min_drainage_area`: Minimum area threshold [L²]
- `max_search_length`: Maximum upstream search distance (optional)

**Returns:**
Dictionary with keys:
- `'x'`, `'y'`: Coordinates [L]
- `'z'`: Elevations [L]
- `'rec_array'`: Receiver indices
- `'area_array'`: Drainage areas [L²]
- `'slope_array'`: Slopes
- `'n_pixels'`: Number of pixels

**Use when:** You want to analyze a specific watershed from a known outlet.

### `prepare_inversion_data_simple()`

Simplified extraction without outlet specification.

**Parameters:**
- `dem`: Elevation object
- `area`: Area object
- `flow_direction`: FlowDirectionD8 object
- `min_drainage_area`: Minimum area threshold [L²]

**Returns:** Same structure as `prepare_inversion_data()`

**Use when:** You want all pixels above a threshold, not just a single watershed.

## Workflow Details

### Step 1: Load DEM

```python
from dem import Elevation, GeographicElevation

# For projected coordinates (UTM, etc.)
dem = Elevation(gdal_filename='dem.tif')

# For geographic coordinates (lat/lon)
dem = GeographicElevation(gdal_filename='dem.tif')
```

Supported formats: Any GDAL-readable format (GeoTIFF, Arc ASCII, etc.)

### Step 2: Fill Depressions

```python
from dem import FilledElevation, GeographicFilledElevation

filled_dem = FilledElevation(elevation=dem)
```

This removes pits and ensures continuous flow paths.

### Step 3: Calculate Flow Directions

```python
from dem import FlowDirectionD8

flow_dir = FlowDirectionD8(flooded_dem=filled_dem)
```

Uses D8 algorithm with flow codes:
- 1 = East
- 2 = Southeast
- 4 = South
- 8 = Southwest
- 16 = West
- 32 = Northwest
- 64 = North
- 128 = Northeast

### Step 4: Calculate Drainage Area

```python
from dem import Area, GeographicArea

area_grid = Area(flow_direction=flow_dir)
```

Accumulates flow from upstream pixels.

### Step 5: Extract Network

```python
data = prepare_inversion_data(
    dem=filled_dem,
    area=area_grid,
    flow_direction=flow_dir,
    outlet_location=(x_outlet, y_outlet),
    min_drainage_area=1e6  # Adjust for your region
)
```

**Choosing `min_drainage_area`:**
- Too small: Includes hillslopes, noisy data
- Too large: Misses tributary information
- Typical: 0.1-10 km² (1e5 to 1e7 m²) depending on DEM resolution

### Step 6: Run Inversion

See main package README for full inversion workflow.

## Complete Example

See `examples/dem_to_inversion_workflow.py` for a complete end-to-end example including visualization.

```bash
cd examples
python dem_to_inversion_workflow.py
```

## Receiver Array Explained

The receiver array encodes flow network topology. For each pixel `i`:
- `rec_array[i] = j` means pixel `i` flows to pixel `j`
- `rec_array[i] = i` means pixel `i` is an outlet

This compact representation allows efficient upstream/downstream traversal.

Example:
```
Pixel:      0    1    2    3    4
rec_array: [1,   2,   4,   4,   4]
            ↓    ↓    ↓    ↓    ↓
Flow:      0→1→2 ─→ 3─→ 4 (outlet)
```

## Coordinate Systems

The wrapper preserves the DEM's coordinate system:
- If DEM is in UTM: outputs are UTM (meters)
- If DEM is in geographic: outputs are lat/lon (degrees)

For inversion, units must be consistent:
- Coordinates, elevations, chi: all in same length unit [L]
- Areas: [L²]
- Slopes: [L/L] = dimensionless

## Troubleshooting

### Error: "No pixels found with drainage area >= X"

**Cause:** Threshold too high or units mismatch

**Fix:**
1. Check max drainage area: `np.max(area_grid._griddata)`
2. Verify units (m² vs km²)
3. Reduce threshold

### Error: "No connected fluvial network found"

**Cause:** Outlet location incorrect or not on network

**Fix:**
1. Verify outlet coordinates match DEM projection
2. Check outlet is downstream (low elevation)
3. Use `prepare_inversion_data_simple()` instead

### Network too small/large

**Adjust:** `min_drainage_area` parameter
- Smaller threshold → larger network
- Larger threshold → smaller network (main channels only)

### Slope values seem wrong

**Cause:** DEM may have artifacts or need smoothing

**Fix:**
```python
# Smooth DEM before processing
from scipy.ndimage import gaussian_filter
dem._griddata = gaussian_filter(dem._griddata, sigma=2)
```

## Performance Tips

For large DEMs (> 10,000 x 10,000):

1. **Clip to region of interest** before processing
2. **Increase min_drainage_area** to reduce network size
3. **Use max_search_length** to limit upstream search
4. **Process in geographic coordinates** only if necessary (projected coords are faster)

## Data Format Requirements

**DEM (`dem.py` classes):**
- Must have `_griddata` attribute (2D numpy array)
- Must have `_georef_info` with coordinate metadata
- NoData values should be `np.nan`

**Grid alignment:**
- DEM, area, and flow_direction must have same dimensions
- Same pixel size (`dx`)
- Same coordinate system

## See Also

- Main package README: `../README.md`
- Example workflow: `examples/dem_to_inversion_workflow.py`
- Function documentation: `fluvial_inversion/prepare_inversion_data.py`
- MATLAB validation: `tests/README_MATLAB_VALIDATION.md`

---

**Date:** November 12, 2025
**Status:** ✅ TESTED AND DOCUMENTED
