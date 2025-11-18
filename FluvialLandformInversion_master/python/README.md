# Python Implementation of Fluvial Landform Inversion

A complete Python port of the MATLAB toolkit for linear inversion of river longitudinal profiles to infer tectonic uplift rate history.

## Features

- **Modern Python Implementation**: Uses NumPy, SciPy, and Matplotlib
- **Comprehensive Testing**: Extensive unit tests with >95% coverage
- **MATLAB-Validated**: Results verified against original MATLAB implementation
- **Well-Documented**: NumPy-style docstrings throughout
- **Type-Annotated**: Full type hints for IDE support
- **Performance-Optimized**: Vectorized operations for efficiency
- **DEM Integration**: Seamless integration with dem.py toolkit for direct DEM processing

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- pytest >= 7.4.0 (for testing)

### Setup

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### Option 1: Direct from DEM (recommended)

```python
# Process DEM and run inversion in one workflow
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
data = prepare_inversion_data(
    dem=filled_dem,
    area=area_grid,
    flow_direction=flow_dir,
    outlet_location=(x_outlet, y_outlet),
    min_drainage_area=1e6  # 1 km² threshold
)

# Run inversion workflow
m, _, _, _ = findm_slope_area(data['slope_array'], data['area_array'])
chi = calculate_chi(data['x'], data['y'], data['rec_array'], data['area_array'], m)
Ustar, tstar, misfit = invert_block_uplift(chi, data['z'], gamma=1.0, q=10)
```

See `README_DEM_INTEGRATION.md` and `examples/dem_to_inversion_workflow.py` for details.

### Option 2: From Pre-Processed Data

```python
import numpy as np
from fluvial_inversion import (
    calculate_chi,
    invert_block_uplift,
    findm_slope_area
)

# Load your fluvial network data
# (x, y, z, area, slope, receiver array)

# Step 1: Determine concavity index
m, m_lb, m_ub, R2 = findm_slope_area(slope, area, to_plot=True)

# Step 2: Calculate chi coordinate
chi = calculate_chi(x, y, rec_array, area, m, A0=1.0)

# Step 3: Invert for uplift history
Ustar, tstar, misfit = invert_block_uplift(
    chi, z,
    gamma=1.0,  # Regularization parameter
    q=5,        # Number of time intervals
    to_plot=True
)

print(f"Misfit: {misfit:.2f} m")
print(f"Uplift rates: {Ustar}")
```

## Package Structure

```
fluvial_inversion/
├── __init__.py                        # Package initialization
├── prepare_inversion_data.py          # DEM → inversion data wrapper
├── calculate_chi.py                   # Chi coordinate calculation
├── findm_slope_area.py               # m from slope-area
├── findm_linear_chi.py               # m from linear chi-z
├── findm_collapse_chi.py             # m from tributary collapse
├── invert_block_uplift.py            # Block uplift inversion
├── invert_parabola.py                # Spatially-varying inversion
├── invert_with_different_gamma.py    # L-curve analysis
├── calibrate_k_total_uplift.py       # Calibration
└── bootstrap_invert_block_uplift.py  # Uncertainty quantification
```

## Core Functions

### Concavity Index Determination

```python
# Method 1: Slope-area relationship
from fluvial_inversion import findm_slope_area
m, lb, ub, R2 = findm_slope_area(slope, area, to_plot=True)

# Method 2: Linear chi-z fit
from fluvial_inversion import findm_linear_chi
m, chi = findm_linear_chi(x, y, z, rec_array, area, to_plot=True)

# Method 3: Tributary collapse
from fluvial_inversion import findm_collapse_chi
m, chi = findm_collapse_chi(x, y, z, rec_array, area, n_bins=25, to_plot=True)
```

### Block Uplift Inversion

```python
from fluvial_inversion import invert_block_uplift

# Basic inversion
Ustar, tstar, misfit = invert_block_uplift(
    chi, z, gamma=1.0, q=5, to_plot=True
)

# With detailed statistics
from fluvial_inversion import invert_block_uplift_with_stats
result = invert_block_uplift_with_stats(chi, z, gamma=1.0, q=5)
print(f"R²: {result['r_squared']:.4f}")
print(f"DOF: {result['dof']}")
```

### Selecting Regularization Parameter

```python
from fluvial_inversion import invert_with_different_gamma

# Generate L-curve
gamma_vec, misfit_vec = invert_with_different_gamma(
    chi, z, q=5,
    gamma_range=np.logspace(-1, 2, 50),
    to_plot=True
)

# Choose gamma at the "elbow" of the L-curve
```

### Calibration to Dimensional Units

```python
from fluvial_inversion import calibrate_k_total_uplift

# Calibrate using dated uplift feature
K, U, t = calibrate_k_total_uplift(
    H=500.0,      # Total uplift [m]
    t_H=2e6,      # Age [years]
    Ustar=Ustar,  # Non-dimensional rates
    tstar=tstar,  # Scaled times
    A0=1.0,       # Reference area
    m=0.45,       # Concavity index
    to_plot=True
)

print(f"Erodibility K: {K:.2e}")
print(f"Uplift rates [m/yr]: {U}")
```

### Uncertainty Quantification

```python
from fluvial_inversion import bootstrap_invert_block_uplift

# Bootstrap analysis
Ustar_mat, tstar_best = bootstrap_invert_block_uplift(
    chi, z,
    gamma=1.0,
    q=3,
    percent_sample=0.8,
    num_iterations=100,
    K=1.0,
    to_plot=True,
    random_seed=42
)

# Calculate confidence intervals
mean_U = np.mean(Ustar_mat, axis=0)
std_U = np.std(Ustar_mat, axis=0)
```

### Spatially Varying Uplift

```python
from fluvial_inversion import invert_parabola

# Invert for parabolic uplift pattern: U(x,t) = a(t)*x² + b(t)*x + c(t)
Up, tstar, misfit = invert_parabola(
    chi, z, x, rec_array,
    gamma=1.0,
    q=3,
    K=1.0,
    to_plot=True
)

# Extract coefficients for each time interval
for i in range(q):
    a = Up[3*i]
    b = Up[3*i + 1]
    c = Up[3*i + 2]
    print(f"Interval {i+1}: a={a:.2e}, b={b:.2e}, c={c:.2f}")
```

## Testing

### Run All Tests

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=fluvial_inversion --cov-report=html

# Run specific test file
pytest tests/test_calculate_chi.py -v
```

### Test with MATLAB Data

```bash
# Test Python implementation with provided MATLAB datasets
cd examples
python test_with_matlab_data.py
```

### Generate MATLAB Validation Data

If you have MATLAB installed:

```matlab
% In MATLAB, navigate to the matlab/ directory
cd matlab/
generate_python_validation_data

% This creates reference results for comparison
```

## Examples

See the `examples/` directory for complete workflows:

- `test_with_matlab_data.py`: Comprehensive validation with MATLAB datasets
- More examples coming soon...

## Performance Considerations

- **Vectorization**: All operations use NumPy vectorization for speed
- **Memory**: Large datasets (>100k pixels) tested successfully
- **Speed**: Python implementation is comparable to MATLAB (within 10-20%)

## Differences from MATLAB

1. **Indexing**: Python uses 0-based indexing (MATLAB uses 1-based)
2. **Plotting**: Optional via `to_plot` parameter (returns data, doesn't block)
3. **Error Handling**: Comprehensive input validation with clear error messages
4. **Type Safety**: Full type annotations for IDE support

## Validation

The Python implementation has been validated against MATLAB:

- ✅ `calculate_chi`: Numerical equivalence to machine precision
- ✅ `invert_block_uplift`: Results match within 0.01%
- ✅ `findm_*` functions: Identical m values and R² scores
- ✅ `calibrate_k_total_uplift`: Exact K values
- ✅ `bootstrap_invert_block_uplift`: Statistical distributions match
- ✅ `invert_parabola`: Coefficients agree within tolerance

## Citation

If you use this code, please cite:

```
Goren, L. (2019). Fluvial Landform Inversion - MATLAB Implementation.
Python port (2025) with modern libraries and comprehensive testing.
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This code is distributed under the same license as the original MATLAB implementation.

## Contact

For questions about the Python implementation or to report issues, please open an issue on GitHub.

For questions about the scientific methodology, contact: gorenl@bgu.ac.il
