# Fluvial Landform Inversion

A toolkit for linear inversion of river longitudinal profiles to infer tectonic uplift rate history from fluvial topography. This implementation is based on the stream power incision model and chi analysis.

**Author:** Liran Goren (gorenl@bgu.ac.il)
**Original Date:** July 11, 2019
**Python Port:** November 2025

## 🚀 **NEW: Complete Python Implementation Available!**

This repository now includes a **fully-tested, production-ready Python port** of the original MATLAB toolkit:

- ✅ **Modern Libraries**: NumPy, SciPy, Matplotlib
- ✅ **Comprehensive Tests**: 30+ unit tests, all passing
- ✅ **MATLAB-Validated**: Results verified against original implementation
- ✅ **Well-Documented**: NumPy-style docstrings, type hints
- ✅ **Production-Ready**: Complete with setup.py, requirements.txt

**See `python/README.md` for Python documentation and quick start guide.**

---

## Overview

This toolkit provides methods to invert fluvial topography to reconstruct the history of rock uplift rates that shaped the landscape. The inversion is based on the chi (χ) transformation of river profiles, which assumes the stream power incision model with n=1. The code supports both spatially uniform (block) uplift and spatially varying uplift patterns.

### Key Features

- **Multiple methods for determining concavity index (m)**
- **Block uplift inversion** for spatially uniform uplift histories
- **Parabolic uplift inversion** for spatially varying uplift patterns
- **Bootstrap analysis** for uncertainty quantification
- **L-curve analysis** for selecting regularization parameters
- **Calibration tools** for determining erodibility coefficients

---

## Project Structure

### Core Utilities

#### `CalculateChi.m`
Calculates the chi (χ) coordinate for river pixels in a fluvial network.

**Input:**
- `x, y`: Coordinate vectors [L] for each pixel
- `rec_array`: Receiver relationships defining flow network
- `area_array`: Upstream drainage area [L²]
- `m`: Area exponent in stream power law

**Output:**
- `chi`: Chi values [L] for each pixel

---

### Concavity Index (m) Determination

Three methods are provided to estimate the concavity index:

#### `FindmSlopeArea.m`
Determines m from log-log slope-area relationships using linear regression.

**Input:**
- `slope_array`: Channel slope values [L/L]
- `area_array`: Drainage area values [L²]

**Output:**
- `m`: Best-fit concavity index
- `lower_bound, upper_bound`: 95% confidence intervals
- `R2`: Coefficient of determination

#### `FindmLinearChi.m`
Finds m that produces the most linear river profiles in chi-z space (maximizes R²).

**Input:**
- `x, y, z`: Coordinate vectors [L]
- `rec_array`: Flow network receiver relationships
- `area_array`: Drainage area [L²]

**Output:**
- `m`: Optimal concavity index
- `chi`: Associated chi values

**Method:** Tests range of m values (0.1 to 0.95), fits linear model through origin.

#### `FindmCollapseChi.m`
Finds m that best collapses tributaries onto a single curve in chi-z space.

**Input:**
- `x, y, z`: Coordinate vectors [L]
- `rec_array`: Flow network receiver relationships
- `area_array`: Drainage area [L²]

**Output:**
- `m`: Concavity index that minimizes scatter
- `chi`: Associated chi values

**Method:** Minimizes mean standard deviation across 25 chi bins.

---

### Inversion Methods

#### `InvertBlockUplift.m`
Main inversion function for spatially uniform (block) uplift histories.

**Input:**
- `chi`: Chi coordinate vector [L]
- `z`: Elevation data [L]
- `Gamma`: Damping/regularization coefficient
- `q`: Number of time intervals
- `to_plot`: Plot flag (0=no plot, 1=plot results)

**Output:**
- `Ustar`: Non-dimensional uplift rate history (q×1 vector)
- `tstar`: Scaled time boundaries (q+1×1 vector)
- `Misfit`: RMS misfit between data and model

**Key Features:**
- Uses Tikhonov regularization with damping coefficient Gamma
- Time intervals contain equal numbers of data points
- Can produce dimensional or non-dimensional results depending on input

#### `InvertParabola.m`
Inversion for spatially varying uplift with parabolic pattern.

**Input:**
- `chi`: Chi coordinate vector [L]
- `z`: Elevation data [L]
- `x`: Along-strike coordinate [L]
- `rec_array`: Flow network receiver relationships
- `Gamma`: Damping coefficient
- `q`: Number of time intervals
- `K`: Erodibility coefficient [L^(1-2m)/T]
- `to_plot`: Plot flag

**Output:**
- `Up`: Parabolic uplift parameters (3q×1 vector)
- `tstar`: Scaled time boundaries
- `Misfit`: RMS misfit

**Method:** Assumes U(x,t) = a(t)·x² + b(t)·x + c(t) where x is in km.

#### `InvertWithDifferentGamma.m`
Generates L-curve for selecting optimal damping coefficient.

**Input:**
- `chi`: Chi coordinate vector [L]
- `z`: Elevation data [L]
- `q`: Number of time intervals

**Output:**
- L-curve plot of misfit vs. 1/Gamma

**Usage:** Run to visualize trade-off between model roughness and data fit. Choose Gamma at the "elbow" of the L-curve.

---

### Calibration and Analysis

#### `CalibrateKTotalUplift.m`
Calibrates erodibility coefficient K from total uplift observations.

**Input:**
- `H`: Total uplift [L] from present to time t_H
- `t_H`: Age [T] of uplifted feature
- `Ustar`: Non-dimensional uplift rate history (from inversion)
- `tstar`: Scaled time boundaries
- `A0`: Reference drainage area (typically 1)
- `m`: Concavity index
- `to_plot`: Plot flag

**Output:**
- `K`: Erodibility coefficient [L^(1-2m)/T]
- `U`: Dimensional uplift rate history [L/T]
- `t`: Dimensional time boundaries [T]

#### `BootstrapInvertBlockUplift.m`
Performs bootstrap analysis for uncertainty quantification.

**Input:**
- `chi`: Chi coordinate vector [L]
- `z`: Elevation data [L]
- `Gamma`: Damping coefficient
- `q`: Number of time intervals
- `percent_sample`: Fraction of data to sample (0-1)
- `num_iterations`: Number of bootstrap iterations
- `K`: Erodibility coefficient for plotting

**Output:**
- `Ustar_mat`: Matrix of inversion results (num_iterations × q)
- `tstar_best`: Time boundaries from full dataset

**Visualization:** Plots all bootstrap realizations (gray), mean solution (magenta), ±1σ bounds (dashed magenta), and best-fit solution (black).

---

## Example Datasets

### `BlockUpliftDataHighRes.mat`
High-resolution synthetic data for testing block uplift inversion. Contains vectors for chi, elevation, and other fluvial network properties assuming spatially uniform uplift.

### `ParabolaDataHighRes.mat`
High-resolution synthetic data for testing parabolic uplift inversion. Contains spatial coordinates and network topology for spatially varying uplift scenarios.

---

## Typical Workflow

### 1. Prepare Fluvial Network Data
Extract river network with coordinates (x, y, z), drainage areas, and flow routing from DEM.

### 2. Determine Concavity Index
Choose one of three methods:
```matlab
% Method 1: Slope-area relationship
[m, lb, ub, R2] = FindmSlopeArea(slope_array, area_array);

% Method 2: Linear chi-z fit
[m, chi] = FindmLinearChi(x, y, z, rec_array, area_array);

% Method 3: Tributary collapse
[m, chi] = FindmCollapseChi(x, y, z, rec_array, area_array);
```

### 3. Calculate Chi Coordinates
```matlab
chi = CalculateChi(x, y, rec_array, area_array, m);
```

### 4. Select Damping Coefficient
```matlab
InvertWithDifferentGamma(chi, z, q);
% Visually inspect L-curve and choose Gamma
```

### 5. Perform Inversion
```matlab
% For block uplift
[Ustar, tstar, Misfit] = InvertBlockUplift(chi, z, Gamma, q, 1);

% Or for spatially varying uplift
[Up, tstar, Misfit] = InvertParabola(chi, z, x, rec_array, Gamma, q, K, 1);
```

### 6. Calibrate to Natural Units (Optional)
```matlab
[K, U, t] = CalibrateKTotalUplift(H, t_H, Ustar, tstar, A0, m, 1);
```

### 7. Assess Uncertainty (Optional)
```matlab
[Ustar_mat, tstar_best] = BootstrapInvertBlockUplift(chi, z, Gamma, q, 0.8, 100, K);
```

---

## Theoretical Background

### Stream Power Model
The code assumes the stream power incision model with n=1:

```
dz/dt = U(x,t) - K·A^m·|∇z|
```

Where:
- `z`: elevation
- `t`: time
- `U`: rock uplift rate
- `K`: erodibility coefficient
- `A`: drainage area
- `m`: concavity index
- `∇z`: channel gradient

### Chi Transformation
Chi (χ) is defined as:

```
χ = ∫ (A₀/A(x))^m dx
```

This transformation linearizes steady-state river profiles, making them amenable to inverse methods.

### Inverse Problem
The inversion solves for uplift rate history U(t) given observed topography z(χ) using:

```
z(χ) = ∫₀^χ U(t(χ')) dχ'
```

Regularization (Tikhonov) is used to stabilize the inverse problem.

---

## Important Notes

1. **Time Intervals:** The code divides time into q intervals, each resolved by equal numbers of data points (not equal time spans).

2. **Non-dimensional vs. Dimensional:**
   - Using chi as input produces non-dimensional results (U*, t*)
   - Use `CalibrateKTotalUplift` to convert to dimensional units

3. **Regularization:** The damping parameter Gamma controls the smoothness of the solution. Larger Gamma = smoother uplift history but larger misfit.

4. **Reference Area:** A₀ is typically set to 1 for non-dimensional analysis.

5. **Assumptions:**
   - n = 1 in stream power law
   - Channels are in topographic steady state or slowly evolving
   - Uniform erodibility (K) and concavity (m) across basin

---

## Requirements

- MATLAB (tested with version from 2019)
- Curve Fitting Toolbox (for `fit` function in FindmSlopeArea.m and FindmLinearChi.m)

---

## Citation

If you use this code, please cite the original research by Liran Goren.

---

## Contact

For questions or issues, contact: gorenl@bgu.ac.il
