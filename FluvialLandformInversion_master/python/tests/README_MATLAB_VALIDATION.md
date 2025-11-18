# MATLAB Validation Testing

This directory contains tests that compare Python outputs with MATLAB reference results.

## Setup

### Step 1: Generate MATLAB Reference Data

On your machine with MATLAB installed:

```matlab
% Open MATLAB
cd /path/to/FluvialLandformInversion/matlab/

% Run the validation script
generate_python_validation_data

% This creates two files:
%   - matlab_validation_block_uplift.mat
%   - matlab_validation_parabola.mat
```

### Step 2: Run Python Validation Tests

Back on this machine:

```bash
cd /path/to/FluvialLandformInversion/python

# Activate virtual environment
source venv/bin/activate

# Run the equivalence tests
pytest tests/test_matlab_equivalence.py -v

# Or run all tests including MATLAB validation
pytest tests/ -v
```

## What Gets Tested

The `test_matlab_equivalence.py` script compares:

1. **FindmSlopeArea**: Concavity index m and R² values
2. **CalculateChi**: Chi coordinate values (should match to machine precision)
3. **InvertBlockUplift**: Uplift rates, time boundaries, and misfit
4. **CalibrateKTotalUplift**: Erodibility K and dimensional rates
5. **BootstrapInvertBlockUplift**: Statistical distributions (means, std devs)

## Expected Results

- **Chi values**: Should match to ~1e-10 (machine precision)
- **Concavity m**: Should match to ~0.01% relative error
- **Uplift rates**: Should match to ~0.1% relative error
- **Erodibility K**: Should match to machine precision
- **Bootstrap stats**: Should be similar (not exact due to different RNGs)

## If Tests Fail

1. **File not found errors**: Make sure you ran the MATLAB script first
2. **Numerical differences**: Small differences (<1%) are expected due to:
   - Different linear algebra libraries (MATLAB vs NumPy/LAPACK)
   - Different floating point implementations
   - Round-off accumulation in iterative algorithms
3. **Bootstrap differences**: Expected - MATLAB and Python use different random number generators

## Skipping MATLAB Tests

If you don't have MATLAB or haven't generated reference data yet, these tests will automatically skip with a message:

```
SKIPPED [1] test_matlab_equivalence.py:XX: MATLAB validation data not found.
Run matlab/generate_python_validation_data.m first.
```

This is normal and expected. The other tests in the suite don't require MATLAB.
