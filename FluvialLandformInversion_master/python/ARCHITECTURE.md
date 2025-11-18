# Python Architecture for Fluvial Landform Inversion

## Design Philosophy

1. **Modern Python libraries**: NumPy, SciPy, Matplotlib
2. **Modular design**: Each function is a standalone module
3. **Type hints**: Full type annotations for clarity
4. **Documentation**: NumPy-style docstrings
5. **Testing**: Comprehensive unit and integration tests
6. **Performance**: Vectorized operations, optional Numba JIT compilation
7. **MATLAB compatibility**: Exact numerical equivalence verified

## Dependency Graph

```
Level 0 (No dependencies):
├── calculate_chi.py
├── findm_slope_area.py
└── invert_block_uplift.py

Level 1 (Depends on Level 0):
├── findm_linear_chi.py (uses calculate_chi)
└── findm_collapse_chi.py (uses calculate_chi)

Level 2 (Depends on Level 0-1):
├── invert_parabola.py (uses flow network concepts)
├── invert_with_different_gamma.py (uses invert_block_uplift)
├── calibrate_k_total_uplift.py (uses inversion outputs)
└── bootstrap_invert_block_uplift.py (uses invert_block_uplift)
```

## Module Structure

```
python/
├── fluvial_inversion/
│   ├── __init__.py
│   ├── calculate_chi.py
│   ├── findm_slope_area.py
│   ├── findm_linear_chi.py
│   ├── findm_collapse_chi.py
│   ├── invert_block_uplift.py
│   ├── invert_parabola.py
│   ├── invert_with_different_gamma.py
│   ├── calibrate_k_total_uplift.py
│   ├── bootstrap_invert_block_uplift.py
│   └── utils.py (plotting, I/O helpers)
├── tests/
│   ├── test_calculate_chi.py
│   ├── test_findm_functions.py
│   ├── test_invert_block_uplift.py
│   ├── test_invert_parabola.py
│   ├── test_calibration.py
│   ├── test_bootstrap.py
│   └── test_matlab_equivalence.py
├── matlab_validation/
│   ├── generate_reference_results.m
│   └── reference_results.npz
├── examples/
│   ├── example_block_uplift.py
│   └── example_parabola.py
├── setup.py
├── requirements.txt
└── README.md
```

## Key Libraries

- **NumPy**: Core numerical operations, array manipulation
- **SciPy**: Linear algebra (solve, lstsq), optimization, statistics
- **Matplotlib**: Plotting and visualization
- **pytest**: Testing framework
- **scipy.io**: Reading MATLAB .mat files

## Implementation Strategy

1. **Phase 1**: Core utilities (calculate_chi, findm functions)
2. **Phase 2**: Inversion methods (block uplift, parabola)
3. **Phase 3**: Analysis tools (gamma selection, calibration, bootstrap)
4. **Phase 4**: MATLAB cross-validation
5. **Phase 5**: Documentation and examples

## Testing Strategy

### Unit Tests
- Test each function with synthetic data
- Verify edge cases and error handling
- Check numerical stability

### Integration Tests
- Test complete workflows
- Verify interdependent functions work together

### MATLAB Equivalence Tests
- Load MATLAB reference results
- Compare Python outputs with tight tolerances
- Test with BlockUpliftDataHighRes.mat and ParabolaDataHighRes.mat

## Performance Considerations

- Vectorized operations wherever possible
- Avoid explicit loops (use NumPy broadcasting)
- Profile critical sections
- Consider Numba JIT for hot paths if needed
- Memory-efficient algorithms for large datasets
