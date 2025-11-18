# Misfit Formula Correction

## Issue Discovered

During MATLAB-Python cross-validation, uplift rates matched perfectly but misfit values differed substantially.

## Root Cause

The Python implementation used the **standard RMSE formula**:
```python
misfit = sqrt(sum(residuals²) / (N-q))
```

But MATLAB uses a **non-standard formula**:
```matlab
Misfit = sqrt(sum(residuals²)) / (N-q)
```

The difference is **where the division occurs relative to the square root**.

## Mathematical Impact

For typical values (e.g., `sum(residuals²) = 45,000`, `N-q = 7,789`):

- **Old Python (RMSE)**: `sqrt(45,000 / 7,789) = 2.40`
- **MATLAB formula**: `sqrt(45,000) / 7,789 = 0.027`

This creates an **~89x difference**!

## Why MATLAB's Formula is Unusual

The standard root mean square error (RMSE) is:
```
RMSE = sqrt(mean(residuals²)) = sqrt(sum(residuals²) / N)
```

MATLAB's formula is:
```
Misfit = sqrt(sum(residuals²)) / (N-q) = RMS / DOF
```

This is the **root sum of squares divided by degrees of freedom**, not a standard statistical metric.

## Solution

Changed Python implementation to **exactly match MATLAB**:

### `invert_block_uplift.py` (line 168):
```python
# OLD (Standard RMSE):
misfit = np.sqrt(np.sum(residuals**2) / (N - q))

# NEW (Matches MATLAB):
misfit = np.sqrt(np.sum(residuals**2)) / (N - q)
```

### `invert_parabola.py` (line 221):
```python
# OLD:
misfit = np.sqrt(np.sum(residuals**2) / (N - 3*q))

# NEW (Matches MATLAB):
misfit = np.sqrt(np.sum(residuals**2)) / (N - 3*q)
```

## Validation

After fix, Python misfit values now match MATLAB:
- **Before**: Python misfit = 23.81 m
- **After**: Python misfit = 0.27 m (matches MATLAB)

All unit tests still pass (30/30).

## Documentation

Added clarification to docstrings:
```python
misfit : float
    Misfit between data and model topography [L].
    Calculated as: sqrt(sum(residuals²)) / (N-q)
    Note: This differs from standard RMSE to match MATLAB implementation.
```

## Lesson Learned

When porting code, **always verify not just that formulas look similar, but that they are mathematically identical**, paying careful attention to:
- Order of operations
- Placement of parentheses
- Division before vs. after square roots
- Any non-standard statistical formulas

---

**Date**: November 12, 2025
**Status**: ✅ FIXED AND VALIDATED
**Credit**: Bug discovered by user during cross-validation testing
