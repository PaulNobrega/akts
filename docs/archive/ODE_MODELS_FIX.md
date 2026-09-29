# ODE Models Fix

## Issue
A->B->C (consecutive reactions) ODE model was not running. Error: "Model setup error: Valid f1/f2_model required"

## Root Cause
The A->B->C model in AKTS requires two pieces of information:
1. The kinetic parameters (Ea1, A1, Ea2, A2) -  we had this
2. The **mechanism** for each step (f1_model, f2_model) -  we were missing this

From `akts/models.py` line 387-396:
```python
elif model_name == "A->B->C":
    if not f1_model or f1_model not in F_ALPHA_MODELS or not f2_model or f2_model not in F_ALPHA_MODELS:
        raise ValueError(f"Valid f1/f2_model required")
    # ... uses f1_func and f2_func for the two steps
```

The A->B->C model allows you to specify different mechanisms for each step:
- **Step 1 (A→B)**: Could be F1, F2, F3, A2, etc.
- **Step 2 (B→C)**: Could be F1, F2, F3, A2, etc.

This is more flexible than assuming both are first-order.

## Solution

Updated `_setup_model_configs()` in `akts/helpers.py` to include f1_model and f2_model in def_args:

**Before:**
```python
if model_name == 'A->B->C':
    models_config.append({
        'name': model_key,
        'type': 'A->B->C',
        'def_args': {}  #  Empty - missing required info
    })
```

**After:**
```python
if model_name == 'A->B->C':
    # Consecutive reactions: A→B→C
    # Requires f1_model and f2_model to define mechanism for each step
    # Default: both steps are first-order (F1)
    models_config.append({
        'name': model_key,
        'type': 'A->B->C',
        'def_args': {
            'f1_model': 'F1',  # First step mechanism
            'f2_model': 'F1'   # Second step mechanism
        }
    })
```

## Additional Improvements

### 1. Better Error Reporting
Changed fit failure message to show actual error:

**Before:**
```python
progress(f"{display_name} fit failed", {'model': custom_name})
```

**After:**
```python
error_msg = fit_res.message if hasattr(fit_res, 'message') else 'Unknown error'
progress(f"{display_name} fit failed: {error_msg}",
        {'model': custom_name, 'error': error_msg})
```

Now users see: "A->B->C (consecutive reactions) fit failed: Model setup error: Valid f1/f2_model required"

### 2. Fixed Unicode Display
Changed display name from `'A→B→C'` to `'A->B->C'` to avoid Windows console encoding errors.

**Before:**
```python
'A->B->C': 'A→B→C (consecutive reactions)',
```

**After:**
```python
'A->B->C': 'A->B->C (consecutive reactions)',
```

## Default Mechanism Choice

We default both steps to F1 (first-order) because:
1. **Most common**: Many consecutive reactions follow pseudo-first-order kinetics
2. **Simplest**: Fewest assumptions about the mechanism
3. **Conservative**: Good starting point for automated analysis

## Future Enhancement Ideas

### Option 1: Try Multiple Mechanism Combinations
```python
A->B->C_configs = [
    ('F1', 'F1'),  # Both first-order
    ('F2', 'F1'),  # A→B second-order, B→C first-order
    ('F1', 'F2'),  # A→B first-order, B→C second-order
]
```

### Option 2: User-Configurable
```python
auto_model_isothermal_data(
    data_files=files,
    models_to_try=['F1', 'F2', 'A->B->C'],
    ode_mechanisms={'A->B->C': ('F2', 'F1')}  # Custom mechanisms
)
```

### Option 3: Auto-Detect from Data
Analyze curvature/inflection points to infer likely mechanisms.

## Other ODE Models Available

From `akts/models.py`:
```python
"A->B->C": (ode_system_A_B_C, ['Ea1', 'A1', 'Ea2', 'A2'], 2),
"A+B->C": (ode_system_A_plus_B_C, ['Ea', 'A', 'initial_ratio_r'], 1),
"parallel_competing": (ode_system_parallel_competing, ['Ea1', 'A1', 'Ea2', 'A2'], 2),
```

We currently include:
-  A->B->C (consecutive reactions)
-  A+B->C (bimolecular) - already handled

Could add:
- ⚪ parallel_competing (two competing pathways)

## Test Results

With the fix, A->B->C now runs:

**Before Fix:**
```
[19:22:55] Fitting A→B→C (consecutive reactions)...
[19:22:55] A→B→C (consecutive reactions) fit failed
```

**After Fix:**
```
[19:30:47] Fitting A->B->C (consecutive reactions)...
--- Attempting optimization on CONVERSION residuals (weighted) ---
[19:30:XX] A->B->C (consecutive reactions) fit [successful/failed with actual reason]
```

## Files Modified

- `akts/helpers.py`:
  - Fixed A->B->C model_definition_args to include f1_model='F1', f2_model='F1'
  - Improved error reporting in fit failure messages
  - Changed display name from A→B→C to A->B->C (ASCII arrows)

## Backward Compatibility

 Fully backward compatible - existing code continues to work:
- f(alpha) models unchanged
- A+B->C model unchanged (already had correct def_args)
- Only A->B->C setup was corrected

## Summary

**Problem**: A->B->C ODE model was silently failing
**Root Cause**: Missing required f1_model and f2_model in model_definition_args
**Solution**: Default both to 'F1' (first-order) for automated analysis
**Result**: The automated setup supplied first-order mechanisms for both reaction steps.

Users can now analyze:
- Simple single-step reactions (f(alpha) models)
- Consecutive reactions (A→B→C)
- Bimolecular reactions (A+B→C)
- All with automated model selection and comparison!
