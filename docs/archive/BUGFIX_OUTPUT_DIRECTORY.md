# Bug Fix: Output Directory Creation

## Issue

When running demo scripts, the HTML report generation failed with:

```
FileNotFoundError: [Errno 2] No such file or directory: 'output\isothermal_stability_report.html'
```

## Root Cause

The `generate_isothermal_report()` function attempted to write to a file path without ensuring the parent directory exists. While the demo scripts tried to create the `output/` directory, they used relative paths which could fail depending on the working directory.

## Solution

### 1. Robust Directory Creation in Reporting Module

**File**: `akts/reporting.py`

Added automatic directory creation before writing the report:

```python
# Save or return
if report_path:
    report_path = Path(report_path)
    # Create parent directory if it doesn't exist
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return str(report_path)
```

**Benefits**:
- Works regardless of working directory
- Creates nested directories if needed (`parents=True`)
- Idempotent (`exist_ok=True`)
- Makes the function more robust

### 2. Absolute Paths in Demo Scripts

**Files**:
- `Examples/Isothermal/demo_auto_model.py`
- `Examples/Isothermal/demo_auto_model_json.py`

Changed from relative paths to absolute paths:

**Before**:
```python
report_path='output/isothermal_stability_report.html'
```

**After**:
```python
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
...
report_path=output_dir / 'isothermal_stability_report.html'
```

**Benefits**:
- Works regardless of where script is called from
- Explicit and clear
- IDE-friendly (shows full path)

## Testing

### Test Case 1: Run from Different Directory

```bash
# Before fix: Failed
cd C:\
python C:\Programming\akts-main\Examples\Isothermal\demo_auto_model.py
# Error: FileNotFoundError

# After fix: Works
cd C:\
python C:\Programming\akts-main\Examples\Isothermal\demo_auto_model.py
# Success: Report saved to C:\Programming\akts-main\Examples\Isothermal\output\isothermal_stability_report.html
```

### Test Case 2: Run from Script Directory

```bash
# Before and after: Works
cd C:\Programming\akts-main\Examples\Isothermal
python demo_auto_model.py
# Success: Report saved to output\isothermal_stability_report.html
```

### Test Case 3: Nested Directories

```python
# Now supports nested paths automatically
auto_model_isothermal_data(
    ...,
    report_path='reports/2024/january/analysis.html'
)
# Creates: reports/2024/january/ directory structure
```

## Impact

### Before Fix
-  Required running scripts from specific directory
-  Failed with nested output paths
-  Poor user experience

### After Fix
-  Works from any directory
-  Creates nested directories automatically
- Robust directory handling
-  No breaking changes (backward compatible)

## Files Modified

1. `akts/reporting.py` - Added directory creation
2. `Examples/Isothermal/demo_auto_model.py` - Absolute paths
3. `Examples/Isothermal/demo_auto_model_json.py` - Absolute paths

## Backward Compatibility

 **Fully backward compatible**

Existing code continues to work:
- Absolute paths: Still work
- Relative paths: Now work from any directory
- Existing directories: Still work (idempotent)

## Recommendation for Users

When calling `auto_model_isothermal_data()` or `generate_isothermal_report()`:

**Good** (explicit, absolute):
```python
from pathlib import Path

output_dir = Path('my_reports')
output_dir.mkdir(exist_ok=True)

results = auto_model_isothermal_data(
    ...,
    report_path=output_dir / 'analysis.html'
)
```

**Also works** (relative, handled automatically):
```python
results = auto_model_isothermal_data(
    ...,
    report_path='reports/analysis.html'  # Directory created automatically
)
```

**Best practice** (full path):
```python
from pathlib import Path

report_path = Path.cwd() / 'output' / 'report.html'

results = auto_model_isothermal_data(
    ...,
    report_path=report_path
)
```

## Summary

The reporting function creates missing parent directories before writing the output file.
