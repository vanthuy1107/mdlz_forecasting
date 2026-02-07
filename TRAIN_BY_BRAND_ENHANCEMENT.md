# train_by_brand.py Enhancement

## Change Summary

Made `--category` argument **optional** in `train_by_brand.py`. The script now supports two modes:

### Mode 1: Auto-detect from config.yaml (NEW)
```bash
python train_by_brand.py
```

**Behavior:**
- Reads `major_categories` from `config/config.yaml`
- Trains brand-level models for **all** major categories listed
- Example: If `config.yaml` has `major_categories: ["DRY"]`, it will train all DRY brands

### Mode 2: Explicit category (EXISTING)
```bash
python train_by_brand.py --category DRY
```

**Behavior:**
- Trains brand-level models for the specified category only
- Same as before, but now `--category` is optional

## Implementation Details

### Changes Made to `train_by_brand.py`:

1. **Made --category optional:**
   ```python
   parser.add_argument(
       '--category',
       type=str,
       required=False,  # Changed from True
       default=None,
       help='Category to train brands for (e.g., DRY, FRESH, TET). If not specified, reads major_categories from config.yaml'
   )
   ```

2. **Added config.yaml reading logic:**
   - If `args.category is None`, loads `config.yaml`
   - Extracts `major_categories` list
   - Trains each category in sequence

3. **Extracted training logic into `train_category()` function:**
   - Original training code moved to `train_category(args)`
   - Can be called for single category or looped for multiple

4. **Added overall summary for multi-category mode:**
   - Shows success/failure for each category
   - Aggregates results across all categories

## Usage Examples

### Train all categories from config.yaml:
```bash
python train_by_brand.py
```

Output:
```
[INFO] No --category specified, reading from config.yaml...
[INFO] Found major_categories: ['DRY']

================================================================================
CATEGORY: DRY
================================================================================

[1/5] Loading configuration for category 'DRY'...
[2/5] Loading data...
[3/5] Discovering brands in category 'DRY'...
  - Found 15 brand(s) in DRY:
    1. AFC (25,432 samples)
    2. COSY (18,765 samples)
    ...
[4/5] Will train 15 brand model(s)
[5/5] Training brand models...
...
```

### Train specific category:
```bash
python train_by_brand.py --category DRY
```

### Train specific brands in a category:
```bash
python train_by_brand.py --category DRY --brands AFC COSY OREO
```

### Train with auto-detect, skip existing models:
```bash
python train_by_brand.py --skip-existing
```

## Benefits

1. **Convenience**: No need to specify category if you want to train all
2. **Consistency**: Reads from same config.yaml used by other scripts
3. **Flexibility**: Still supports explicit --category for selective training
4. **CI/CD Ready**: Can run `python train_by_brand.py` in automation scripts

## Related Files

- `config/config.yaml`: Contains `major_categories` list (line 23-27)
- `train_by_brand.py`: Main training script (modified)

## Testing

To test the changes:

1. **Check help message:**
   ```bash
   python train_by_brand.py --help
   ```

2. **Test auto-detection:**
   ```bash
   python train_by_brand.py
   ```

3. **Test explicit category:**
   ```bash
   python train_by_brand.py --category DRY
   ```

---

**Date**: 2026-02-07
**Issue**: `train_by_brand.py` required explicit `--category` flag
**Solution**: Made `--category` optional, auto-reads from `config.yaml` if not specified
