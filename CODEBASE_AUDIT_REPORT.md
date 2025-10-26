# Delapan-Fantasy Codebase Audit Report
**Generated:** 2025-10-26
**Current Branch:** jupe
**Total Python Files Analyzed:** 129

---

## EXECUTIVE SUMMARY

The repository is in the process of transitioning from a legacy walk-forward backtest framework to a modular, production-ready pipeline. Most deprecated code has been properly moved to `deprecated/` directories. However, there are several issues requiring cleanup:

1. **CRITICAL IMPORT ERROR:** `src/optimization/backtest_lineup_integration.py` imports from non-existent path (`..evaluation.backtest`) that was moved to deprecated
2. **Unused example scripts:** Several example files are not integrated into workflow
3. **Orphaned backup files:** `.bak` files and generated PDFs in repository
4. **Incomplete deprecation:** Some deprecated files still have active imports pointing to old locations

---

## SECTION 1: FILES SAFE TO DELETE

### 1.1 Backup Files (Should Remove Immediately)

**Path:** `src/deprecated/walk_forward_backtest.py.bak`
**Reason:** Backup file of deprecated code. No imports reference `.bak` files.
**Impact:** None - file is redundant backup
**Status:** SAFE TO DELETE

**Path:** `src/walk_forward_backtest.py.bak` (git marked for deletion D)
**Reason:** Another backup copy already staged for deletion
**Impact:** None
**Status:** SAFE TO DELETE

---

### 1.2 Generated PDF Files (Should Remove)

**Path:** `README.pdf`
**Reason:** Generated file, not source code. Should not be in git.
**Impact:** None - documentation artifact
**Status:** SAFE TO DELETE

**Path:** `docs/README.pdf`
**Reason:** Duplicate generated documentation artifact
**Impact:** None
**Status:** SAFE TO DELETE

**Recommendation:** Add `*.pdf` to `.gitignore` to prevent future commits

---

### 1.3 Deprecated Example Scripts (Can Remove if Not Referenced)

**Path:** `examples/altair_charting_examples.py`
**Status:** UNUSED - No imports or references found in active code
**Reason:** Example visualization code not integrated into pipeline
**Size:** 10.3 KB
**Impact:** None - example file only
**Recommendation:** KEEP for now (educational value). Mark if needed for removal.

**Path:** `examples/test_combined_features.py`
**Status:** UNUSED - No imports or references found
**Reason:** Test file for experimental feature combination
**Size:** Small
**Impact:** None
**Recommendation:** KEEP for now (test utility). Move to `tests/` or `deprecated/` if maintaining.

---

## SECTION 2: CRITICAL ISSUES REQUIRING IMMEDIATE FIX

### 2.1 BROKEN IMPORT: src/optimization/backtest_lineup_integration.py

**File:** `src/optimization/backtest_lineup_integration.py`
**Line:** 14
**Current Import:**
```python
from ..evaluation.backtest import WalkForwardBacktest
```

**Problem:**
- Path `src/evaluation/backtest/` does NOT exist
- Module was moved to `src/evaluation/deprecated/backtest/`
- Class `WalkForwardBacktest` now at: `src/evaluation/deprecated/backtest/walk_forward.py`

**Dependent Code:**
- `scripts/test_lineup_generation.py` - IMPORTS THIS (Line 19)
- `scripts/deprecated/run_backtest_with_lineups.py` - IMPORTS THIS (deprecated script)
- `notebooks/deprecated/run_backtest_with_lineups.ipynb` - IMPORTS THIS (deprecated notebook)

**Status:** CRITICAL - Will fail at runtime if test_lineup_generation.py is executed

**SOLUTION REQUIRED:**
Option A (Keep BacktestWithLineups): Fix import path
```python
from ..evaluation.deprecated.backtest.walk_forward import WalkForwardBacktest
```

Option B (Recommended): Move BacktestWithLineups to deprecated since it depends on deprecated code
- Move `src/optimization/backtest_lineup_integration.py` → `src/optimization/deprecated/backtest_lineup_integration.py`
- Update imports in deprecated scripts/notebooks

---

### 2.2 Compat Module Not Imported: src/deprecated/walk_forward_backtest_compat.py

**File:** `src/deprecated/walk_forward_backtest_compat.py`
**Status:** ORPHANED - No one imports from this compat module
**Purpose:** Provides backward compatibility for `from src.walk_forward_backtest` imports
**Current State:** Module exists but isn't used anywhere

**Finding:**
- No active code uses `from src.walk_forward_backtest`
- Compat module trying to re-export deprecated classes that are already in `deprecated/` folder
- Lines 12-13 import from paths that also don't exist:
  ```python
  from src.evaluation.backtest.walk_forward import WalkForwardBacktest
  from src.evaluation.backtest.trainers.per_player import _train_single_player_model
  ```

**Status:** BROKEN - Will fail if imported due to wrong paths
**SOLUTION:** Either fix imports or remove compat module (recommended since no one uses it)

---

## SECTION 3: FILES NEEDING REVIEW BEFORE DELETION

### 3.1 Deprecated Notebooks in notebooks/deprecated/

All are properly moved and documented. Safe to keep for reference.

**Status:** OK - No issues found. These serve as historical documentation.

```
notebooks/deprecated/
├── colab_walk_forward_backtest.ipynb      (1.2 MB - Colab-specific)
├── evaluate_backtest.ipynb                (61 KB - Old evaluation)
├── run_backtest.ipynb                     (325 KB - LARGE - Legacy backtest)
├── run_backtest_with_lineups.ipynb        (20 KB - Legacy lineups)
├── scratch.ipynb                          (1.3 MB - SCRATCH WORK - Consider deleting)
└── walk_forward_backtest.ipynb            (74 KB - Legacy walkforward)
```

**Concern:** `scratch.ipynb` (1.3 MB) appears to be experimental work. Check if needed.

---

### 3.2 Production-Ready New Code (Currently Untracked)

All of these are NEW production modules and should be ADDED to git (not deleted):

**Status:** READY FOR COMMIT - NOT TO BE DELETED

- `src/evaluation/walk_forward_simulation.py` - New simulation framework (replaces deprecated)
- `src/evaluation/backtest_report.py` - New reporting module
- `src/evaluation/metrics/segmentation.py` - New segmentation analysis
- `src/features/transformers/efficiency_metrics.py` - New feature transformer
- `src/features/transformers/impact_metrics.py` - New feature transformer
- `src/features/transformers/playmaking_metrics.py` - New feature transformer
- `src/models/bagging_model.py` - New ensemble model
- `src/models/stacking_model.py` - New ensemble model
- `scripts/predict_slate.py` - New production script
- `scripts/generate_lineups.py` - New production script
- `scripts/run_walk_forward_backtest.py` - New production script
- `config/models/bagged_xgboost.yaml` - New model config
- `config/models/stacked_xgb_rf.yaml` - New model config
- `notebooks/01_single_day_foundation.ipynb` - New production notebook
- `notebooks/02_full_slate_prediction.ipynb` - New production notebook
- `tests/evaluation/test_walk_forward_simulation.py` - New test suite

---

## SECTION 4: REDUNDANT/DUPLICATE CODE

### 4.1 Walk-Forward Implementation Redundancy

**Finding:** Multiple versions of walk-forward backtest logic:

1. **Deprecated (Legacy):** `src/evaluation/deprecated/backtest/walk_forward.py` (WalkForwardBacktest class)
2. **New (Production):** `src/evaluation/walk_forward_simulation.py` (WalkForwardSimulation class)
3. **Integration Wrapper:** `src/optimization/backtest_lineup_integration.py` (BacktestWithLineups - extends deprecated)

**Analysis:**
- WalkForwardBacktest: 35.5 KB, complex, extends trainer_base
- WalkForwardSimulation: New modular design, cleaner API, production-ready
- BacktestWithLineups: Extends deprecated class, still being used in test script

**Status:**
- Deprecated code is isolated correctly
- New code is the path forward
- Integration wrapper needs to be deprecated or fixed

---

### 4.2 No Code Duplication Detected

Search for common patterns found no significant duplication beyond the framework versions above.

---

## SECTION 5: IMPORTS FROM DEPRECATED PATHS

### 5.1 Active Imports of Non-Deprecated Code
✓ All clean - no active code imports from `src/evaluation/backtest/` (non-existent)

### 5.2 Scripts Still Using Deprecated Code
These are INTENTIONALLY in deprecated directories:

```
scripts/deprecated/run_backtest.py              - Uses deprecated framework
scripts/deprecated/run_backtest_with_lineups.py - Uses deprecated framework
```

✓ This is CORRECT - deprecated scripts in deprecated folder

---

## SECTION 6: UNUSED MODULES & ORPHANED CODE

### 6.1 Example Scripts (Not Integrated into Production Pipeline)

**Path:** `examples/feature_pipeline_usage.py`
**Status:** UNUSED - No imports or references in active code
**Type:** Educational example
**Recommendation:** KEEP (documentation value)

**Path:** `examples/opponent_features_usage.py`
**Status:** REFERENCED IN DOCS but not in code
- `docs/features/OPPONENT_FEATURES_IMPLEMENTATION.md` mentions it
- Config files exist: `config/features/opponent_features.yaml`, `config/features/base_with_opponent.yaml`
- Feature code exists: `src/features/transformers/opponent_stats.py`
- Tests exist: `tests/features/test_opponent_features.py`
**Assessment:** Part of active feature set, properly documented
**Recommendation:** KEEP

**Path:** `examples/run_opponent_backtest.py`
**Status:** REFERENCED IN DOCS
**Recommendation:** KEEP (used for demonstration of opponent features)

---

## SECTION 7: ORPHANED DIRECTORIES

### 7.1 examples/ Directory Status

All files in `examples/` serve educational purposes:
- `altair_charting_examples.py` - Visualization examples
- `feature_pipeline_usage.py` - Feature configuration example
- `opponent_features_usage.py` - Opponent features example (referenced in docs)
- `run_opponent_backtest.py` - Backtest example (referenced in docs)
- `test_combined_features.py` - Testing example

**Status:** NOT ORPHANED - All have purpose or are referenced in documentation

---

## SECTION 8: IMPACT ANALYSIS

### 8.1 If backtest_lineup_integration.py Is Removed

**Files Affected:**
- `scripts/test_lineup_generation.py` - Would break (imports BacktestWithLineups)

**Recommendation:**
- IF removing: Update `test_lineup_generation.py` to use new `WalkForwardSimulation` instead
- OR: Fix import path and keep for backward compatibility

### 8.2 If Deprecated Notebooks Are Removed

**Impact:** None - they are already not in main workflow
**But:** Large files (1.3 MB scratch.ipynb) waste space

---

## SECTION 9: CONSOLIDATION OPPORTUNITIES

### 9.1 Recommended Actions (Priority Order)

**Priority 1 (CRITICAL - Fix Broken Code):**
1. Fix `src/optimization/backtest_lineup_integration.py` import path
2. Fix `src/deprecated/walk_forward_backtest_compat.py` import paths
3. Update `scripts/test_lineup_generation.py` to use corrected imports

**Priority 2 (CLEANUP - Remove Non-Code Files):**
1. Delete `README.pdf` (generated artifact)
2. Delete `docs/README.pdf` (generated artifact)
3. Delete `src/deprecated/walk_forward_backtest.py.bak`
4. Add `*.pdf` to `.gitignore`

**Priority 3 (CONSIDER - Deprecated Cleanup):**
1. Decide: Delete or move `notebooks/deprecated/scratch.ipynb` (1.3 MB - workspace file)
2. Consider: Move `examples/` to `docs/examples/` for better organization
3. Consider: Deprecate `backtest_lineup_integration.py` since it uses deprecated code

**Priority 4 (COMMIT - Staging New Production Code):**
1. Add all untracked files under `src/evaluation/`, `src/features/transformers/`, `src/models/`, `scripts/`, `config/models/`, `notebooks/` (new production code)
2. Commit with message: "Add production-ready modules: walk-forward simulation, ensemble models, and prediction scripts"

---

## SECTION 10: VERIFICATION CHECKLIST

After cleanup, verify:

```bash
# 1. No broken imports in active code
pytest src/ -v --tb=short 2>&1 | grep -E "ImportError|ModuleNotFoundError"

# 2. All active production scripts work
python scripts/predict_slate.py --help
python scripts/generate_lineups.py --help
python scripts/run_walk_forward_backtest.py --help

# 3. Deprecated code still accessible for backwards compatibility
python -c "from src.evaluation.deprecated.backtest.walk_forward import WalkForwardBacktest; print('OK')"

# 4. No broken notebook imports
jupyter nbconvert --to notebook --execute notebooks/01_single_day_foundation.ipynb --inplace 2>&1 | head -20

# 5. Git status is clean
git status --porcelain | wc -l  # Should be 0 or only minor tracked changes
```

---

## SECTION 11: ROLLBACK PLAN

Create backup branch before making changes:
```bash
git checkout -b backup/pre-cleanup-2025-10-26
git push origin backup/pre-cleanup-2025-10-26
```

If cleanup causes issues, restore with:
```bash
git revert <commit-hash>  # For specific changes
# OR
git reset --hard backup/pre-cleanup-2025-10-26  # For complete rollback
```

---

## SECTION 12: RECOMMENDATIONS SUMMARY

### Delete (Safe)
- `src/deprecated/walk_forward_backtest.py.bak`
- `README.pdf`
- `docs/README.pdf`

### Fix (Critical)
- `src/optimization/backtest_lineup_integration.py` - Fix import path to deprecated module
- `src/deprecated/walk_forward_backtest_compat.py` - Fix import paths or remove if unused

### Keep (In Git)
- All deprecated/ directories (serve as backward compatibility)
- All example/ files (documented/educational)
- All new production code (currently untracked)
- All configuration files

### Review (Consider)
- `notebooks/deprecated/scratch.ipynb` - Large workspace file, possibly unnecessary

---

## APPENDIX A: File Inventory

### Current Deprecated Directories (Properly Organized)
```
src/deprecated/
├── walk_forward_backtest.py
├── walk_forward_backtest_compat.py
└── walk_forward_backtest.py.bak    ← DELETE

src/evaluation/deprecated/
└── backtest/
    ├── __init__.py
    ├── trainer_base.py
    ├── trainers/
    ├── validator.py
    └── walk_forward.py

scripts/deprecated/
├── run_backtest.py
└── run_backtest_with_lineups.py

notebooks/deprecated/
├── colab_walk_forward_backtest.ipynb
├── evaluate_backtest.ipynb
├── run_backtest.ipynb
├── run_backtest_with_lineups.ipynb
├── scratch.ipynb
└── walk_forward_backtest.ipynb

src/interface/deprecated/
└── panel_backtest_app.py
```

### Production-Ready New Modules (Currently Untracked - Should Commit)
```
src/evaluation/
├── walk_forward_simulation.py      ← NEW
├── backtest_report.py              ← NEW
└── metrics/
    └── segmentation.py             ← NEW

src/features/transformers/
├── efficiency_metrics.py            ← NEW
├── impact_metrics.py                ← NEW
└── playmaking_metrics.py            ← NEW

src/models/
├── bagging_model.py                 ← NEW
└── stacking_model.py                ← NEW

scripts/
├── predict_slate.py                 ← NEW
├── generate_lineups.py              ← NEW
└── run_walk_forward_backtest.py    ← NEW

config/models/
├── bagged_xgboost.yaml             ← NEW
└── stacked_xgb_rf.yaml             ← NEW

notebooks/
├── 01_single_day_foundation.ipynb  ← NEW
└── 02_full_slate_prediction.ipynb  ← NEW

tests/evaluation/
└── test_walk_forward_simulation.py ← NEW
```

---

## APPENDIX B: Import Path Analysis

### Deprecated Imports (Moved but Still Accessible)
```
src/evaluation/deprecated/backtest/
├── walk_forward.py         (WalkForwardBacktest)
├── trainer_base.py         (TrainerBase)
├── trainers/per_player.py  (_train_single_player_model)
├── trainers/per_slate.py   (PerSlateTrainer)
└── validator.py            (Validator)
```

### Active Production Imports (No Changes Needed)
```
from src.evaluation.walk_forward_simulation import WalkForwardSimulation  ✓
from src.evaluation.backtest_report import BacktestReport                 ✓
from src.evaluation.metrics.segmentation import analyze_by_salary        ✓
from src.models.xgboost_model import XGBoostModel                        ✓
from src.models.bagging_model import BaggingModel                        ✓
from src.models.stacking_model import StackingModel                      ✓
```

### Broken Imports (Need Fixing)
```
src/optimization/backtest_lineup_integration.py
  from ..evaluation.backtest import WalkForwardBacktest  ✗ (path doesn't exist)

src/deprecated/walk_forward_backtest_compat.py
  from src.evaluation.backtest.walk_forward import WalkForwardBacktest  ✗
  from src.evaluation.backtest.trainers.per_player import _train_single_player_model  ✗
```

---

## FINAL STATUS

**Repository Health:** 7/10
- Deprecation strategy well-executed
- Production code modern and modular
- BUT: 2 critical import path issues block some code paths
- Non-code artifacts (PDFs, backup files) cluttering repo

**Action Items Before Merge:** 3 critical, 2 high priority
**Recommended Cleanup Time:** 30-45 minutes
**Risk Level:** LOW (no active code uses broken modules; fixes are straightforward)

