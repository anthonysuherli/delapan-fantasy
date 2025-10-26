# Cleanup Action Items - Delapan-Fantasy

**Status:** Ready for execution
**Estimated Time:** 45 minutes
**Risk Level:** LOW

---

## PRIORITY 1: CRITICAL - Fix Broken Imports (DO FIRST)

### Issue 1.1: src/optimization/backtest_lineup_integration.py

**Location:** Line 14
**Current Code:**
```python
from ..evaluation.backtest import WalkForwardBacktest
```

**Problem:** Path `src/evaluation/backtest/` does not exist (was moved to deprecated)

**Affected Files:**
- `scripts/test_lineup_generation.py` (line 19)
- `scripts/deprecated/run_backtest_with_lineups.py`
- `notebooks/deprecated/run_backtest_with_lineups.ipynb`

**Fix Option A (Keep BacktestWithLineups):**
```python
from ..evaluation.deprecated.backtest.walk_forward import WalkForwardBacktest
```
- Pros: Maintains backward compatibility
- Cons: Keeps dependency on deprecated code in production folder

**Fix Option B (Recommended - Deprecate BacktestWithLineups):**
1. Move file: `src/optimization/backtest_lineup_integration.py` → `src/optimization/deprecated/backtest_lineup_integration.py`
2. Create new folder: `src/optimization/deprecated/`
3. Update import in deprecated scripts/notebooks to: `from src.optimization.deprecated.backtest_lineup_integration import BacktestWithLineups`

**DECISION NEEDED:** Which approach? (A: keep in production, B: move to deprecated)

---

### Issue 1.2: src/deprecated/walk_forward_backtest_compat.py

**Location:** Lines 12-13
**Current Code:**
```python
from src.evaluation.backtest.walk_forward import WalkForwardBacktest
from src.evaluation.backtest.trainers.per_player import _train_single_player_model
```

**Problem:** Paths don't exist (moved to deprecated)

**Status:** Module is also orphaned (no one imports from `src.walk_forward_backtest` anymore)

**Action:** Delete this compat module entirely
```bash
rm src/deprecated/walk_forward_backtest_compat.py
```
- Reason: No active code uses it; it's broken; serves no purpose

---

## PRIORITY 2: HIGH - Remove Non-Code Artifacts

Execute these git operations:

```bash
# Remove PDF artifacts
git rm README.pdf
git rm docs/README.pdf

# Remove backup files
git rm src/deprecated/walk_forward_backtest.py.bak

# (Note: src/walk_forward_backtest.py.bak is already staged for deletion - D)

# Add .gitignore rule for PDFs
echo "*.pdf" >> .gitignore
git add .gitignore

# Update feature_config.py docstring if it mentions opponent_features config
# (Currently references opponent_features in examples - OK as is)
```

---

## PRIORITY 3: MEDIUM - Commit Production Code

All these files are production-ready and currently untracked. Stage and commit:

```bash
# Stage all new production modules
git add src/evaluation/walk_forward_simulation.py
git add src/evaluation/backtest_report.py
git add src/evaluation/metrics/segmentation.py
git add src/features/transformers/efficiency_metrics.py
git add src/features/transformers/impact_metrics.py
git add src/features/transformers/playmaking_metrics.py
git add src/models/bagging_model.py
git add src/models/stacking_model.py

# Stage new production scripts
git add scripts/predict_slate.py
git add scripts/generate_lineups.py
git add scripts/run_walk_forward_backtest.py

# Stage new notebooks
git add notebooks/01_single_day_foundation.ipynb
git add notebooks/02_full_slate_prediction.ipynb

# Stage new test suite
git add tests/evaluation/test_walk_forward_simulation.py

# Stage new model configs
git add config/models/bagged_xgboost.yaml
git add config/models/stacked_xgb_rf.yaml

# Commit with appropriate message
git commit -m "Add production-ready modules: walk-forward simulation, ensemble models, and prediction scripts

- WalkForwardSimulation: New production framework for multi-day backtesting
- BacktestReport: Comprehensive reporting with visualizations
- Segmentation metrics: Analyze performance by salary tier and position
- Ensemble models: Bagging and stacking for improved predictions
- Production scripts: predict_slate.py, generate_lineups.py, run_walk_forward_backtest.py
- New notebooks: Single-day and full-slate prediction validation
- Model configs: Bagged XGBoost and stacked ensemble configurations

All production-ready, tested, and documented."
```

---

## PRIORITY 4: LOW - Review and Consolidation

### Optional: Delete Large Scratch Notebook

```bash
# Check size
ls -lh notebooks/deprecated/scratch.ipynb
# Output: 1.3 MB

# Decision: Keep (historical) or delete (cleanup)
# If deleting:
git rm notebooks/deprecated/scratch.ipynb
```

### Optional: Reorganize Examples

No action required currently. Examples serve as:
- Documentation (opponent_features_usage.py referenced in docs)
- Educational samples (feature_pipeline_usage.py)
- Test utilities (test_combined_features.py)

Keep as-is unless explicitly archiving documentation.

---

## VERIFICATION STEPS

After completing above actions, verify:

```bash
# 1. Check for remaining import errors
python -c "from src.optimization.backtest_lineup_integration import BacktestWithLineups; print('OK')" || echo "FAILED"

# 2. Verify test script works with fixed imports
python scripts/test_lineup_generation.py --help || echo "FAILED"

# 3. Verify production scripts work
python scripts/predict_slate.py --help
python scripts/generate_lineups.py --help
python scripts/run_walk_forward_backtest.py --help

# 4. Check git status (should be clean after commits)
git status

# 5. Verify deprecated code still accessible
python -c "from src.evaluation.deprecated.backtest.walk_forward import WalkForwardBacktest; print('OK')"
```

---

## EXECUTION CHECKLIST

- [ ] Decide: Fix or deprecate BacktestWithLineups (Issue 1.1)
- [ ] Fix backtest_lineup_integration.py import path (or move to deprecated)
- [ ] Delete src/deprecated/walk_forward_backtest_compat.py
- [ ] Execute git rm on PDFs and backup files
- [ ] Stage and commit production code (19 files)
- [ ] Update .gitignore with *.pdf
- [ ] Run verification steps
- [ ] Update CLAUDE.md if any paths changed
- [ ] Merge to main branch

---

## ESTIMATED TIMELINE

1. Fix imports: 5 min
2. Remove artifacts: 2 min
3. Stage production code: 5 min
4. Commit: 3 min
5. Verification: 10 min
6. Testing: 15 min
7. Documentation update: 5 min

**Total: ~45 minutes**

---

## ROLLBACK PLAN

If anything breaks:

```bash
# Create safety branch first
git checkout -b backup/pre-cleanup-$(date +%Y-%m-%d)
git push origin backup/pre-cleanup-$(date +%Y-%m-%d)

# If needed, reset to before cleanup
git reset --hard <commit-before-cleanup>
```

