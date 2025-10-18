# Git Worktree Quick Start Guide

Isolated development branches without switching contexts in your main repository.

## What is a Worktree?

A worktree creates a separate working directory linked to your git repository, allowing parallel development on multiple branches simultaneously without `git checkout` context switching.

**Benefits:**
- Maintain multiple feature branches without stashing/switching
- Run tests on different branches in parallel
- Backup your current work before major refactors
- Easier code review workflow

## Quick Commands

### Create Worktree

**PowerShell (Windows):**
```powershell
cd delapan-fantasy  # Main repo directory
.\scripts\create_worktree.ps1 -BranchName "feat/opponent-features"
```

**Bash (macOS/Linux/WSL):**
```bash
cd delapan-fantasy  # Main repo directory
bash scripts/create_worktree.sh feat/opponent-features
```

**Manual (any OS):**
```bash
# Create new worktree on new branch from main
git worktree add -b feat/opponent-features ../delapan-fantasy-opponent-features origin/main

# Or create worktree on existing branch
git worktree add ../delapan-fantasy-opponent-features origin/feat/opponent-features
```

### Work in Worktree

```bash
# Navigate to worktree
cd ../delapan-fantasy-opponent-features

# Install dependencies
pip install -r requirements.txt

# Create/edit files
mkdir -p src/features/transformers
touch src/features/transformers/opponent_features.py

# Commit changes
git add src/features/transformers/opponent_features.py
git commit -m "Add opponent features transformer"

# Push to remote
git push origin feat/opponent-features
```

### Clean Up Worktree

```bash
# When done, remove worktree
git worktree remove ../delapan-fantasy-opponent-features

# Or prune stale worktrees
git worktree prune
```

### List Active Worktrees

```bash
# Show all worktrees
git worktree list

# Example output:
# /path/to/delapan-fantasy                        679f8c4 [main]
# /path/to/delapan-fantasy-opponent-features      a1b2c3d [feat/opponent-features]
```

## Multi-Worktree Workflow

### Scenario 1: Feature + Testing

```bash
# Terminal 1: Main repo (run tests)
cd delapan-fantasy
pytest tests/

# Terminal 2: Feature branch (develop)
cd delapan-fantasy-opponent-features
# ... edit code ...
git add .
git commit -m "WIP: opponent features"

# Terminal 3: GPU branch (experimental)
cd delapan-fantasy-gpu-optimization
# ... test GPU code ...
```

All three work independently without affecting each other.

### Scenario 2: Code Review & Bugfix

```bash
# Terminal 1: Main branch (production)
cd delapan-fantasy
git checkout main
python scripts/run_backtest.py --test-start 20250205 --test-end 20250206

# Terminal 2: Feature branch (under review)
cd delapan-fantasy-feature-branch
# Review code, make changes

# Terminal 3: Hotfix branch (critical bug)
git worktree add -b hotfix/critical-bug ../delapan-fantasy-hotfix origin/main
cd ../delapan-fantasy-hotfix
# ... fix critical bug ...
git push origin hotfix/critical-bug
```

Create PR for hotfix while feature branch is still in review.

## Advanced Usage

### Rebase Worktree on Latest Main

```bash
cd ../delapan-fantasy-opponent-features
git fetch origin
git rebase origin/main
# Fix conflicts if needed
git push origin feat/opponent-features --force-with-lease
```

### Share Worktree Between PCs

Worktrees are filesystem-specific. To work on same branch from different PC:

```bash
# PC A: Push branch
git push origin feat/opponent-features

# PC B: Create new worktree
git fetch origin
git worktree add -b feat/opponent-features ../delapan-fantasy-opponent-features origin/feat/opponent-features
```

### Create Worktree on Existing Branch

```bash
# If branch already exists locally
git worktree add -b feat/opponent-features ../delapan-fantasy-opponent-features feat/opponent-features

# If branch only exists on remote
git worktree add --track -b feat/opponent-features ../delapan-fantasy-opponent-features origin/feat/opponent-features
```

## Troubleshooting

### Error: "already checked out"

```
fatal: 'feat/opponent-features' is already checked out at '/other/path'
```

**Solution:** Worktree is already active elsewhere.
```bash
# List worktrees
git worktree list

# Remove stale worktree if stuck
git worktree remove --force /other/path
```

### Error: "no such file"

```
fatal: '../delapan-fantasy-opponent-features' does not contain a valid git repository
```

**Solution:** Create directory or use existing directory.
```bash
# Create directory first
mkdir ../delapan-fantasy-opponent-features
git worktree add -b feat/opponent-features ../delapan-fantasy-opponent-features origin/main
```

### Worktree locked/won't delete

```bash
# Force unlock and remove
git worktree remove --force ../delapan-fantasy-opponent-features

# If still stuck, remove admin lock file
rm -rf ../delapan-fantasy-opponent-features/.git/index.lock
```

### Syncing Data Between Worktrees

Worktrees share git history but have separate working directories.

```bash
# Worktree 1: Collect data
cd ../delapan-fantasy-opponent-features
python scripts/collect_games.py --start-date 20250101 --end-date 20250110

# Main repo can access same data
cd ../delapan-fantasy
ls data/inputs/box_scores/  # Can see files from worktree

# But source code changes are isolated
cat src/features/transformers/opponent_features.py  # Not available in main
```

## Best Practices

1. **Always push before removing worktree** to avoid losing work
2. **Keep worktree directories close** (../worktree-name, not deep nested)
3. **Use descriptive branch names** (feat/opponent-features, fix/cache-bug)
4. **Rebase frequently** to stay in sync with main
5. **Clean up old worktrees** periodically (git worktree prune)
6. **Don't commit to main from worktree** - it's isolated and confusing

## References

- [Git Worktree Documentation](https://git-scm.com/docs/git-worktree)
- [GitHub Worktree Guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)
