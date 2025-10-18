#!/bin/bash
# Git Worktree Creation Script for Bash
# Creates a new worktree for feature branch development

set -e

# Default values
BASE_BRANCH="main"
WORKTREE_DIR_PREFIX="../delapan-fantasy"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <branch-name> [base-branch] [worktree-dir]"
    echo ""
    echo "Examples:"
    echo "  $0 feat/opponent-features"
    echo "  $0 feat/opponent-features main ../delapan-fantasy-opponent"
    exit 1
fi

BRANCH_NAME=$1
BASE_BRANCH=${2:-$BASE_BRANCH}
WORKTREE_DIR="${3:-${WORKTREE_DIR_PREFIX}-${BRANCH_NAME#*/}}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}Creating worktree for branch: $BRANCH_NAME${NC}"
echo -e "${CYAN}Base branch: $BASE_BRANCH${NC}"
echo -e "${CYAN}Worktree path: $WORKTREE_DIR${NC}"
echo ""

# Verify base branch exists
if ! git branch -a | grep -q "$BASE_BRANCH"; then
    echo -e "${RED}Error: Base branch '$BASE_BRANCH' not found${NC}"
    exit 1
fi

# Create new branch from base
echo -e "${YELLOW}Creating branch from $BASE_BRANCH...${NC}"
git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" "origin/$BASE_BRANCH"

echo -e "${GREEN}✓ Worktree created successfully${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "1. cd $WORKTREE_DIR"
echo "2. pip install -r requirements.txt"
echo "3. Start development"
echo ""
echo -e "${CYAN}To clean up when done:${NC}"
echo "git worktree remove $WORKTREE_DIR"
