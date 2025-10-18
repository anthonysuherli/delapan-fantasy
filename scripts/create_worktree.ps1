# Git Worktree Creation Script for PowerShell
# Creates a new worktree for feature branch development

param(
    [Parameter(Mandatory=$true)]
    [string]$BranchName,

    [Parameter(Mandatory=$false)]
    [string]$BaseBranch = "main",

    [Parameter(Mandatory=$false)]
    [string]$WorktreeDir = "../delapan-fantasy-$BranchName"
)

function Create-Worktree {
    param(
        [string]$Branch,
        [string]$Base,
        [string]$Dir
    )

    Write-Host "Creating worktree for branch: $Branch" -ForegroundColor Cyan
    Write-Host "Base branch: $Base" -ForegroundColor Cyan
    Write-Host "Worktree path: $Dir" -ForegroundColor Cyan
    Write-Host ""

    # Verify base branch exists
    $branchExists = git branch -a | Select-String "$Base"
    if (-not $branchExists) {
        Write-Host "Error: Base branch '$Base' not found" -ForegroundColor Red
        exit 1
    }

    # Create new branch from base
    Write-Host "Creating branch from $Base..." -ForegroundColor Yellow
    git worktree add -b "$Branch" "$Dir" "origin/$Base"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Worktree created successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. cd $Dir"
        Write-Host "2. pip install -r requirements.txt"
        Write-Host "3. Start development"
        Write-Host ""
        Write-Host "To clean up when done:" -ForegroundColor Cyan
        Write-Host "git worktree remove $Dir"
    }
    else {
        Write-Host "Error creating worktree" -ForegroundColor Red
        exit 1
    }
}

Create-Worktree -Branch $BranchName -Base $BaseBranch -Dir $WorktreeDir
