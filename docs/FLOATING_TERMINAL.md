# Floating Terminal Panel - Implementation Guide

## Overview

Added an interactive floating terminal panel to the Panel backtest interface that appears at the bottom half of the screen. The terminal automatically activates when the backtest starts running and mirrors all execution logs in real-time.

## Features

- **Fixed Position**: Anchored to the bottom 50% of the viewport, leaving top 50% for configuration and results
- **Auto-activation**: Terminal automatically shows when logs start flowing during backtest execution
- **Real-time Updates**: Mirrors all logs from main display with color-coded severity levels
- **Color Coding**:
  - Blue (`#58a6ff`): INFO messages and timestamps
  - Red (`#f85149`): ERROR messages
  - Orange (`#d29922`): WARNING messages
  - Gray (`#8b949e`): Timestamps
- **Terminal Controls**:
  - **Clear**: Wipe all logs from floating terminal
  - **Minimize**: Toggle terminal visibility (hide/show)
- **Auto-scroll**: Terminal automatically scrolls to show latest logs
- **Extended History**: Maintains up to 200 log entries (vs 100 in main display)

## Visual Design

- Dark terminal theme matching GitHub's color scheme
- Monospace font (Hack/Consolas/Monaco) for authentic terminal appearance
- Smooth rounded borders and subtle shadow for modern aesthetics
- Clear visual hierarchy with header bar containing title and controls
- Responsive to log volume with smooth overflow handling

## Technical Implementation

### CSS Styling (lines 102-201)

**Key Classes:**
- `#floating-terminal-container`: Fixed position panel (50vh height, z-index 1000)
- `#floating-terminal-header`: Header with title and control buttons
- `#floating-terminal-controls`: Button container with flex layout
- `.terminal-btn`: Individual button styling with hover effects
- `#floating-terminal-content`: Scrollable log container
- `#floating-terminal-log`: Log text area with monospace font

### JavaScript Functions (lines 554-589)

**Dynamic Functions:**
- `showTerminal()`: Display floating terminal and add body class
- `hideTerminal()`: Hide floating terminal and remove body class
- `toggleTerminal()`: Toggle between show/hide states
- `clearTerminal()`: Clear all log entries from display

### Update Loop (lines 525-590)

The `update_display()` function now:
1. Builds floating terminal HTML with last 200 logs
2. Applies color coding based on log severity
3. Injects auto-scroll JavaScript
4. Auto-shows terminal when logs arrive
5. Maintains body state class for CSS viewport adjustment

## Layout Structure

```
┌─────────────────────────────────────────┐
│         Configuration & Results          │  ← Top 50vh
│                                          │
├──────────────────────────────────────────┤
│  ⚡ Live Terminal  [Clear] [Minimize]   │  ← Header
├──────────────────────────────────────────┤
│ [HH:MM:SS] INFO    Starting backtest...  │
│ [HH:MM:SS] INFO    Loading data...       │  ← Log entries
│ [HH:MM:SS] ERROR   Failed to load...     │     (scrollable, 50vh)
└──────────────────────────────────────────┘
```

## Usage

### Automatic Behavior

1. Start a backtest via "Run Backtest" button
2. Floating terminal automatically appears at screen bottom
3. Real-time logs stream from backtest execution
4. Terminal auto-scrolls to latest entries

### Manual Controls

- Click **Minimize**: Hide terminal and restore full config view
- Click **Clear**: Wipe terminal logs (doesn't clear main logs)
- Scroll within terminal: Browse log history
- Re-run backtest: Terminal will auto-show again

## Integration Points

### Modified Functions

1. **`update_display()` (lines 485-590)**
   - Builds floating terminal HTML from runner logs
   - Applies color classes to log entries
   - Updates floating terminal on every poll cycle
   - Auto-scrolls and auto-shows on log arrival

2. **`on_run_backtest()` (lines 421-482)**
   - Unchanged; works with new terminal via callback system

### New Global

- `floating_terminal_html`: Panel HTML pane that holds terminal markup

## Browser Compatibility

- Chrome/Edge/Firefox: Full support
- Safari: Full support
- Mobile browsers: Terminal appears but may be difficult to interact with

## Performance Considerations

- Floating terminal rebuilds on every update (1s interval)
- Maintains last 200 logs in memory (minimal overhead)
- CSS-based positioning and animation (no layout thrashing)
- HTML injection is safe (logs are HTML-escaped)

## Future Enhancements

1. Add search/filter functionality in terminal
2. Export logs to file
3. Resizable terminal height (drag from top edge)
4. Log level filtering (show only errors, etc)
5. Syntax highlighting for specific log patterns
6. Terminal history persistence across sessions
