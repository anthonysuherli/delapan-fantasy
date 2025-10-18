"""
Script to apply dark theme to all Plotly charts.
Updates plotly_visualizations.py to use consistent dark styling.
"""

import re
from pathlib import Path


def add_dark_theme_helper():
    """Add helper method to apply dark theme to figures."""
    return '''
    def _apply_dark_theme(self, fig, height: int = 600):
        """Apply dark theme to a Plotly figure."""
        fig.update_layout(
            **self.dark_template['layout'],
            height=height,
            hovermode='closest'
        )

        # Update all axes with dark theme
        fig.update_xaxes(
            gridcolor=self.dark_template['layout']['xaxis']['gridcolor'],
            linecolor=self.dark_template['layout']['xaxis']['linecolor'],
            tickfont=dict(color=self.dark_template['layout']['xaxis']['tickfont']['color'])
        )
        fig.update_yaxes(
            gridcolor=self.dark_template['layout']['yaxis']['gridcolor'],
            linecolor=self.dark_template['layout']['yaxis']['linecolor'],
            tickfont=dict(color=self.dark_template['layout']['yaxis']['tickfont']['color'])
        )

        return fig
'''


def update_chart_methods(file_path: Path):
    """Update all chart generation methods to use dark theme."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add the dark theme helper method after __init__
    init_end = content.find('def generate_all_charts')
    if init_end != -1:
        helper_method = add_dark_theme_helper()
        content = content[:init_end] + helper_method + '\n' + content[init_end:]

    # Replace all fig.update_layout calls to use dark theme
    # Pattern: find fig.update_layout with various parameters
    def replace_update_layout(match):
        """Replace update_layout with dark theme version."""
        params = match.group(1)

        # Extract height if present
        height_match = re.search(r'height\s*=\s*(\d+)', params)
        height = height_match.group(1) if height_match else '600'

        # Extract title if present
        title_match = re.search(r'title(?:_text)?\s*=\s*["\']([^"\']+)["\']', params)
        title = title_match.group(1) if title_match else ''

        # Build replacement
        replacement = f'self._apply_dark_theme(fig, height={height})\n        '
        if title:
            replacement += f'fig.update_layout(title_text="{title}")\n        '

        return replacement

    # This is a simplified version - the actual implementation would need more sophisticated parsing
    # For now, let's just append the _apply_dark_theme call before write_html

    # Pattern: find all write_html calls and add dark theme application before them
    pattern = r'(\s+)(output_path = self\.charts_dir / [^\n]+)\n(\s+)(fig\.write_html\(output_path\))'

    def add_theme_before_write(match):
        indent = match.group(1)
        output_line = match.group(2)
        write_indent = match.group(3)
        write_line = match.group(4)

        return f'{indent}{output_line}\n{indent}self._apply_dark_theme(fig)\n{write_indent}{write_line}'

    content = re.sub(pattern, add_theme_before_write, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Updated {file_path}")


if __name__ == '__main__':
    file_path = Path(__file__).parent.parent / 'src' / 'evaluation' / 'plotly_visualizations.py'
    update_chart_methods(file_path)
    print("Dark theme applied to all charts!")