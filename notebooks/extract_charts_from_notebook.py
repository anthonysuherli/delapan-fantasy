import json
import base64
from pathlib import Path

notebook_path = Path(__file__).parent / 'backtest_1d_by_player.ipynb'
output_dir = Path(__file__).parent / 'report_charts'
output_dir.mkdir(exist_ok=True)

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

chart_count = 0
chart_cells = [13, 26, 31, 33, 36]

for cell_idx in chart_cells:
    cell = notebook['cells'][cell_idx]

    if 'outputs' in cell and len(cell['outputs']) > 0:
        for output in cell['outputs']:
            if 'data' in output and 'image/png' in output['data']:
                chart_count += 1
                img_data = output['data']['image/png']

                img_bytes = base64.b64decode(img_data)

                output_file = output_dir / f'chart_{chart_count:02d}_cell_{cell_idx}.png'
                with open(output_file, 'wb') as img_file:
                    img_file.write(img_bytes)

                print(f"Extracted chart {chart_count} from cell {cell_idx}: {output_file.name}")

print(f"\nTotal charts extracted: {chart_count}")
print(f"Charts saved to: {output_dir}")
