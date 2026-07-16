"""Compatibility shim for the predict notebook module."""
import json
from pathlib import Path

notebook_dir = Path(__file__).resolve().parent
module_candidates = [
    notebook_dir / 'predict.ipynb',
    notebook_dir / 'predict - Copy.ipynb',
    *sorted(notebook_dir.glob('predict*.ipynb')),
]
module_path = next((path for path in module_candidates if path.exists()), None)
if module_path is None:
    raise ImportError(f'Notebook module not found in {notebook_dir}')


def _load_notebook_code(path: Path) -> str:
    """Extract executable Python code from a Jupyter notebook."""
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', [])
        if isinstance(source, list):
            code_cells.append(''.join(source))
        elif isinstance(source, str):
            code_cells.append(source)

    if not code_cells:
        raise ImportError(f'No code cells found in notebook: {path}')

    return '\n\n'.join(code_cells)


namespace = {
    '__name__': __name__,
    '__file__': str(module_path),
    '__package__': None,
}
exec(_load_notebook_code(module_path), namespace)

globals().update(namespace)

__all__ = [
    'app',
    'model',
    'PatientData',
    'PredictionResponse',
    'health',
    'metrics',
    'root',
    'predict',
    'predict_batch',
]
