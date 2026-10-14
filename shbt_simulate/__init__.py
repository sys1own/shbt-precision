import importlib.util
import os

# Resolve the path to the original script file
_script_path = os.path.join(os.path.dirname(__file__), '..', 'shbt_simulate.py')
_spec = importlib.util.spec_from_file_location('shbt_simulate', _script_path)
_module = importlib.util.module_from_spec(_spec)
if _spec and _spec.loader:
    _spec.loader.exec_module(module)  # type: ignore[assignment]
# Re‑export public attributes from the script
for _name in dir(module):
    if not _name.startswith('_'):
        globals()[_name] = getattr(module, _name)
