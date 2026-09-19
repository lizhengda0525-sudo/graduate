import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3] / "Neural-GC-original"
SOURCE_FILE = REPOSITORY_ROOT / "models" / "cmlp.py"

if not SOURCE_FILE.is_file():
    raise FileNotFoundError(f"Original Neural-GC cMLP source was not found: {SOURCE_FILE}")

sys.path.insert(0, str(REPOSITORY_ROOT))
module = importlib.import_module("models.cmlp")
if not Path(module.__file__).resolve().is_relative_to(REPOSITORY_ROOT.resolve()):
    raise ImportError(f"Unexpected cMLP module location: {module.__file__}")

cMLP = module.cMLP
prox_update = module.prox_update
regularize = module.regularize
ridge_regularize = module.ridge_regularize
train_model_ista = module.train_model_ista
