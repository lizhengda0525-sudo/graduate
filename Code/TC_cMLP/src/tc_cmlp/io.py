import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tc_cmlp.models.training import FitResult


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return super().default(value)


def prepare_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)


def save_fit_result(output_dir: Path, method: str, result: FitResult) -> None:
    method_dir = prepare_output_dir(output_dir / method)
    np.save(method_dir / "causal_matrices.npy", result.causal_matrices)
    pd.DataFrame(result.history).to_csv(method_dir / "training_history.csv", index=False)
    state_dicts = [model.state_dict() for model in result.models]
    torch.save(state_dicts, method_dir / "models.pt")
