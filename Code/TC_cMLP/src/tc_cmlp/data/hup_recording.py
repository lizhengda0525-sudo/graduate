from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HUPRecording:
    windows: NDArray[np.float32]
    channel_names: list[str]
    soz_mask: NDArray[np.bool_]
    window_start_seconds: NDArray[np.float64]
    ictal_mask: NDArray[np.bool_]
    sampling_rate: float
