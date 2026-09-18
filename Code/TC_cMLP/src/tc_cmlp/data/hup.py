from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tc_cmlp.data.windows import sliding_window_bounds


@dataclass(frozen=True)
class HUPRecording:
    windows: NDArray[np.float32]
    channel_names: list[str]
    soz_mask: NDArray[np.bool_]
    window_start_seconds: NDArray[np.float64]
    ictal_mask: NDArray[np.bool_]
    sampling_rate: float
    seizure_duration: float


def _recording_paths(dataset_root: Path, subject: str, run: int) -> tuple[Path, Path, Path]:
    subject_name = subject if subject.startswith("sub-") else f"sub-{subject}"
    ieeg_dir = dataset_root / subject_name / "ses-presurgery" / "ieeg"
    pattern = f"{subject_name}_ses-presurgery_task-ictal_*_run-{run:02d}_ieeg.edf"
    matches = list(ieeg_dir.glob(pattern))
    if len(matches) != 1:
        message = f"Expected one EDF for {subject_name} run {run}, found {len(matches)}"
        raise FileNotFoundError(message)
    edf_path = matches[0]
    stem = edf_path.name.removesuffix("_ieeg.edf")
    return (
        edf_path,
        edf_path.with_name(f"{stem}_channels.tsv"),
        edf_path.with_name(f"{stem}_events.tsv"),
    )


def _read_metadata(
    channels_path: Path,
    events_path: Path,
) -> tuple[list[str], set[str], float, float]:
    channels = pd.read_csv(channels_path, sep="\t", dtype=str, keep_default_na=False)
    events = pd.read_csv(events_path, sep="\t")
    required_channel_columns = {"name", "status", "status_description"}
    if not required_channel_columns.issubset(channels.columns):
        raise ValueError(f"Missing channel columns in {channels_path}")
    if not {"onset", "trial_type"}.issubset(events.columns):
        raise ValueError(f"Missing event columns in {events_path}")

    good_rows = channels[channels["status"].str.lower() == "good"]
    good_channels = good_rows["name"].tolist()
    soz_channels = set(
        good_rows.loc[
            good_rows["status_description"].str.lower().str.contains("soz", regex=False),
            "name",
        ]
    )

    onset_rows = events[events["trial_type"].str.lower().str.contains("onset", regex=False)]
    if len(onset_rows) != 1:
        raise ValueError(f"Expected one seizure onset in {events_path}")
    seizure_onset = float(onset_rows.iloc[0]["onset"])

    offset_rows = events[events["trial_type"].str.lower().str.contains("offset", regex=False)]
    seizure_duration = (
        float(offset_rows.iloc[0]["onset"]) - seizure_onset
        if len(offset_rows) == 1
        else float("inf")
    )
    return good_channels, soz_channels, seizure_onset, seizure_duration


def load_hup_recording(
    dataset_root: str | Path,
    subject: str,
    run: int,
    seconds_before_onset: float,
    seconds_after_onset: float,
    lowpass_hz: float,
    notch_hz: float,
    target_sampling_rate: float,
    window_seconds: float,
    step_seconds: float,
) -> HUPRecording:
    root = Path(dataset_root).resolve()
    edf_path, channels_path, events_path = _recording_paths(root, subject, run)
    good_channels, soz_channels, seizure_onset, seizure_duration = _read_metadata(
        channels_path, events_path
    )

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    missing_channels = sorted(set(good_channels) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Channels missing from EDF: {missing_channels}")

    start_time = seizure_onset - seconds_before_onset
    stop_time = seizure_onset + seconds_after_onset
    if start_time < 0 or stop_time > raw.times[-1]:
        raise ValueError("Requested interval is outside the EDF recording")

    raw.pick(good_channels)
    raw.crop(tmin=start_time, tmax=stop_time, include_tmax=False)
    raw.notch_filter(freqs=[notch_hz], verbose="ERROR")
    raw.filter(l_freq=None, h_freq=lowpass_hz, verbose="ERROR")
    raw.resample(target_sampling_rate, verbose="ERROR")

    signal = raw.get_data().T
    channel_mean = signal.mean(axis=0, keepdims=True)
    channel_std = signal.std(axis=0, keepdims=True)
    if np.any(channel_std == 0):
        constant_channels = [raw.ch_names[index] for index in np.flatnonzero(channel_std[0] == 0)]
        raise ValueError(f"Constant channels after preprocessing: {constant_channels}")
    signal = (signal - channel_mean) / channel_std

    window_size = round(window_seconds * raw.info["sfreq"])
    step_size = round(step_seconds * raw.info["sfreq"])
    bounds = sliding_window_bounds(signal.shape[0], window_size, step_size)
    windows = np.stack([signal[start:stop] for start, stop in bounds]).astype(np.float32)
    window_starts = np.asarray(
        [-seconds_before_onset + start / raw.info["sfreq"] for start, _ in bounds],
        dtype=np.float64,
    )
    ictal_mask = (window_starts >= 0) & (window_starts < seizure_duration)
    soz_mask = np.asarray([name in soz_channels for name in raw.ch_names], dtype=np.bool_)
    if not np.any(soz_mask):
        raise ValueError("No clinical SOZ channels were found")

    return HUPRecording(
        windows=windows,
        channel_names=list(raw.ch_names),
        soz_mask=soz_mask,
        window_start_seconds=window_starts,
        ictal_mask=ictal_mask,
        sampling_rate=float(raw.info["sfreq"]),
        seizure_duration=seizure_duration,
    )
