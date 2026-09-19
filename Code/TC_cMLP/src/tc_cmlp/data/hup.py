from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from tc_cmlp.data.hup_recording import HUPRecording


def _recording_paths(dataset_root: Path, subject: str, run: int) -> tuple[Path, Path, Path]:
    subject_name = subject if subject.startswith("sub-") else f"sub-{subject}"
    ieeg_dir = dataset_root / subject_name / "ses-presurgery" / "ieeg"
    pattern = f"{subject_name}_ses-presurgery_task-ictal_*_run-{run:02d}_ieeg.edf"
    matches = list(ieeg_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one EDF for {subject_name} run {run}, found {len(matches)}"
        )
    edf_path = matches[0]
    stem = edf_path.name.removesuffix("_ieeg.edf")
    return (
        edf_path,
        edf_path.with_name(f"{stem}_channels.tsv"),
        edf_path.with_name(f"{stem}_events.tsv"),
    )


def _metadata(
    channels_path: Path,
    events_path: Path,
) -> tuple[list[str], set[str], float, float]:
    channels = pd.read_csv(channels_path, sep="\t", dtype=str, keep_default_na=False)
    events = pd.read_csv(events_path, sep="\t")
    if not {"name", "status", "status_description"}.issubset(channels.columns):
        raise ValueError(f"required channel fields are missing from {channels_path}")
    if not {"onset", "trial_type"}.issubset(events.columns):
        raise ValueError(f"required event fields are missing from {events_path}")

    good_rows = channels[channels["status"].str.lower() == "good"]
    channel_names = good_rows["name"].tolist()
    soz_names = set(
        good_rows.loc[
            good_rows["status_description"].str.lower().str.contains("soz", regex=False),
            "name",
        ]
    )
    onset_rows = events[events["trial_type"].str.lower().str.contains("onset", regex=False)]
    offset_rows = events[events["trial_type"].str.lower().str.contains("offset", regex=False)]
    if len(onset_rows) != 1 or len(offset_rows) != 1:
        raise ValueError(f"expected one seizure onset and offset in {events_path}")
    seizure_onset = float(onset_rows.iloc[0]["onset"])
    seizure_offset = float(offset_rows.iloc[0]["onset"])
    if seizure_offset <= seizure_onset:
        raise ValueError("seizure offset must follow onset")
    if not channel_names or not soz_names:
        raise ValueError("good channels and clinical SOZ channels are required")
    return channel_names, soz_names, seizure_onset, seizure_offset


def load_hup_recording(
    dataset_root: str | Path,
    subject: str,
    run: int,
    seconds_before_onset: float,
    seconds_after_onset: float,
    lowpass_hz: float,
    lowpass_order: int,
    notch_hz: float,
    notch_quality_factor: float,
    target_sampling_rate: float,
    window_seconds: float,
    step_seconds: float,
    signal_segment_path: str | Path,
    progress: bool = False,
) -> HUPRecording:
    if min(
        seconds_before_onset,
        seconds_after_onset,
        lowpass_hz,
        notch_hz,
        notch_quality_factor,
        target_sampling_rate,
        window_seconds,
        step_seconds,
    ) <= 0:
        raise ValueError("recording settings must be positive")
    if lowpass_order < 1:
        raise ValueError("lowpass_order must be positive")
    root = Path(dataset_root).resolve()
    _, channels_path, events_path = _recording_paths(root, subject, run)
    channel_names, soz_names, seizure_onset, seizure_offset = _metadata(
        channels_path, events_path
    )

    start_time = seizure_onset - seconds_before_onset
    stop_time = seizure_onset + seconds_after_onset
    if progress:
        print("正在读取已提取的信号区间...", flush=True)
    with np.load(signal_segment_path, allow_pickle=False) as segment:
        signal = segment["signal"]
        saved_names = segment["channel_names"].tolist()
        sampling_rate = float(segment["sampling_rate"])
        saved_start = float(segment["start_seconds"])
        saved_stop = float(segment["stop_seconds"])
    if saved_names != channel_names or saved_start != start_time or saved_stop != stop_time:
        raise ValueError("cached signal segment does not match HUP metadata")
    if signal.shape != (round((stop_time - start_time) * sampling_rate), len(channel_names)):
        raise ValueError("cached signal segment has incompatible dimensions")
    if not np.all(np.isfinite(signal)):
        raise ValueError("cached signal segment contains non-finite values")
    if not 0 < notch_hz < sampling_rate / 2:
        raise ValueError("notch_hz must be below the input Nyquist frequency")
    if not 0 < lowpass_hz < min(sampling_rate, target_sampling_rate) / 2:
        raise ValueError("lowpass_hz must be below both Nyquist frequencies")

    if progress:
        print("正在进行陷波滤波...", flush=True)
    notch_b, notch_a = iirnotch(notch_hz, notch_quality_factor, fs=sampling_rate)
    signal = filtfilt(notch_b, notch_a, signal, axis=0)
    if progress:
        print("正在进行低通滤波...", flush=True)
    lowpass_sos = butter(lowpass_order, lowpass_hz, fs=sampling_rate, output="sos")
    signal = sosfiltfilt(lowpass_sos, signal, axis=0)
    if progress:
        print("正在重采样和生成时间窗口...", flush=True)
    ratio = Fraction(str(target_sampling_rate)) / Fraction(str(sampling_rate))
    signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=0, padtype="line")
    channel_mean = signal.mean(axis=0, keepdims=True)
    channel_std = signal.std(axis=0, keepdims=True)
    if np.any(channel_std == 0):
        raise ValueError("constant channels remain after preprocessing")
    signal = (signal - channel_mean) / channel_std

    window_size = round(window_seconds * target_sampling_rate)
    step_size = round(step_seconds * target_sampling_rate)
    if not 1 <= step_size <= window_size <= signal.shape[0]:
        raise ValueError("window and step sizes are incompatible with the recording")
    bounds = [
        (start, start + window_size)
        for start in range(0, signal.shape[0] - window_size + 1, step_size)
    ]
    windows = np.stack([signal[start:stop] for start, stop in bounds]).astype(np.float32)
    window_starts = np.asarray(
        [-seconds_before_onset + start / target_sampling_rate for start, _ in bounds],
        dtype=np.float64,
    )
    ictal_mask = (window_starts >= 0) & (window_starts < seizure_offset - seizure_onset)
    soz_mask = np.asarray([name in soz_names for name in channel_names], dtype=np.bool_)
    if not np.any(ictal_mask) or not np.any(soz_mask):
        raise ValueError("recording must contain ictal windows and clinical SOZ channels")
    return HUPRecording(
        windows=windows,
        channel_names=channel_names,
        soz_mask=soz_mask,
        window_start_seconds=window_starts,
        ictal_mask=ictal_mask,
        sampling_rate=target_sampling_rate,
    )
