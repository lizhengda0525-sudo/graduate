import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an HUP EDF interval")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config["data"]
    dataset_root = (PROJECT_ROOT / data["dataset_root"]).resolve()
    output = (PROJECT_ROOT / data["signal_segment_path"]).resolve()
    if not output.is_relative_to((PROJECT_ROOT / "results").resolve()):
        raise ValueError("signal_segment_path must be inside the project results directory")
    if output.exists():
        raise FileExistsError(f"signal segment already exists: {output}")

    subject = data["subject"]
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"
    ieeg_dir = dataset_root / subject / "ses-presurgery" / "ieeg"
    pattern = f"{subject}_ses-presurgery_task-ictal_*_run-{data['run']:02d}_ieeg.edf"
    matches = list(ieeg_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one EDF for {subject} run {data['run']}")
    edf_path = matches[0]
    stem = edf_path.name.removesuffix("_ieeg.edf")
    channels = pd.read_csv(edf_path.with_name(f"{stem}_channels.tsv"), sep="\t")
    events = pd.read_csv(edf_path.with_name(f"{stem}_events.tsv"), sep="\t")
    names = channels.loc[channels["status"].str.lower() == "good", "name"].tolist()
    onset_rows = events[events["trial_type"].str.lower().str.contains("onset", regex=False)]
    if len(onset_rows) != 1:
        raise ValueError("expected one seizure onset")
    onset = float(onset_rows.iloc[0]["onset"])
    start_seconds = onset - data["seconds_before_onset"]
    stop_seconds = onset + data["seconds_after_onset"]
    if start_seconds < 0 or stop_seconds <= start_seconds:
        raise ValueError("requested interval is invalid")

    with pyedflib.EdfReader(str(edf_path)) as reader:
        labels = reader.getSignalLabels()
        missing = sorted(set(names) - set(labels))
        if missing:
            raise ValueError(f"channels are missing from EDF: {missing}")
        frequencies = {reader.getSampleFrequency(labels.index(name)) for name in names}
        if len(frequencies) != 1:
            raise ValueError("selected channels have different sampling rates")
        sampling_rate = frequencies.pop()
        start_sample = round(start_seconds * sampling_rate)
        sample_count = round((stop_seconds - start_seconds) * sampling_rate)
        if start_sample + sample_count > min(reader.getNSamples()):
            raise ValueError("requested interval is outside the EDF recording")
        signal = np.stack(
            [
                reader.readSignal(labels.index(name), start=start_sample, n=sample_count)
                for name in names
            ],
            axis=1,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        signal=signal,
        channel_names=np.asarray(names),
        sampling_rate=np.asarray(sampling_rate),
        start_seconds=np.asarray(start_seconds),
        stop_seconds=np.asarray(stop_seconds),
    )
    print(f"saved {signal.shape} signal samples to {output}")


if __name__ == "__main__":
    main()
