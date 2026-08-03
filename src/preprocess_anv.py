"""Download and preprocess ANV/Swivuriso Tshivenda speech for ASR fine-tuning.

Per the Tshivenda EDA findings (notebooks/za_next_voices_eda.ipynb):
- ANV audio is 48 kHz mono; it is resampled to 16 kHz here (via the datasets
  library's Audio cast, which the fine-tuning notebook also relies on).
- 8,028 clips (20%) exceed 30 s. A clip's transcript cannot be split without
  forced alignment, so clips > 30 s are EXCLUDED from the training CSVs by
  default. With --segment-long, their audio is additionally VAD-segmented
  (WebRTC VAD) into 5-30 s windows written WITHOUT transcripts to
  segments.csv, for later forced alignment or unsupervised use.
- SNR needs no filtering (minimum in the corpus is 30 dB).
- Transcripts are normalised with src/text_norm.py.

Output: dataset/processed/anv_ven/<split>.csv with columns
    audio       absolute path to the 16 kHz wav (under dataset/processed/anv_ven/)
    transcript  normalised text

Note: this dataset is gated on Hugging Face - run `hf auth login` and accept
the terms on the dataset page first.

Usage:
    python src/preprocess_anv.py                  # full run (tens of GB)
    python src/preprocess_anv.py --limit 5        # quick verification subset
    python src/preprocess_anv.py --segment-long   # also VAD-segment >30s clips
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import normalize_transcript

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "dataset" / "processed" / "anv_ven"

DATASET = "dsfsi-anv/za-african-next-voices-compressed"
LANG = "ven"
SPLITS = ["train", "dev", "dev_test"]
TARGET_SR = 16000
MAX_DURATION_S = 30.0
MIN_SEGMENT_S = 5.0

VAD_FRAME_MS = 30
VAD_AGGRESSIVENESS = 2
MIN_SILENCE_FRAMES = 10  # 300 ms of silence = cut candidate


def vad_segments(array, sr):
    """Split a long mono float32 clip into 5-30 s windows on VAD silences."""
    import webrtcvad
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    frame_len = int(sr * VAD_FRAME_MS / 1000)
    pcm16 = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)

    n_frames = len(pcm16) // frame_len
    voiced = []
    for i in range(n_frames):
        frame = pcm16[i * frame_len:(i + 1) * frame_len].tobytes()
        voiced.append(vad.is_speech(frame, sr))

    # cut points = centres of silence runs of >= MIN_SILENCE_FRAMES
    cuts = [0]
    run_start = None
    for i, v in enumerate(voiced):
        if not v:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= MIN_SILENCE_FRAMES:
                cuts.append((run_start + i) // 2 * frame_len)
            run_start = None
    cuts.append(len(pcm16))

    # merge inter-cut chunks greedily into windows of MIN_SEGMENT_S..MAX_DURATION_S
    max_len = int(MAX_DURATION_S * sr)
    min_len = int(MIN_SEGMENT_S * sr)
    segments = []
    start = cuts[0]
    for nxt in cuts[1:]:
        # emit hard windows if no usable silence appeared in time
        while nxt - start > max_len:
            segments.append((start, start + max_len))
            start += max_len
        if nxt == cuts[-1] or nxt - start >= min_len:
            segments.append((start, nxt))
            start = nxt
    # drop tail fragments shorter than the minimum
    return [(s, e) for s, e in segments if e - s >= min_len]


def process_split(split, limit=None, segment_long=False):
    ds = load_dataset(DATASET, LANG, split=split, streaming=True)
    if limit:
        ds = ds.take(limit)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    wav_dir = OUT_DIR / split
    wav_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = wav_dir / "segments"

    kept = long_skipped = dropped_empty = n_segments = 0
    csv_path = OUT_DIR / f"{split}.csv"
    seg_csv_path = OUT_DIR / f"{split}_segments.csv"

    seg_writer = None
    seg_file = None
    if segment_long:
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg_file = open(seg_csv_path, "w", newline="", encoding="utf-8")
        seg_writer = csv.writer(seg_file)
        seg_writer.writerow(["audio", "parent_audio_id"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio", "transcript"])

        for row in ds:
            samples = row["audio"].get_all_samples()
            array = samples.data.numpy()
            sr = int(samples.sample_rate)
            if array.ndim > 1:
                array = array.mean(axis=0)
            array = array.astype(np.float32)

            audio_id = row["audio_id"]
            duration = len(array) / sr

            if duration > MAX_DURATION_S:
                long_skipped += 1
                if segment_long:
                    for k, (s, e) in enumerate(vad_segments(array, sr)):
                        seg_path = seg_dir / f"{audio_id}_{k:03d}.wav"
                        sf.write(seg_path, array[s:e], sr)
                        seg_writer.writerow([str(seg_path), audio_id])
                        n_segments += 1
                continue

            transcript = normalize_transcript(row["transcript"])
            if not transcript:
                dropped_empty += 1
                continue

            wav_path = wav_dir / f"{audio_id}.wav"
            sf.write(wav_path, array, sr)
            writer.writerow([str(wav_path), transcript])
            kept += 1

    if seg_file:
        seg_file.close()

    msg = (f"{split}: kept {kept} (<= {MAX_DURATION_S:.0f}s, resampled to {TARGET_SR} Hz), "
           f"skipped {long_skipped} long clips, {dropped_empty} empty-after-normalisation "
           f"-> {csv_path}")
    if segment_long:
        msg += f" | wrote {n_segments} unlabelled VAD segments -> {seg_csv_path}"
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N rows per split (for testing)")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--segment-long", action="store_true",
                        help="also VAD-segment clips > 30s into unlabelled 5-30s windows")
    args = parser.parse_args()

    for split in args.splits:
        process_split(split, limit=args.limit, segment_long=args.segment_long)
