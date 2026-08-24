"""Download and preprocess NCHLT Tshivenda speech for ASR fine-tuning.

Per the Tshivenda EDA findings (notebooks/za_next_voices_eda.ipynb):
- NCHLT audio is already 16 kHz mono, so it is written out as-is.
- Clips shorter than 1 s are dropped (4 clips in the full corpus).
- Transcripts are normalised with src/text_norm_ven.py.

Output: dataset/processed/nchlt_ven/<split>.csv with columns
    audio       absolute path to the wav file (under dataset/raw/nchlt_ven/)
    transcript  normalised text

These CSVs drop into notebooks/finetune_wav2vec2.ipynb in place of its
streaming load (cell 10). With datasets >= 5, declare the Audio feature at
load time (cast_column on a string column is not supported):
    features = Features({"audio": Audio(sampling_rate=16000),
                         "transcript": Value("string")})
    load_dataset("csv", data_files={...}, features=features)

Usage:
    python src/preprocessing/preprocess_nchlt_ven.py                 # full run (~6.3 GB download)
    python src/preprocessing/preprocess_nchlt_ven.py --limit 30      # quick verification subset
    python src/preprocessing/preprocess_nchlt_ven.py --splits train  # single split
"""

import argparse
import csv
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_norm_ven import normalize_transcript

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw" / "nchlt_ven"
OUT_DIR = REPO_ROOT / "dataset" / "processed" / "nchlt_ven"

DATASET = "dsfsi-anv/multilingual-nchlt-dataset"
LANG = "ven"
SPLITS = ["train", "validation", "test"]
MIN_DURATION_S = 1.0


def process_split(split, limit=None):
    ds = load_dataset(DATASET, LANG, split=split, streaming=True)
    if limit:
        ds = ds.take(limit)

    wav_dir = RAW_DIR / split
    wav_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kept = dropped_short = dropped_empty = 0
    csv_path = OUT_DIR / f"{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio", "transcript"])

        for row in ds:
            samples = row["audio"].get_all_samples()
            array = samples.data.numpy()
            sr = int(samples.sample_rate)
            if array.ndim > 1:  # (channels, n) -> mono
                array = array.mean(axis=0)

            duration = len(array) / sr
            if duration < MIN_DURATION_S:
                dropped_short += 1
                continue

            transcript = normalize_transcript(row["text"])
            if not transcript:
                dropped_empty += 1
                continue

            wav_path = wav_dir / Path(row["filename"]).name
            sf.write(wav_path, array, sr)
            writer.writerow([str(wav_path), transcript])
            kept += 1

    print(f"{split}: kept {kept}, dropped {dropped_short} short (<{MIN_DURATION_S}s), "
          f"{dropped_empty} empty-after-normalisation -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N rows per split (for testing)")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = parser.parse_args()

    for split in args.splits:
        process_split(split, limit=args.limit)
