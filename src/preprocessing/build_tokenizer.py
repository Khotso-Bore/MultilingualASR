"""Build the shared Tshivenda CTC tokenizer from the processed transcripts.

Character-level vocab built from the train CSVs of both corpora
(dataset/processed/nchlt_ven/train.csv and dataset/processed/anv_ven/train.csv).
Per the EDA findings the transcripts contain exactly 32 characters
(a-z, space, and the Tshivenda diacritics ḓ ḽ ṅ ṋ ṱ); the space becomes the
CTC word delimiter "|", plus [UNK] and [PAD] specials.

The tokenizer is saved to tokenizers/ven/ and committed - it is tiny, and the
whole team must use an identical vocab for WER numbers to be comparable.

Usage:
    python src/preprocessing/build_tokenizer.py
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSVS = [
    REPO_ROOT / "dataset" / "processed" / "nchlt_ven" / "train.csv",
    REPO_ROOT / "dataset" / "processed" / "anv_ven" / "train.csv",
]
OUT_DIR = REPO_ROOT / "tokenizers" / "ven"

EXPECTED_CHARS = set(" abcdefghijklmnopqrstuvwxyzḓḽṅṋṱ")


def main():
    chars = set()
    for path in TRAIN_CSVS:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                chars |= set(row["transcript"])

    unexpected = chars - EXPECTED_CHARS
    missing = EXPECTED_CHARS - chars
    if unexpected:
        raise SystemExit(f"unexpected characters in transcripts: {sorted(unexpected)}")
    if missing:
        print(f"note: expected chars absent from train transcripts: {sorted(missing)}")

    # space -> CTC word delimiter
    vocab_chars = sorted(chars - {" "}) + ["|"]
    vocab = {c: i for i, c in enumerate(vocab_chars)}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    from transformers import Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        str(OUT_DIR / "vocab.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    tokenizer.save_pretrained(OUT_DIR)
    print(f"vocab size: {len(vocab)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
