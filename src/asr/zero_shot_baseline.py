"""Zero-shot ASR baseline for Tshivenda (Sub-question 1 "before" numbers).

Transcribes preprocessed Tshivenda test clips with a pretrained Whisper model
(no fine-tuning) and scores WER, CER, MER (Match Error Rate), WIL (Word
Information Lost), and WIP (Word Information Preserved) with jiwer against
the gold transcripts - per Seani's request to report more than WER/CER.

Notes on interpretation:
- Whisper's language token set does NOT include Tshivenda, so the model runs
  with automatic language detection (it typically mis-detects a related or
  African language). High WER here is the expected zero-shot condition, not
  a bug - these numbers are the baseline that fine-tuning must beat.
- Wav2Vec2 XLS-R has no Tshivenda CTC head, so it has no meaningful zero-shot
  mode; this baseline is Whisper-only by design.
- Model predictions are passed through the same normalisation as the training
  transcripts (src/text_norm.py) before scoring, so WER is not inflated by
  punctuation/casing that Whisper adds.

Usage:
    python src/asr/zero_shot_baseline.py                          # whisper-small, 200/corpus
    python src/asr/zero_shot_baseline.py --limit 20               # quick check
    python src/asr/zero_shot_baseline.py --model openai/whisper-large-v3
"""

import argparse
import csv
import random
import time
from pathlib import Path

import soundfile as sf
import torch
from jiwer import cer, process_words
from transformers import pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_norm import normalize_transcript

REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_SETS = {
    "nchlt_test": REPO_ROOT / "dataset" / "processed" / "nchlt_ven" / "test.csv",
    "anv_dev_test": REPO_ROOT / "dataset" / "processed" / "anv_ven" / "dev_test.csv",
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model_name, limit, seed, n_examples, save_predictions=None):
    device = pick_device()
    print(f"model: {model_name} | device: {device} | sample per corpus: {limit} | seed: {seed}")

    asr = pipeline("automatic-speech-recognition", model=model_name, device=device)

    results = {}
    for name, csv_path in EVAL_SETS.items():
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        rng = random.Random(seed)
        sample = rng.sample(rows, min(limit, len(rows))) if limit else rows

        refs, hyps = [], []
        start = time.time()
        for i, row in enumerate(sample):
            audio, sr = sf.read(row["audio"])
            pred = asr({"array": audio, "sampling_rate": sr})["text"]
            refs.append(row["transcript"])
            hyps.append(normalize_transcript(pred))
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - start)
                print(f"  {name}: {i+1}/{len(sample)} ({rate:.1f} clips/s)")

        # jiwer cannot score an empty hypothesis string list-wide; keep pairs as-is,
        # empty hypotheses simply count as full deletions
        word_metrics = process_words(refs, hyps)
        corpus_wer = word_metrics.wer
        corpus_mer = word_metrics.mer
        corpus_wil = word_metrics.wil
        corpus_wip = word_metrics.wip
        corpus_cer = cer(refs, hyps)
        empty_preds = sum(1 for h in hyps if not h)
        elapsed = time.time() - start

        results[name] = {"wer": corpus_wer, "cer": corpus_cer, "mer": corpus_mer,
                         "wil": corpus_wil, "wip": corpus_wip, "n": len(sample),
                         "empty_preds": empty_preds, "seconds": elapsed}

        if save_predictions:
            out_path = Path(save_predictions)
            out_path.mkdir(parents=True, exist_ok=True)
            model_slug = model_name.split("/")[-1]
            pred_file = out_path / f"{model_slug}_{name}.csv"
            with open(pred_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["reference", "hypothesis"])
                writer.writerows(zip(refs, hyps))
            print(f"  predictions saved -> {pred_file}")

        print(f"\n== {name} ==")
        print(f"n={len(sample)}  WER={corpus_wer:.3f}  CER={corpus_cer:.3f}  "
              f"MER={corpus_mer:.3f}  WIL={corpus_wil:.3f}  WIP={corpus_wip:.3f}  "
              f"empty predictions={empty_preds}  ({elapsed:.0f}s)")
        for ref, hyp in list(zip(refs, hyps))[:n_examples]:
            print(f"  ref: {ref}")
            print(f"  hyp: {hyp}")
            print()

    print("== summary ==")
    print(f"{'corpus':<14} {'n':>5} {'WER':>7} {'CER':>7} {'MER':>7} {'WIL':>7} {'WIP':>7}")
    for name, r in results.items():
        print(f"{name:<14} {r['n']:>5} {r['wer']:>7.3f} {r['cer']:>7.3f} "
              f"{r['mer']:>7.3f} {r['wil']:>7.3f} {r['wip']:>7.3f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/whisper-small",
                        help="HF model id (use openai/whisper-large-v3 for the reported numbers)")
    parser.add_argument("--limit", type=int, default=200,
                        help="random clips per corpus (0 = full eval set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=3,
                        help="predicted-vs-reference pairs to print per corpus")
    parser.add_argument("--save-predictions", default=None, metavar="DIR",
                        help="persist all ref/hyp pairs to DIR/<model>_<corpus>.csv "
                             "(feeds corrupt_transcripts.py --error-model)")
    args = parser.parse_args()

    evaluate(args.model, args.limit, args.seed, args.examples,
             save_predictions=args.save_predictions)
