"""Batch run: real Tshivenda audio -> Whisper transcript -> misinformation verdict.

Same two checkpoints as src/demo/asr_to_classifier_pipeline_ven.py
(results/whisper-ven-pilot-v2/final, results/classifier/final), run over many
real NCHLT/ANV clips instead of one, with both models loaded once. Read that
script's module docstring first - the classifier has no ground truth to check
these verdicts against, since no audio exists for the actual real/fake
Vukuzenzele proxy articles the classifier was trained on. What this batch run
actually measures:

- real ASR WER/CER across many real clips, transcribed one at a time through
  a live pipeline (not the batched evaluation used elsewhere in this repo)
- whether the pipeline holds up running back-to-back over a real volume of
  audio without crashing, memory-leaking, or slowing down
- the classifier's verdict distribution and confidence on real (if
  domain-mismatched) transcribed speech, as a descriptive statistic, not an
  accuracy number

Usage:
    python src/demo/run_pipeline_batch_ven.py --limit 20                    # smoke test, ~1-2 min
    python src/demo/run_pipeline_batch_ven.py --limit 4000 --seed 42        # full run
    python src/demo/run_pipeline_batch_ven.py --limit 4000 --resume         # continue after an interruption
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from jiwer import cer, wer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_norm_ven import normalize_transcript

REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_CHECKPOINT = REPO_ROOT / "results" / "whisper-ven-pilot-v2" / "final"
CLASSIFIER_CHECKPOINT = REPO_ROOT / "results" / "classifier" / "final"
EVAL_SETS = {
    "nchlt_test": REPO_ROOT / "dataset" / "processed" / "nchlt_ven" / "test.csv",
    "anv_dev_test": REPO_ROOT / "dataset" / "processed" / "anv_ven" / "dev_test.csv",
}
OUT_DIR = REPO_ROOT / "results" / "demo_pipeline"
LABEL_NAMES = {0: "fake", 1: "real"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sample(limit_per_corpus, seed):
    sample = []
    for corpus, csv_path in EVAL_SETS.items():
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        rng = random.Random(seed)
        picked = rng.sample(rows, min(limit_per_corpus, len(rows))) if limit_per_corpus else rows
        for row in picked:
            sample.append({"corpus": corpus, "audio": row["audio"], "reference": row["transcript"]})
    return sample


def run(limit_per_corpus, seed, resume, out_csv):
    device = pick_device()
    print(f"device: {device} | up to {limit_per_corpus} clips per corpus | seed {seed}")

    print(f"loading ASR checkpoint {WHISPER_CHECKPOINT} ...")
    asr = pipeline("automatic-speech-recognition", model=str(WHISPER_CHECKPOINT), device=device)
    print(f"loading classifier checkpoint {CLASSIFIER_CHECKPOINT} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_CHECKPOINT))
    classifier = AutoModelForSequenceClassification.from_pretrained(str(CLASSIFIER_CHECKPOINT))
    classifier.to(device).eval()

    sample = load_sample(limit_per_corpus, seed)

    done_audio_paths = set()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    if resume and out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            done_audio_paths = {row["audio"] for row in csv.DictReader(f)}
        write_header = False
        print(f"resuming: {len(done_audio_paths)} clips already done, skipping those")

    remaining = [row for row in sample if row["audio"] not in done_audio_paths]
    print(f"{len(remaining)}/{len(sample)} clips left to run")

    fieldnames = ["corpus", "audio", "reference", "hypothesis", "wer", "cer", "label", "confidence"]
    start = time.time()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(remaining):
            audio, sr = sf.read(row["audio"])
            raw = asr({"array": audio, "sampling_rate": sr})["text"]
            hyp = normalize_transcript(raw)
            ref = normalize_transcript(row["reference"])
            clip_wer = wer(ref, hyp) if ref and hyp else (1.0 if ref != hyp else 0.0)
            clip_cer = cer(ref, hyp) if ref and hyp else (1.0 if ref != hyp else 0.0)

            inputs = tokenizer(hyp if hyp else " ", truncation=True, max_length=512,
                              padding="max_length", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = classifier(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            label_idx = int(torch.argmax(probs).item())

            writer.writerow({
                "corpus": row["corpus"], "audio": row["audio"], "reference": ref, "hypothesis": hyp,
                "wer": f"{clip_wer:.4f}", "cer": f"{clip_cer:.4f}",
                "label": LABEL_NAMES[label_idx], "confidence": f"{probs[label_idx].item():.4f}",
            })
            f.flush()

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta_min = (len(remaining) - (i + 1)) / rate / 60 if rate > 0 else float("nan")
                print(f"  {i+1}/{len(remaining)} ({rate:.2f} clips/s, ETA {eta_min:.0f} min)")

    print(f"done. wrote {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=20, help="max clips per corpus (nchlt_test, anv_dev_test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="skip clips already written to the output CSV")
    parser.add_argument("--out", default=str(OUT_DIR / "pipeline_batch_results.csv"))
    args = parser.parse_args()

    run(args.limit, args.seed, args.resume, Path(args.out))
