"""Evaluate a local fine-tuned CTC checkpoint on the Tshivenda eval sets.

Works with any CTC model checkpoint (Wav2Vec2, HuBERT, ...) via AutoModelForCTC,
not just Wav2Vec2 - despite the filename. Companion to zero_shot_baseline.py:
same eval sets (NCHLT test, ANV dev_test), same sampling (--limit clips,
--seed), same output normalisation - so the resulting WER/CER are directly
comparable across every model in the comparison (Whisper, Wav2Vec2, AfriHuBERT).

Usage:
    python src/asr/evaluate_wav2vec2_ven.py --checkpoint results/wav2vec2-ven-pilot/final \
        --save-predictions results/preds_pilot
    python src/asr/evaluate_wav2vec2_ven.py --checkpoint results/hubert-ven-pilot/final
"""

import argparse
import csv
import random
import time
from pathlib import Path

import soundfile as sf
import torch
from jiwer import cer, process_words
from transformers import AutoModelForCTC, Wav2Vec2Processor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))       # sibling: zero_shot_baseline.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/: text_norm.py
from text_norm_ven import normalize_transcript
from zero_shot_baseline_ven import EVAL_SETS, pick_device


def evaluate(checkpoint, limit, seed, n_examples, save_predictions=None):
    device = pick_device()
    print(f"checkpoint: {checkpoint} | device: {device} | sample per corpus: {limit} | seed: {seed}")

    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    # AutoModelForCTC dispatches on the checkpoint's own architecture, so this
    # loads Wav2Vec2, HuBERT, or any other CTC model checkpoint transparently
    model = AutoModelForCTC.from_pretrained(checkpoint, attn_implementation="eager").to(device).eval()

    results = {}
    for name, csv_path in EVAL_SETS.items():
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        rng = random.Random(seed)
        sample = rng.sample(rows, min(limit, len(rows))) if limit else rows

        refs, hyps = [], []
        start = time.time()
        for i, row in enumerate(sample):
            audio, sr = sf.read(row["audio"])
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred = processor.batch_decode(pred_ids)[0]
            refs.append(row["transcript"])
            hyps.append(normalize_transcript(pred))
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - start)
                print(f"  {name}: {i+1}/{len(sample)} ({rate:.1f} clips/s)")

        word_metrics = process_words(refs, hyps)
        corpus_wer = word_metrics.wer
        corpus_mer = word_metrics.mer
        corpus_wil = word_metrics.wil
        corpus_wip = word_metrics.wip
        corpus_cer = cer(refs, hyps)
        empty_preds = sum(1 for h in hyps if not h)
        results[name] = {"wer": corpus_wer, "cer": corpus_cer, "mer": corpus_mer,
                         "wil": corpus_wil, "wip": corpus_wip, "n": len(sample),
                         "empty_preds": empty_preds}

        print(f"\n== {name} ==")
        print(f"n={len(sample)}  WER={corpus_wer:.3f}  CER={corpus_cer:.3f}  "
              f"MER={corpus_mer:.3f}  WIL={corpus_wil:.3f}  WIP={corpus_wip:.3f}  "
              f"empty predictions={empty_preds}")
        for ref, hyp in list(zip(refs, hyps))[:n_examples]:
            print(f"  ref: {ref}")
            print(f"  hyp: {hyp}")
            print()

        if save_predictions:
            out_path = Path(save_predictions)
            out_path.mkdir(parents=True, exist_ok=True)
            slug = Path(checkpoint).name or "checkpoint"
            pred_file = out_path / f"wav2vec2-{slug}_{name}.csv"
            with open(pred_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["reference", "hypothesis"])
                writer.writerows(zip(refs, hyps))
            print(f"  predictions saved -> {pred_file}")

    print("== summary ==")
    print(f"{'corpus':<14} {'n':>5} {'WER':>7} {'CER':>7} {'MER':>7} {'WIL':>7} {'WIP':>7}")
    for name, r in results.items():
        print(f"{name:<14} {r['n']:>5} {r['wer']:>7.3f} {r['cer']:>7.3f} "
              f"{r['mer']:>7.3f} {r['wil']:>7.3f} {r['wip']:>7.3f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="path to a fine-tuned Wav2Vec2 dir (model + processor)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--save-predictions", default=None, metavar="DIR")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.limit, args.seed, args.examples,
             save_predictions=args.save_predictions)
