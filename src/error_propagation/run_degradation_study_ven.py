"""Error-propagation degradation study for Tshivenda (proposal §4.6, Objectives 5-6).

Answers Sub-question 5: how does ASR transcription error rate (WER)
quantitatively affect the misinformation classifier's F1/accuracy, and which
error type (substitution, deletion, insertion) is most harmful.

Reuses the exact grouped-CV split from src/classification/train_classifier_ven.py
(same GroupKFold, same seed) so numbers are directly comparable to the clean-text
Objective 4 result. For each fold: train once on clean text (unchanged from
train_classifier_ven.py), then evaluate that SAME trained model on the held-out
fold's text corrupted at each target WER, for "mixed" mode plus each ablation
(substitution-only, deletion-only, insertion-only) - producing the WER-vs-F1
degradation curve and the per-error-type comparison Objective 6 needs for a
practical reliability threshold.

Corruption uses src/error_propagation/corrupt_transcripts_ven.py. Pass
--error-model with a real ASR ref/hyp predictions CSV (from
zero_shot_baseline_ven.py --save-predictions) to use measured substitution/
deletion/insertion ratios and confusion pairs instead of the engine's default
60:25:15 split - do this once the Whisper full-scale predictions exist.

Usage:
    python src/error_propagation/run_degradation_study_ven.py --smoke-test   # 2 folds, 1 epoch
    python src/error_propagation/run_degradation_study_ven.py \
        --error-model results/preds_full/whisper-final_nchlt_test.csv \
        --folds 5 --model Davlan/afro-xlmr-base
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling: corrupt_transcripts_ven.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
from corrupt_transcripts_ven import ErrorModel, corrupt_sentence
from train_classifier_ven import DATA_CSV, RESULTS_DIR, load_rows

TARGET_WERS = [0.1, 0.2, 0.3, 0.4, 0.5]
MODES = ["mixed", "substitution", "deletion", "insertion"]


def tokenize_rows(tokenizer, rows):
    ds = Dataset.from_list(rows)
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=512,
                                      padding="max_length"), batched=True)


def train_fold(model_name, train_rows, eval_rows, epochs, batch_size, seed,
               learning_rate, freeze_base):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, attn_implementation="eager")

    if freeze_base:
        base = getattr(model, model.base_model_prefix)
        for p in base.parameters():
            p.requires_grad = False

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        return {"accuracy": accuracy_score(pred.label_ids, preds),
                "macro_f1": f1_score(pred.label_ids, preds, average="macro")}

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "tmp_degradation"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        num_train_epochs=epochs,
        seed=seed,
        use_cpu=(device == "cpu"),
        report_to=[],
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=tokenize_rows(tokenizer, train_rows),
                      eval_dataset=tokenize_rows(tokenizer, eval_rows),
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    return trainer, tokenizer


def evaluate_corrupted(trainer, tokenizer, eval_rows, target_wer, mode, seed, error_model):
    rng_rows = []
    import random
    rng = random.Random(seed)
    unigrams_source = [r["text"] for r in eval_rows]
    from collections import Counter
    counts = Counter()
    for t in unigrams_source:
        counts.update(t.split())
    unigrams = [w for w, _ in counts.most_common(2000)]

    for r in eval_rows:
        corrupted = corrupt_sentence(r["text"], target_wer, mode, rng, unigrams, error_model)
        rng_rows.append({"text": corrupted, "label": r["label"], "source_id": r["source_id"]})

    ds = tokenize_rows(tokenizer, rng_rows)
    metrics = trainer.evaluate(eval_dataset=ds)
    return metrics["eval_accuracy"], metrics["eval_macro_f1"]


def run_study(model_name, folds, epochs, batch_size, seed, learning_rate, freeze_base,
             error_model, target_wers, modes):
    rows = load_rows(DATA_CSV)
    groups = [r["source_id"] for r in rows]
    labels = [r["label"] for r in rows]

    gkf = GroupKFold(n_splits=folds)
    # {mode: {wer: [f1 per fold]}}, plus wer=0.0 clean baseline
    results = {mode: {0.0: []} for mode in modes}
    for mode in modes:
        for wer in target_wers:
            results[mode][wer] = []

    for i, (train_idx, eval_idx) in enumerate(gkf.split(rows, labels, groups)):
        train_rows = [rows[j] for j in train_idx]
        eval_rows = [rows[j] for j in eval_idx]
        assert not (set(groups[j] for j in train_idx) & set(groups[j] for j in eval_idx)), \
            f"fold {i}: source_id leaked across train/eval"

        print(f"\n=== fold {i+1}/{folds}: training on clean text (n_train={len(train_rows)}) ===")
        trainer, tokenizer = train_fold(model_name, train_rows, eval_rows, epochs, batch_size,
                                        seed, learning_rate, freeze_base)

        clean_metrics = trainer.evaluate()
        results["mixed"][0.0].append(clean_metrics["eval_macro_f1"])
        for mode in modes:
            if mode != "mixed":
                results[mode][0.0].append(clean_metrics["eval_macro_f1"])
        print(f"fold {i+1} clean: accuracy={clean_metrics['eval_accuracy']:.3f} "
              f"macro_f1={clean_metrics['eval_macro_f1']:.3f}")

        for mode in modes:
            for wer in target_wers:
                acc, f1 = evaluate_corrupted(trainer, tokenizer, eval_rows, wer, mode, seed, error_model)
                results[mode][wer].append(f1)
                print(f"fold {i+1} {mode} wer={wer:.1f}: accuracy={acc:.3f} macro_f1={f1:.3f}")

    print("\n== degradation curve (mean macro F1 across folds) ==")
    print(f"{'mode':<14} {'wer=0.0':>8}" + "".join(f" {'wer='+str(w):>8}" for w in target_wers))
    for mode in modes:
        row = [f"{np.mean(results[mode][0.0]):.3f}"]
        for wer in target_wers:
            row.append(f"{np.mean(results[mode][wer]):.3f}")
        print(f"{mode:<14} " + " ".join(f"{v:>8}" for v in row))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Davlan/afro-xlmr-base")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--freeze-base", action="store_true", default=True)
    parser.add_argument("--no-freeze-base", dest="freeze_base", action="store_false")
    parser.add_argument("--error-model", nargs="*", default=None,
                        help="ref/hyp prediction CSVs from zero_shot_baseline_ven.py --save-predictions")
    parser.add_argument("--smoke-test", action="store_true",
                        help="--folds 2 --epochs 1, overrides those flags")
    args = parser.parse_args()

    folds, epochs = (2, 1) if args.smoke_test else (args.folds, args.epochs)
    em = ErrorModel.from_prediction_files(args.error_model) if args.error_model else ErrorModel()

    run_study(args.model, folds, epochs, args.batch_size, args.seed, args.learning_rate,
             args.freeze_base, em, TARGET_WERS, MODES)
