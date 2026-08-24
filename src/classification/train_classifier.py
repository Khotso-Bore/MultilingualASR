"""Fine-tune a Tshivenda misinformation classifier (proposal §4.5, Objective 4).

Trains on dataset/vukuzenzele/misinfo_proxy_ven.csv - the synthetic real/fake
proxy built by src/classification/build_misinfo_proxy.py, standing in for the inaccessible
Mukwevho et al. (2024) dataset (see notes/tshivenda-classifier-proxy.md).

With only 179 source articles (358 rows with their synthetic counterparts), a
single train/test split is too noisy to report responsibly. Instead this uses
grouped stratified k-fold cross-validation, grouped by source_id so a real
article and its own synthetic fake counterpart always land in the same fold
(never split across train/eval - that would leak the paired construction).

Models (proposal primary + comparison):
    Davlan/afro-xlmr-base   (AfroXLM-RoBERTa - primary, per MphayaNER's
                             finding that it is strongest for Tshivenda NLP)
    xlm-roberta-base        (comparison baseline)

Usage:
    python src/classification/train_classifier.py --model Davlan/afro-xlmr-base
    python src/classification/train_classifier.py --model xlm-roberta-base --folds 5
    python src/classification/train_classifier.py --model Davlan/afro-xlmr-base --folds 2 --epochs 1  # smoke test
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
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "dataset" / "vukuzenzele" / "misinfo_proxy_ven.csv"
RESULTS_DIR = REPO_ROOT / "results" / "classifier"


def load_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"text": row["text"], "label": int(row["label"]),
                        "source_id": int(row["source_id"])})
    return rows


def run_fold(model_name, train_rows, eval_rows, epochs, batch_size, seed, learning_rate=2e-5,
             freeze_base=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # eager attention: MPS's scaled_dot_product_attention path does not support
    # dropout and errors when the base is frozen (different kernel selection) -
    # eager works identically everywhere (MPS/CUDA/CPU), just slightly slower
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, attn_implementation="eager")

    if freeze_base:
        # linear-probe: only the classification head is trainable - far fewer
        # parameters than examples, much more stable on a ~150-example dataset
        base = getattr(model, model.base_model_prefix)
        for p in base.parameters():
            p.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"frozen base: {n_trainable:,} / {n_total:,} parameters trainable")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True)
    eval_ds = Dataset.from_list(eval_rows).map(tokenize, batched=True)

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        return {"accuracy": accuracy_score(pred.label_ids, preds),
                "macro_f1": f1_score(pred.label_ids, preds, average="macro")}

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "tmp"),
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
    from transformers import EarlyStoppingCallback
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_accuracy"], metrics["eval_macro_f1"]


def cross_validate(model_name, folds, epochs, batch_size, seed, learning_rate=2e-5,
                   freeze_base=False):
    rows = load_rows(DATA_CSV)
    groups = [r["source_id"] for r in rows]
    labels = [r["label"] for r in rows]

    gkf = GroupKFold(n_splits=folds)
    accs, f1s = [], []
    for i, (train_idx, eval_idx) in enumerate(gkf.split(rows, labels, groups)):
        train_rows = [rows[j] for j in train_idx]
        eval_rows = [rows[j] for j in eval_idx]
        # verify no group leakage between train and eval for this fold
        assert not (set(groups[j] for j in train_idx) & set(groups[j] for j in eval_idx)), \
            f"fold {i}: source_id leaked across train/eval"

        acc, f1 = run_fold(model_name, train_rows, eval_rows, epochs, batch_size, seed, learning_rate,
                           freeze_base)
        print(f"fold {i+1}/{folds}: n_train={len(train_rows)} n_eval={len(eval_rows)} "
              f"accuracy={acc:.3f} macro_f1={f1:.3f}")
        accs.append(acc)
        f1s.append(f1)

    print(f"\n== {model_name} ({folds}-fold grouped CV) ==")
    print(f"accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"macro F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    return accs, f1s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Davlan/afro-xlmr-base")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--freeze-base", action="store_true",
                        help="linear-probe: freeze the encoder, train only the classification head")
    args = parser.parse_args()

    cross_validate(args.model, args.folds, args.epochs, args.batch_size, args.seed, args.learning_rate,
                   args.freeze_base)
