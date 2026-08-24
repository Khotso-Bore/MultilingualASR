"""Pilot AfriHuBERT fine-tune for Tshivenda on Apple Silicon (MPS).

Third distinct ASR model family for the comparison Seani asked for (Wav2Vec2
and Whisper are the other two - HuBERT uses a different self-supervised
pretraining objective: masked prediction of offline k-means cluster labels,
vs Wav2Vec2's contrastive prediction over quantized latents). Checked and
confirmed ajesujoba/AfriHuBERT covers Tshivenda ('ven' in its 1240 language
tags) and loads via standard transformers (HubertModel/HubertForCTC) - no
extra toolkit needed, unlike e.g. ESPnet-native models.

Mirrors src/pilot_finetune_mps.py's structure/settings exactly (same data,
tokenizer, collator, MPS memory-safety config) with the model swapped.

NOT the real Stage 1 run - see notebooks/colab_hubert_ven.ipynb for the full
Colab training. This is a reduced overnight pilot on the M4 to prove the
recipe works before spending Colab hours, same pattern as the Wav2Vec2 pilot.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python src/pilot_finetune_hubert_mps.py
    ... --train-clips 50 --eval-clips 20 --epochs 1   # smoke test
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, Features, Value, load_dataset
from jiwer import cer, wer
from transformers import (
    HubertForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "dataset" / "processed"
TOKENIZER_DIR = REPO_ROOT / "tokenizers" / "ven"
OUTPUT_DIR = REPO_ROOT / "results" / "hubert-ven-pilot"
BASE_CHECKPOINT = "ajesujoba/AfriHuBERT"


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"AfriHuBERT pilot fine-tune | device: {device} | train clips: {args.train_clips} | "
          f"eval clips: {args.eval_clips} | epochs: {args.epochs}")

    train_files = [str(DATA / "nchlt_ven" / "train.csv")]
    eval_files = [str(DATA / "nchlt_ven" / "validation.csv")]
    if args.include_anv:
        train_files.append(str(DATA / "anv_ven" / "train.csv"))
        eval_files.append(str(DATA / "anv_ven" / "dev.csv"))

    features = Features({"audio": Audio(sampling_rate=16000), "transcript": Value("string")})
    ds = load_dataset("csv", data_files={"train": train_files, "eval": eval_files},
                      features=features)
    ds["train"] = ds["train"].shuffle(seed=42).select(range(min(args.train_clips, len(ds["train"]))))
    ds["eval"] = ds["eval"].shuffle(seed=42).select(range(min(args.eval_clips, len(ds["eval"]))))

    # HuBERT in transformers pairs with the same Wav2Vec2 processor classes -
    # this is the standard HF pattern, not a workaround
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare(batch):
        samples = batch["audio"].get_all_samples()
        array = samples.data.numpy().squeeze()
        batch["input_values"] = processor(array, sampling_rate=16000).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names)

    if args.max_input_seconds:
        max_len = int(args.max_input_seconds * 16000)
        before = {k: len(v) for k, v in ds.items()}
        ds = ds.filter(lambda x: x["input_length"] <= max_len)
        print(f"clip-length cap {args.max_input_seconds}s: "
              f"train {before['train']}->{len(ds['train'])}, "
              f"eval {before['eval']}->{len(ds['eval'])}")

    from dataclasses import dataclass
    from typing import Dict, List, Union

    @dataclass
    class Collator:
        processor: Wav2Vec2Processor

        def __call__(self, feats: List[Dict[str, Union[List[int], torch.Tensor]]]):
            inputs = [{"input_values": f["input_values"]} for f in feats]
            labels = [{"input_ids": f["labels"]} for f in feats]
            batch = self.processor.pad(inputs, padding=True, return_tensors="pt")
            labels_batch = self.processor.pad(labels=labels, padding=True, return_tensors="pt")
            batch["labels"] = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)
            return batch

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": wer(label_str, pred_str), "cer": cer(label_str, pred_str)}

    if args.resume_from:
        model = HubertForCTC.from_pretrained(args.resume_from, attn_implementation="eager")
        print(f"resumed weights from {args.resume_from}")
    else:
        model = HubertForCTC.from_pretrained(
            BASE_CHECKPOINT,
            ctc_loss_reduction="mean", ctc_zero_infinity=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
            attn_implementation="eager",  # MPS's SDPA path errors on dropout - see train_classifier.py
        )
    if not args.unfreeze_feature_encoder:
        model.freeze_feature_encoder()

    if args.disfavor_blank_init and not args.resume_from:
        # counter CTC blank collapse: the freshly-initialized head's blank
        # logit starts on equal footing with every other class, and blank is
        # the "safe" low-loss local minimum CTC gradient descent tends to
        # fall into first. Push the blank bias down at init so the model
        # starts slightly favouring non-blank predictions instead.
        with torch.no_grad():
            model.lm_head.bias[processor.tokenizer.pad_token_id] -= args.blank_bias_penalty
        print(f"disfavoring blank at init: pad_token_id={processor.tokenizer.pad_token_id} "
              f"bias -= {args.blank_bias_penalty}")

    model = model.to(device)

    out_dir = OUTPUT_DIR if not args.resume_from else OUTPUT_DIR.parent / (OUTPUT_DIR.name + "-v2")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=20,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        push_to_hub=False,
        report_to=[],
        use_cpu=(device == "cpu"),
        dataloader_num_workers=0,
    )

    from transformers import EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        data_collator=Collator(processor=processor),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    final = trainer.evaluate()
    print("\n== afrihubert pilot result ==")
    print(f"eval WER: {final.get('eval_wer'):.3f} | eval CER: {final.get('eval_cer'):.3f}")
    print("(zero-shot Whisper Large v3 baseline on NCHLT test: WER 1.108, CER 0.763)")
    print("(Wav2Vec2 pilot v2 on NCHLT test: WER 0.332, CER 0.074)")

    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"model saved -> {out_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-clips", type=int, default=5000)
    parser.add_argument("--eval-clips", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-input-seconds", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--include-anv", action="store_true")
    parser.add_argument("--unfreeze-feature-encoder", action="store_true",
                        help="diagnostic: don't freeze the conv feature encoder")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--disfavor-blank-init", action="store_true",
                        help="push the CTC blank/pad logit down at init to counter blank collapse")
    parser.add_argument("--blank-bias-penalty", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    main(args)
