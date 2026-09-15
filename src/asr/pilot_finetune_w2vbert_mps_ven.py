"""Pilot Wav2Vec2-BERT (w2v-BERT 2.0) fine-tune for Tshivenda on Apple Silicon (MPS).

Fifth ASR model family attempt. AfriHuBERT (masked cluster prediction) and
MMS (Wav2Vec2-style contrastive, same family as our working XLS-R) both
collapsed into predicting blank on every frame - see
notes/pilot-ven-results.md. This tries a genuinely different backbone:
`facebook/w2v-bert-2.0` is Conformer-based (convolution + self-attention,
not a plain transformer over CNN features like Wav2Vec2/HuBERT), trained
with a hybrid contrastive + masked-prediction objective, on 4.5M hours
across 143+ languages - both a different architecture and a much larger,
broader pretraining pool than anything tried so far. No confirmed Tshivenda
coverage (same situation as XLS-R and Whisper, both already working here) -
HF's own fine-tuning writeup for this checkpoint demonstrates the same
"language not in pretraining, fine-tune anyway" approach on Mongolian.

Reuses the existing tokenizers/ven/ CTC tokenizer unchanged - Wav2Vec2-BERT
in transformers pairs with Wav2Vec2CTCTokenizer, same as XLS-R and MMS.
The feature extractor differs though: w2v-BERT expects log-mel-style
`input_features` (SeamlessM4TFeatureExtractor) rather than raw-waveform
`input_values` (Wav2Vec2FeatureExtractor) - the prepare/collator functions
below are adjusted accordingly, closer to the Whisper pilot script's
pattern than the Wav2Vec2/MMS ones.

NOT the real Stage 1 run (needs a CUDA GPU). This is a reduced overnight
pilot on the M4 to (a) prove the training loop end-to-end with this
checkpoint, (b) get a first fine-tuned WER, (c) surface bugs before
spending Colab hours - same role as the other pilots.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python src/asr/pilot_finetune_w2vbert_mps_ven.py
    ... --train-clips 50 --eval-clips 20 --epochs 1   # smoke test
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, Features, Value, load_dataset
from jiwer import cer, wer
from transformers import (
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "dataset" / "processed"
TOKENIZER_DIR = REPO_ROOT / "tokenizers" / "ven"
OUTPUT_DIR = REPO_ROOT / "results" / "w2vbert-ven-pilot"
BASE_CHECKPOINT = "facebook/w2v-bert-2.0"


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"w2v-BERT pilot fine-tune | device: {device} | train clips: {args.train_clips} | "
          f"eval clips: {args.eval_clips} | epochs: {args.epochs}")

    train_files = [str(DATA / "nchlt_ven" / "train.csv")]
    eval_files = [str(DATA / "nchlt_ven" / "validation.csv")]
    if args.include_anv:
        train_files.append(str(DATA / "anv_ven" / "train.csv"))
        eval_files.append(str(DATA / "anv_ven" / "dev.csv"))

    features = Features({"audio": Audio(sampling_rate=16000), "transcript": Value("string")})
    ds = load_dataset("csv", data_files={"train": train_files, "eval": eval_files},
                      features=features)
    # shuffle before select so multi-file loads are mixed, not blockwise
    ds["train"] = ds["train"].shuffle(seed=42).select(range(min(args.train_clips, len(ds["train"]))))
    ds["eval"] = ds["eval"].shuffle(seed=42).select(range(min(args.eval_clips, len(ds["eval"]))))

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(BASE_CHECKPOINT)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare(batch):
        samples = batch["audio"].get_all_samples()
        array = samples.data.numpy().squeeze()
        features = processor(audio=array, sampling_rate=16000)
        batch["input_features"] = features.input_features[0]
        batch["input_length"] = len(array) / 16000
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names)

    if args.max_input_seconds:
        before = {k: len(v) for k, v in ds.items()}
        ds = ds.filter(lambda x: x["input_length"] <= args.max_input_seconds)
        print(f"clip-length cap {args.max_input_seconds}s: "
              f"train {before['train']}->{len(ds['train'])}, "
              f"eval {before['eval']}->{len(ds['eval'])}")

    from dataclasses import dataclass
    from typing import Dict, List, Union

    @dataclass
    class Collator:
        processor: Wav2Vec2BertProcessor

        def __call__(self, feats: List[Dict[str, Union[List[int], torch.Tensor]]]):
            input_features = [{"input_features": f["input_features"]} for f in feats]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            labels = [{"input_ids": f["labels"]} for f in feats]
            labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
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
        # continue from an earlier pilot checkpoint - CTC head already sized to our vocab
        model = Wav2Vec2BertForCTC.from_pretrained(args.resume_from)
        print(f"resumed weights from {args.resume_from}")
    else:
        model = Wav2Vec2BertForCTC.from_pretrained(
            BASE_CHECKPOINT,
            ctc_loss_reduction="mean", ctc_zero_infinity=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
        )
    # no freeze_feature_encoder() here: unlike Wav2Vec2/MMS, w2v-BERT takes
    # precomputed log-mel features as input (like Whisper), so there is no
    # raw-waveform CNN feature encoder module to freeze
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
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        fp16=False,  # not supported on MPS
        gradient_checkpointing=True,  # essential on 24GB unified memory
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    final = trainer.evaluate()
    print("\n== w2v-BERT pilot result ==")
    print(f"eval WER: {final.get('eval_wer'):.3f} | eval CER: {final.get('eval_cer'):.3f}")
    print("(zero-shot Whisper Large v3 baseline on NCHLT test: WER 1.108, CER 0.763)")
    print("(Wav2Vec2 XLS-R-300M pilot v2 on NCHLT test: WER 0.332, CER 0.074)")
    print("(Whisper pilot v2 on NCHLT test: WER 0.182, CER 0.048)")
    print("(MMS pilot v1: total blank collapse, WER 0.971/CER 0.961)")

    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"model saved -> {out_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-clips", type=int, default=5000)
    parser.add_argument("--eval-clips", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-input-seconds", type=float, default=10.0,
                        help="drop clips longer than this (memory cap for MPS); 0 disables")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-from", default=None,
                        help="path to a previous pilot final dir to continue training from")
    parser.add_argument("--include-anv", action="store_true",
                        help="mix ANV train/dev CSVs into the data (still capped by max-input-seconds)")
    args = parser.parse_args()
    main(args)
