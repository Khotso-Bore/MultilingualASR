"""Pilot Wav2Vec2 fine-tune for Tshivenda on Apple Silicon (MPS).

NOT the real Stage 1 run (that needs a CUDA GPU - see
notebooks/finetune_wav2vec2_ven.ipynb). This is a reduced overnight pilot on
the M4 to (a) prove the training loop end-to-end, (b) get a first fine-tuned
WER well below the 110.8% zero-shot baseline, (c) surface bugs before spending
Colab hours.

Reduced scope: subset of NCHLT train clips, few epochs, XLS-R-300M with the
shared committed tokenizer (tokenizers/ven/). CTC loss has no MPS kernel, so
run with PYTORCH_ENABLE_MPS_FALLBACK=1 (loss computes on CPU; the encoder,
which dominates compute, stays on the GPU).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python src/asr/pilot_finetune_wav2vec2_mps_ven.py
    ... --train-clips 50 --eval-clips 20 --epochs 1   # smoke test
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, Features, Value, load_dataset
from jiwer import cer, wer
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_augment_ven import SPEED_RATES, speed_perturb

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "dataset" / "processed"
TOKENIZER_DIR = REPO_ROOT / "tokenizers" / "ven"
OUTPUT_DIR = REPO_ROOT / "results" / "wav2vec2-ven-pilot"


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"pilot fine-tune | device: {device} | train clips: {args.train_clips} | "
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

    if args.augment:
        # speed perturbation (proposal Objective 2, §4.3): add 0.9x/1.1x copies
        # of every training clip - eval set is left untouched, we only want
        # more/varied training signal, not to change what "correct" means.
        paths = ds["train"].cast_column("audio", Audio(decode=False))["audio"]
        transcripts = ds["train"]["transcript"]
        expanded = []
        for p, t in zip(paths, transcripts):
            array, sr = sf.read(p["path"])
            assert sr == 16000, f"expected 16kHz audio, got {sr}Hz for {p['path']}"
            array = array.astype(np.float32)
            expanded.append({"array": array, "transcript": t})
            for rate in SPEED_RATES:
                expanded.append({"array": speed_perturb(array, rate), "transcript": t})
        ds["train"] = Dataset.from_list(expanded)
        print(f"augmentation: {len(paths)} clips -> {len(expanded)} "
              f"(speed {SPEED_RATES} added)")

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare(batch):
        if "array" in batch:
            array = np.asarray(batch["array"])
        else:
            samples = batch["audio"].get_all_samples()
            array = samples.data.numpy().squeeze()
        batch["input_values"] = processor(array, sampling_rate=16000).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    # train/eval may now have different column names (augment adds "array"
    # instead of "audio"), so map each split separately instead of ds.map()
    ds["train"] = ds["train"].map(prepare, remove_columns=ds["train"].column_names)
    ds["eval"] = ds["eval"].map(prepare, remove_columns=ds["eval"].column_names)

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
        # continue from an earlier pilot checkpoint - CTC head already sized to our vocab
        model = Wav2Vec2ForCTC.from_pretrained(args.resume_from)
        print(f"resumed weights from {args.resume_from}")
    else:
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            ctc_loss_reduction="mean", ctc_zero_infinity=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
        )
    model.freeze_feature_encoder()

    if args.spec_augment:
        # HuggingFace's own SpecAugment (Park et al., 2019) implementation -
        # config is read dynamically each forward pass, so setting it here
        # (fresh or resumed model) is enough, no need to touch model init.
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = args.mask_time_prob
        model.config.mask_feature_prob = args.mask_feature_prob
        print(f"SpecAugment enabled: mask_time_prob={args.mask_time_prob} "
              f"mask_feature_prob={args.mask_feature_prob}")

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
    print("\n== pilot result ==")
    print(f"eval WER: {final.get('eval_wer'):.3f} | eval CER: {final.get('eval_cer'):.3f}")
    print(f"(zero-shot Whisper Large v3 baseline on NCHLT test: WER 1.108, CER 0.763)")

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
    parser.add_argument("--max-input-seconds", type=float, default=10.0,
                        help="drop clips longer than this (memory cap for MPS); 0 disables")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-from", default=None,
                        help="path to a previous pilot final dir to continue training from")
    parser.add_argument("--include-anv", action="store_true",
                        help="mix ANV train/dev CSVs into the data (still capped by max-input-seconds)")
    parser.add_argument("--augment", action="store_true",
                        help="speed perturbation (Objective 2): add 0.9x/1.1x copies of every "
                             "training clip (3x the training data, eval set untouched)")
    parser.add_argument("--spec-augment", action="store_true",
                        help="enable HuggingFace's built-in SpecAugment (Objective 2) on the encoder")
    parser.add_argument("--mask-time-prob", type=float, default=0.05)
    parser.add_argument("--mask-feature-prob", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
