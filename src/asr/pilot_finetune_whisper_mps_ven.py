"""Pilot Whisper fine-tune for Tshivenda on Apple Silicon (MPS).

Second half of the ASR comparison (Wav2Vec2 = contrastive CTC, Whisper =
weakly-supervised encoder-decoder). Zero-shot Whisper Large v3 was already
benchmarked (src/asr/zero_shot_baseline_ven.py); this is the actual fine-tuning that
was never built. Uses whisper-small for fast local pilot iteration - swap to
openai/whisper-large-v3 for the real reported numbers (much slower, GPU
recommended - see notebooks/colab_whisper_ven.ipynb once built).

Design decision - no Tshivenda language token: Whisper's tokenizer has no
<|ven|> token. Fine-tuning research on unsupported languages commonly uses
an existing language tag purely as a conditioning placeholder (the token's
embedding gets adapted during fine-tuning regardless of its literal name).
Using "sw" (Swahili) - the closest available Bantu-family language Whisper
supports - rather than an arbitrary unrelated one. Document this choice in
the report; it is a deliberate, defensible choice, not an oversight.

Unlike the CTC models, Whisper's tokenizer already represents Tshivenda text
fine (general BPE, not a fixed small vocab) - no custom tokenizer needed,
unlike tokenizers/ven/ built for the CTC models.

Usage:
    python src/asr/pilot_finetune_whisper_mps_ven.py
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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_augment_ven import SPEED_RATES, speed_perturb

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "dataset" / "processed"
OUTPUT_DIR = REPO_ROOT / "results" / "whisper-ven-pilot"
PLACEHOLDER_LANGUAGE = "sw"  # Swahili token as a Tshivenda placeholder - see module docstring


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Whisper pilot fine-tune | model: {args.model} | device: {device} | "
          f"train clips: {args.train_clips} | eval clips: {args.eval_clips} | epochs: {args.epochs}")

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

    if args.augment:
        # speed perturbation (proposal Objective 2, §4.3): add 0.9x/1.1x copies
        # of every training clip - eval set is left untouched.
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

    processor = WhisperProcessor.from_pretrained(
        args.model, language=PLACEHOLDER_LANGUAGE, task="transcribe")

    def prepare(batch):
        if "array" in batch:
            array = np.asarray(batch["array"])
        else:
            samples = batch["audio"].get_all_samples()
            array = samples.data.numpy().squeeze()
        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=16000).input_features[0]
        batch["input_length"] = len(array) / 16000
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    # train/eval may now have different column names (augment adds "array"
    # instead of "audio"), so map each split separately instead of ds.map()
    ds["train"] = ds["train"].map(prepare, remove_columns=ds["train"].column_names)
    ds["eval"] = ds["eval"].map(prepare, remove_columns=ds["eval"].column_names)

    if args.max_input_seconds:
        before = {k: len(v) for k, v in ds.items()}
        ds = ds.filter(lambda x: x["input_length"] <= args.max_input_seconds)
        print(f"clip-length cap {args.max_input_seconds}s: "
              f"train {before['train']}->{len(ds['train'])}, "
              f"eval {before['eval']}->{len(ds['eval'])}")

    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, feats: List[Dict[str, Union[List[int], torch.Tensor]]]):
            input_features = [{"input_features": f["input_features"]} for f in feats]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in feats]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)
            # drop the forced BOS token if the tokenizer already prepended one -
            # the model re-adds it via forced_decoder_ids at generation time
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer(label_str, pred_str), "cer": cer(label_str, pred_str)}

    if args.resume_from:
        model = WhisperForConditionalGeneration.from_pretrained(args.resume_from)
        print(f"resumed weights from {args.resume_from}")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)

    model.generation_config.language = PLACEHOLDER_LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False

    if args.spec_augment:
        # HuggingFace's own SpecAugment (Park et al., 2019) implementation,
        # same mechanism as the Wav2Vec2 pilot script - config read
        # dynamically each forward pass.
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = args.mask_time_prob
        model.config.mask_feature_prob = args.mask_feature_prob
        print(f"SpecAugment enabled: mask_time_prob={args.mask_time_prob} "
              f"mask_feature_prob={args.mask_feature_prob}")

    # LoRA (proposal Objective 3, §4.4): --lora wraps a fresh/--resume-from
    # base with new LoRA adapters (Stage 1); --lora-adapter-from loads a
    # previously-saved adapter on top of the base and continues training it
    # (Stage 2, continuing from Stage 1's adapter without touching the base
    # weights - this is what "prevents catastrophic forgetting" means here).
    if args.lora_adapter_from:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter_from, is_trainable=True)
        model.enable_input_require_grads()  # required for gradient_checkpointing + a frozen base
        print(f"loaded LoRA adapter from {args.lora_adapter_from} (continuing training)")
        model.print_trainable_parameters()
    elif args.lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    model = model.to(device)

    if args.out_name:
        out_dir = OUTPUT_DIR.parent / args.out_name
    else:
        out_dir = OUTPUT_DIR if not args.resume_from else OUTPUT_DIR.parent / (OUTPUT_DIR.name + "-v2")

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        generation_max_length=225,
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
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    final = trainer.evaluate()
    print("\n== whisper pilot result ==")
    print(f"eval WER: {final.get('eval_wer'):.3f} | eval CER: {final.get('eval_cer'):.3f}")
    print("(zero-shot Whisper Large v3 baseline on NCHLT test: WER 1.108, CER 0.763)")
    print("(Wav2Vec2 pilot v2 on NCHLT test: WER 0.332, CER 0.074)")

    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"model saved -> {out_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/whisper-small",
                        help="use openai/whisper-large-v3 for the reported numbers (slow locally)")
    parser.add_argument("--train-clips", type=int, default=5000)
    parser.add_argument("--eval-clips", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-input-seconds", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="freeze the audio encoder, only fine-tune the decoder")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--include-anv", action="store_true")
    parser.add_argument("--augment", action="store_true",
                        help="speed perturbation (Objective 2): add 0.9x/1.1x copies of every "
                             "training clip (3x the training data, eval set untouched)")
    parser.add_argument("--spec-augment", action="store_true",
                        help="enable HuggingFace's built-in SpecAugment (Objective 2) on the encoder")
    parser.add_argument("--mask-time-prob", type=float, default=0.05)
    parser.add_argument("--mask-feature-prob", type=float, default=0.05)
    parser.add_argument("--lora", action="store_true",
                        help="wrap a fresh/--resume-from base with new LoRA adapters (Objective 3, Stage 1)")
    parser.add_argument("--lora-adapter-from", default=None,
                        help="load a previously-saved LoRA adapter dir and continue training it "
                             "on top of --model/--resume-from (Objective 3, Stage 2)")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--out-name", default=None,
                        help="override the output dir name under results/ (default: derived from --resume-from)")
    args = parser.parse_args()
    main(args)
