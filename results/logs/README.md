# Run Logs

Raw (progress-bar-stripped) console output from every training run, kept as
evidence alongside the summarised numbers in `notes/`. Chronological order
below matches how the classifier's methodology actually evolved - the
"failed" attempts are kept deliberately, they are the debugging story
referenced in `notes/tshivenda-classifier-proxy.md`.

## Misinformation classifier (Tshivenda proxy dataset)

1. `classifier_attempt1_unstable_afroxlmr.log` - first full fine-tune,
   lr 2e-5: loss stuck at ln(2) for 3 epochs, no learning at all. Killed
   after fold 1.
2. `classifier_attempt2_bestcheckpoint_still_unstable.log` - lr 5e-5 +
   `load_best_model_at_end`: found real signal (epoch 2, ~59% accuracy) but
   still collapsed on some folds. 2-fold CV: 0.589 +/- 0.089 accuracy.
3. `classifier_attempt3_frozen_crashed_mps_bug.log` - first attempt at
   freezing the base encoder (linear probe): crashed immediately, MPS's
   `scaled_dot_product_attention` does not support dropout when the base is
   frozen.
4. `classifier_attempt4_frozen_eager_2fold_validation.log` - same config,
   fixed with `attn_implementation="eager"`: stable, 2-fold CV
   0.542 +/- 0.042 accuracy (std dropped ~2x vs. attempt 2).
5. `classifier_final_5fold_afroxlmr.log` / `classifier_final_5fold_xlmr.log`
   - the reported result: full 5-fold grouped CV with the validated frozen
   config (`--freeze-base --learning-rate 1e-3 --epochs 15`). AfroXLM-RoBERTa
   0.562 +/- 0.030 accuracy; XLM-RoBERTa exactly 0.500 +/- 0.000 (chance on
   every fold).

## ASR (Wav2Vec2 zero-shot baseline, pilot v1/v2)

Not preserved as raw logs - those runs happened earlier in a long session
and their task output files rotated out before this results/logs/ directory
was created. Only the summarised numbers survive, in
`notes/pilot-ven-results.md`. Every run from this point forward is redirected
to a persistent log file and copied here, specifically to avoid losing this
again.

## AfriHuBERT pilot - failed, total CTC blank collapse

Attempted as the third distinct ASR model family (Wav2Vec2 = contrastive
self-supervision, Whisper = weakly-supervised encoder-decoder, HuBERT =
masked cluster-prediction self-supervision - confirmed to cover Tshivenda,
see the model-search writeup in conversation history / notes). Four
configurations, all collapsed to the identical degenerate solution:

1. `hubert_attempt1_lr1e4_frozen_collapsed.log` - default config (same
   hyperparameters that worked for Wav2Vec2), frozen feature encoder,
   lr 1e-4, 5000 clips / 3 epochs. WER/CER frozen at 0.9709/0.9614 across
   all 3 epochs despite loss slowly decreasing.
2. `hubert_attempt2_lr3e5_frozen_collapsed.log` - lr lowered 3x to 3e-5
   (hypothesis: AfriHuBERT's post-norm "base" architecture, unlike Wav2Vec2
   XLS-R's pre-norm "stable" architecture, needed a lower LR). Same
   collapse, frozen at 0.9668/0.962.
3. `hubert_attempt3_lr3e5_unfrozen_collapsed.log` - feature encoder
   unfrozen (hypothesis: frozen group-norm features not separable enough).
   Identical frozen result, 0.9668/0.962 on the same eval subset.
4. `hubert_attempt4_lr5e5_warmup03_collapsed.log` - warmup_ratio raised
   0.1 -> 0.3 (hypothesis: blank collapse from too-aggressive early
   steps - the standard mitigation). Still identical, 0.9668/0.962.

**Root cause, confirmed by direct inspection** (not just inferred from WER):
loaded the trained checkpoint from attempt 1 and ran raw predictions on 5
training clips - the model predicts the `[PAD]` (blank) token for 100% of
frames on every single example, decoding to an empty string every time. WER
against an empty hypothesis is a near-fixed function of the reference length
alone, which is exactly why every attempt above scored identically on a
given eval subset regardless of what the model actually learned elsewhere.
Ruled out as the cause: CTC length-constraint violations (checked directly,
0/100 examples affected).

**Conclusion**: total blank collapse, not resolved by four standard
mitigations (lower LR, unfreezing, longer warmup) within the project
timeline. `src/pilot_finetune_hubert_mps.py` and
`notebooks/colab_hubert_ven.ipynb` are kept in the repo (with
`--warmup-ratio` and `--unfreeze-feature-encoder` flags added during this
investigation) in case further debugging resumes later, but AfriHuBERT is
not currently usable as the third model without more work than the timeline
allows. See `notes/pilot-ven-results.md` for the write-up and the proposed
message to Seani about picking a different third model.
