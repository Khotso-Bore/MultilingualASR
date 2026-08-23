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

## AfriHuBERT pilot

`hubert_pilot_v1.log` (added once the current run completes) - see
`notes/pilot-ven-results.md` for the summary once available.
