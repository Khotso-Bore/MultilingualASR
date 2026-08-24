# Run Logs

**Run count so far: 17 total training runs** (6 classifier, 2 Wav2Vec2 pilots,
6 AfriHuBERT attempts, 3 Whisper runs - v1 done, an aborted v2 attempt,
a rescoped v2 done). Updated as each new run finishes; every run
(success, failure, or abort) gets one entry here.

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

## Whisper pilot - works cleanly, best ASR result so far

1. `whisper_pilot_v1_wer0265.log` - whisper-small, 5,000 NCHLT clips (<=10s),
   3 epochs. Steady improvement every epoch, no collapse: WER 0.428 -> 0.281
   -> 0.265, CER 0.110 -> 0.064 -> 0.060. Beats Wav2Vec2 pilot v2 (WER 0.332)
   despite fewer epochs and no ANV data. See `notes/pilot-ven-results.md`.
2. Whisper pilot v2, attempt 1 - **aborted, over-scoped, not a model
   failure**. Used `--train-clips 100000` (i.e. the entire 60,087-clip
   combined NCHLT+ANV train pool) intending "more data = better", but at
   16.4s/step x 11,505 total steps that's a ~51-hour run (tqdm's own ETA) -
   wildly out of proportion with Wav2Vec2's v1->v2 jump (6.5h for 7,325
   clips). Killed after ~1.4h (300/11,505 steps, no results to salvage) once
   the real step-rate revealed the scale problem.
3. `whisper_pilot_v2_wer0182.log` - Whisper pilot v2, attempt 2 (rescoped):
   same resume-from-v1 + ANV + 5-epoch recipe, capped at `--train-clips 12000`
   raw (7,325 kept after the 10s filter, matching Wav2Vec2 v2's scale).
   Steady improvement every epoch, no collapse: WER 0.296 -> 0.245 -> 0.206 ->
   0.184 -> **0.182**, CER 0.078 -> 0.062 -> 0.055 -> 0.050 -> **0.048**. Best
   ASR result across every pilot so far, beating Whisper pilot v1 (WER 0.265)
   despite the aborted attempt 1. ~12.3h on the M4. See
   `notes/pilot-ven-results.md`.

## AfriHuBERT pilot - failed, total CTC blank collapse

Attempted as the third distinct ASR model family (Wav2Vec2 = contrastive
self-supervision, Whisper = weakly-supervised encoder-decoder, HuBERT =
masked cluster-prediction self-supervision - confirmed to cover Tshivenda,
see the model-search writeup in conversation history / notes). Six
configurations attempted in total; the first four collapsed to the
identical degenerate solution:

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

Two more attempts after the root cause was confirmed:

5. `hubert_attempt5_blankbias_still_collapsed_to_a.log` - manually pushed the
   freshly-initialized CTC head's blank-token bias down at init
   (`--disfavor-blank-init`), the standard targeted fix for blank collapse
   specifically. Result: the collapse just moved - the model now predicts
   `'a'` (the single most frequent character in Tshivenda) for ~100% of
   frames instead of blank. Confirms the failure mode is "collapse to
   whichever single class is easiest," not something specific to the blank
   token - a deeper optimisation issue than a blank-bias nudge can fix.
6. `hubert_attempt6_25epochs_patience_still_collapsed.log` - definitive
   patience test: 25 epochs, early stopping disabled (`--patience 30`) so it
   could not be cut short. WER/CER stayed bit-for-bit pinned at 0.9668/0.962
   for all 25 epochs; loss fell steadily through epoch ~17 then visibly
   plateaued (2.998 at epoch 16 -> 2.957 at epoch 25, essentially flat) with
   zero corresponding change in predictions. Rules out "just needs more
   epochs" - this is a stable local minimum, not slow convergence.

**Conclusion**: total collapse into a single-dominant-class degenerate
solution, not resolved by six systematic attempts (default config, 2x lower
LR, unfrozen encoder, 3x longer warmup, blank-bias correction, and disabled
early stopping across 25 epochs) within the project timeline. Root cause
confirmed by direct inspection of decoded predictions at every stage, not
just inferred from WER. `src/pilot_finetune_hubert_mps.py` and
`notebooks/colab_hubert_ven.ipynb` are kept in the repo (with the diagnostic
flags added during this investigation: `--unfreeze-feature-encoder`,
`--warmup-ratio`, `--disfavor-blank-init`, `--blank-bias-penalty`,
`--patience`) in case debugging resumes later - e.g. on a CUDA GPU, in case
this is specific to the MPS/eager-attention fallback path this machine
requires. AfriHuBERT is not currently usable as the third model without more
work than the timeline allows. See `notes/pilot-ven-results.md` for the
write-up and the proposed message to Seani about picking a different third
model.
