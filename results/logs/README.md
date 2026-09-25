# Run Logs

**Run count so far: 35 total training runs** (6 classifier + 1 error-propagation
degradation study, 2 Wav2Vec2 pilots + 1 Wav2Vec2 full-scale + 1 Wav2Vec2
augmentation pilot, 6 AfriHuBERT attempts, 3 pilot-scale Whisper runs (v1
done, an aborted v2 attempt, a rescoped v2 done) + 1 Whisper full-scale +
1 Whisper augmentation pilot + 1 Whisper matched-volume augmentation
follow-up, 2 MMS attempts, 1 w2v-BERT attempt, 1 data2vec-audio attempt,
1 UniSpeech attempt, 2 XLSR-53 attempts, 1 SSA-HuBERT attempt, 3 two-stage
LoRA runs (Stage 1 bad-LR + Stage 1 fixed-LR + Stage 2) + 1 same-capacity
LoRA control). Updated as each new run finishes; every run (success,
failure, or abort) gets one entry here.

**A note on where later logs live**: runs from Objective 1's full-scale
push onward (2026-09-21 onward) were originally redirected to `/tmp/*.log`
during the session rather than straight to this directory, since they were
long-running background jobs checked on incrementally. Copied into this
directory afterward once each run finished, so the evidence trail stays
complete here rather than depending on `/tmp` (which doesn't persist).
Filenames below are the copied names, not the original `/tmp` ones.

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
just inferred from WER. `src/asr/pilot_finetune_hubert_mps_ven.py` and
`notebooks/asr/colab_hubert_ven.ipynb` are kept in the repo (with the diagnostic
flags added during this investigation: `--unfreeze-feature-encoder`,
`--warmup-ratio`, `--disfavor-blank-init`, `--blank-bias-penalty`,
`--patience`) in case debugging resumes later - e.g. on a CUDA GPU, in case
this is specific to the MPS/eager-attention fallback path this machine
requires. AfriHuBERT is not currently usable as the third model without more
work than the timeline allows. See `notes/pilot-ven-results.md` for the
write-up and the proposed message to Seani about picking a different third
model.

## MMS pilot - failed, same total CTC blank collapse as AfriHuBERT

Fourth model attempt, per Seani's guidance to keep trying model families
even without confirmed Tshivenda coverage. `facebook/mms-300m` - the MMS
self-supervised base checkpoint, architecturally identical to Wav2Vec2
XLS-R-300M (`Wav2Vec2ForCTC`, contrastive pretraining, not AfriHuBERT's
masked-cluster objective).

1. `mms_attempt1_collapsed.log` - 5,000 raw NCHLT clips (4,974 kept), 2
   epochs (fewer epochs per Seani's guidance). `eval_wer`/`eval_cer` =
   0.971/0.961 after epoch 1, 0.998/0.877 after epoch 2 - numbers close
   enough to AfriHuBERT's own frozen 0.9709/0.9614 to be suspicious.
   Confirmed by direct inspection of decoded predictions (not just the WER
   score): loaded the saved checkpoint and ran raw predictions on 5 training
   clips - 100% of frames predict the pad/blank token on every example,
   decoding to an empty string every time. Identical failure mode to
   AfriHuBERT's first four attempts.

Notable: this rules out "architecture family" as the predictor of collapse.
MMS and XLS-R share the same architecture and pretraining objective, yet one
collapses on Tshivenda and the other doesn't - something about the specific
pretraining data/scale/init differs.

**Decision: MMS ruled out as a second confirmed collapse, not re-running
AfriHuBERT's full mitigation sweep against it too** - the same fixes (lower
LR, blank-bias disfavor) already failed to save AfriHuBERT, no reason to
expect a different outcome here. ESPnet's XEUS (the one candidate with
confirmed native Tshivenda coverage) was checked and not pursued - requires
a non-mainline ESPnet fork, a work-in-progress community fine-tuning repo,
and states CUDA as a prerequisite with no confirmed local-MPS-pilot path.

## w2v-BERT 2.0 pilot - failed, collapsed to a different single token

Fifth model attempt. `facebook/w2v-bert-2.0` - Conformer-based, hybrid
contrastive + masked-prediction objective, 4.5M hours/143+ languages -
architecturally the most different checkpoint tried so far.

1. `w2vbert_attempt1_collapsed.log` - 5,000 raw NCHLT clips (4,974 kept), 2
   epochs. `eval_wer`/`eval_cer` = 0.9535/0.9329, bit-for-bit identical
   between epoch 1 and epoch 2. Confirmed by direct inspection: ~99% of
   frames predict blank, but the one non-blank frame decodes to `'n'`, so
   every sample outputs just `'n'` - same "collapse to whichever single
   class is easiest" pattern as AfriHuBERT's attempt 5 (`'a'`), just a
   different token.

## data2vec-audio pilot - failed, disproves the discretization hypothesis

Sixth model attempt. `facebook/data2vec-audio-large` - regresses onto
continuous teacher representations, no discretized pretraining target at
all (unlike AfriHuBERT/MMS/w2v-BERT, which all discretize in some way).
English-only pretraining (Librispeech) - weakest multilingual transfer
prior tried so far.

1. `data2vec_attempt1_collapsed.log` - 5,000 raw NCHLT clips (4,974 kept),
   2 epochs. `eval_wer`/`eval_cer` = 0.9709/0.9614, bit-for-bit identical
   between epochs and identical to AfriHuBERT's own original blank-collapse
   numbers. Confirmed by direct inspection: 100% blank on every frame of
   every checked clip, decoding to an empty string every time.

**This rules out discretization as the explanation** - data2vec-audio has
none, and collapsed exactly like the three that do.

**Status: 4 of 5 non-Whisper CTC fine-tunes have now collapsed**
(AfriHuBERT, MMS, w2v-BERT, data2vec-audio) across four different
architectures and four different pretraining objectives - only Wav2Vec2
XLS-R-300M hasn't.

## MMS re-test at 3x lower learning rate - still collapses

Testing whether the shared training recipe (not model choice) explains the
pattern above.

1. `mms_attempt2_lowlr_still_collapsed.log` - same MMS pilot, `--learning-rate
   3e-5` instead of the default `1e-4` (3x lower, same reduction factor
   tried against AfriHuBERT). Identical result: `eval_wer`/`eval_cer` =
   0.9709/0.9614, frozen across both epochs, 100% blank confirmed by direct
   inspection.

**The recipe theory is disproven too.** Two explanations tried and ruled
out in turn - not discretization (data2vec-audio has none, collapsed
anyway), not the learning rate (3x lower didn't save MMS). XLS-R-300M
remains the only non-Whisper CTC checkpoint that trains cleanly, out of 5
tried, with no confirmed explanation yet for why.

## UniSpeech pilot - works, no collapse

Seventh model attempt. `microsoft/unispeech-large-1500h-cv` - multi-task
phonetic-CTC + contrastive pretraining on CommonVoice, specifically
validated in its own paper for cross-lingual transfer to unseen languages.

1. `unispeech_attempt1_works_wer0610.log` - 5,000 raw NCHLT clips (4,974
   kept), 2 epochs. `eval_wer`/`eval_cer` fell every epoch: 0.765/0.179 ->
   **0.610/0.144**, comparable in shape to XLS-R's own first pilot
   (0.614/0.151 at the same scale). Confirmed by direct inspection: decoded
   predictions on 5 training clips are genuinely close to references - one
   exact match, the rest off by a word-boundary or single character.

**UniSpeech is the second working non-Whisper model, after XLS-R.** Out of
6 non-Whisper checkpoints tried: 2 work (XLS-R, UniSpeech), 4 collapse
(AfriHuBERT, MMS, w2v-BERT, data2vec-audio). Both working checkpoints share
a multi-task/discriminative element beyond pure self-supervision. See
`notes/pilot-ven-results.md`.

## Objective 1: Whisper and Wav2Vec2 full-scale (2026-09-21/24)

Both models named in Objective 1, taken from pilot scale to the full
60,087-clip combined NCHLT+ANV pool - the actual headline numbers, not
another pilot.

1. `whisper_fullscale_wer0103.log` - resumed from pilot v2 (0.182), 1
   epoch, full pool, ~11h21m training. No errors. Training-time eval
   WER 0.111/CER 0.032. Standardized 200-clip eval (not in this log,
   generated separately via `zero_shot_baseline_ven.py`): **WER 0.103,
   CER 0.032** on NCHLT.
2. `wav2vec2_fullscale_train.log` - same methodology, resumed from
   Wav2Vec2 pilot v2 (0.332). Training-time eval WER 0.425/CER 0.099
   (mixed NCHLT+ANV eval set).
3. `wav2vec2_fullscale_standardized_eval_wer0252.log` - standardized
   200-clip eval of the checkpoint from run 2: **WER 0.252, CER 0.056**
   on NCHLT, 0.457/0.106 on ANV.

Both full-scale runs completed with no errors. See `notes/pilot-ven-results.md`
("Whisper full-scale" and "Wav2Vec2 XLS-R-300M full-scale" sections) for
the real reference-vs-hypothesis examples and checkpoint-by-checkpoint
training-progression tables built from these runs.

## Objective 2: data augmentation - SpecAugment + speed perturbation (2026-09-23/25)

`src/asr/audio_augment_ven.py` (speed perturbation) plus HuggingFace's
built-in SpecAugment config flags, added to both pilot scripts.

1. `wav2vec2_augment_pilot_wer0269.log` - 5,000 NCHLT clips (tripled to
   15,000 by augmentation), 3 epochs, fresh XLS-R-300M. Standardized eval:
   **WER 0.269, CER 0.060** - down from the un-augmented pilot's 0.332/0.074,
   a 56% relative WER reduction from this one change alone.
2. `whisper_augment_pilot_wer0217.log` / `whisper_augment_pilot_standardized_eval.log`
   - same recipe on Whisper: **WER 0.217, CER 0.052** - down from 0.265/0.060,
   an 18% relative reduction (smaller than Wav2Vec2's, since Whisper had
   less headroom to begin with).
3. `whisper_augment_matchedvolume_wer0187.log` / `whisper_augment_matchedvolume_standardized_eval.log`
   - follow-up at larger scale: 20,000 real clips augmented to 60,000
   effective examples (matching the full-scale run's total training
   volume). Standardized eval: **WER 0.187, CER 0.047** - better than the
   5k-clip augmented pilot, but still clearly behind full-scale's real-data
   result (0.103). Conclusion: augmentation helps at every scale tried, but
   doesn't substitute for genuinely more real data.

**A real bug caught along the way**: the first two attempts at run 3 died
silently within seconds (no traceback, just a multiprocessing semaphore
warning) - `augment_oom_diagnostic_isolation_test.log` is the isolated test
that helped confirm this was a memory issue (materializing all 60,000
decoded audio arrays in one Python list before building the dataset), not
the `nohup`/`caffeinate` launch wrapper it looked like at first. Fixed in
both pilot scripts by switching to `Dataset.from_generator()`.

See `notes/pilot-ven-results.md` ("Objective 2" section and its "Follow-up
run" subsection) for full interpretation and real examples.

## Objective 3: two-stage fine-tuning + LoRA (2026-09-22/24)

LoRA support added to `pilot_finetune_whisper_mps_ven.py` via `peft`
(`--lora` for a fresh wrap, `--lora-adapter-from` to continue a saved
adapter - i.e. Stage 2 continuing Stage 1's adapter without touching the
frozen base weights).

1. `twostage_stage1_attempt1_badlr.log` - Stage 1 with the script's
   full-fine-tune-tuned default learning rate (1e-5) - too low for LoRA to
   move meaningfully in 3 epochs. Finished at WER 0.947, barely better than
   zero-shot. Diagnosed (LoRA generally needs a much higher rate than full
   fine-tuning) and relaunched rather than accepted.
2. `twostage_stage1_attempt2_fixedlr_wer0517.log` - relaunched with
   `--learning-rate 3e-4`. WER 0.628 -> 0.544 -> **0.517** across 3 epochs -
   epoch 1 alone already beat attempt 1's entire 3-epoch result.
3. `twostage_stage2_wer0360.log` - continued Stage 1's adapter on
   NCHLT-only clips, 3 more epochs: 0.450 -> 0.377 -> **0.360**
   (training-time eval).
4. `twostage_standardized_eval_attempt1_buggygenconfig.log` - first merge
   attempt lost the model's `language="sw"`/`task="transcribe"` generation
   config during the adapter merge (a fresh base was reloaded for merging
   without re-applying it). Came back at an impossible WER 1.989, caught
   from the numbers not matching the qualitative evidence rather than
   trusted.
5. `twostage_standardized_eval_attempt2_fixed_wer0321.log` - fixed merge
   (generation config set before merging, verified present in the saved
   `generation_config.json`), re-run: **WER 0.321, CER 0.080** on NCHLT,
   0.797/0.215 on ANV.
6. `singlestage_lora_control_wer0404.log` / `singlestage_lora_control_standardized_eval_wer0384.log`
   - the missing control: single-stage training with the *same* LoRA
   config (rank 8, alpha 16, lr 3e-4) on the same 5k NCHLT clips, no Stage
   1. Standardized eval: **WER 0.384, CER 0.095** - clearly worse than
   two-stage's 0.321, confirming staging genuinely helps once LoRA capacity
   is held constant (the original two-stage-vs-full-fine-tune comparison
   had mixed staging and capacity together).

See `notes/pilot-ven-results.md` ("Objective 3" section - documented as
Run 1 / Run 2, not rewritten, since Run 1's confound-aware reasoning is
what motivated Run 2's control) for the full interpretation, real examples,
and checkpoint-by-checkpoint training-progression table.

## Eighth model attempt: XLSR-53 (2026-09-24) - collapsed

`facebook/wav2vec2-large-xlsr-53` - a candidate from the project's own
literature review (Dar and Pushparaj, 2026, found it beat XLS-R-300M for
low-resource Kashmiri).

1. `xlsr53_attempt1_slow_step_falsealarm.log` - initial smoke test logged
   one training step taking 15 minutes, which looked like a severe
   architecture-specific slowdown. Diagnosed with a `--grad-accum 1` timing
   probe rather than assumed: real per-step time was ~1-1.5s, normal for
   this model size - the 15-minute figure was system memory pressure right
   after a previous large run finished, not XLSR-53 itself.
2. `xlsr53_attempt2_collapsed.log` - real pilot (5,000 NCHLT clips, 3
   epochs, matching XLS-R-300M's own original validating pilot scale).
   WER/CER frozen at exactly 0.9709/0.9614 across all 3 epochs. Confirmed
   by direct inspection: every hypothesis is an empty string, the same
   total-collapse signature as AfriHuBERT/MMS/w2v-BERT/data2vec-audio.

## Ninth model attempt: SSA-HuBERT (2026-09-24) - collapsed

`Orange/SSA-HuBERT-base-5k` - a genuinely different Africa-centric HuBERT
pretrain from AfriHuBERT (not a variant of it), found via a literature
search for other candidates.

1. `ssahubert_attempt1_collapsed_to_a.log` - `--disfavor-blank-init`
   enabled from the very first run (already known necessary from the
   AfriHuBERT investigation). WER/CER frozen at 0.9705/0.9612 across all 3
   epochs. Confirmed by direct inspection: every hypothesis is the single
   character `'a'`, the exact same fallback failure mode AfriHuBERT only
   reached *after* its blank-bias fix - reached immediately here instead.

**Updated tally: 9 checkpoints tried, 2 work (XLS-R-300M, UniSpeech), 6
collapse** (AfriHuBERT, MMS, w2v-BERT, data2vec-audio, XLSR-53,
SSA-HuBERT). See `notes/pilot-ven-results.md` for why this closes the
non-Whisper architecture search rather than motivating a tenth attempt.

## Error propagation degradation study (Objectives 5/6, 2026-09-22)

`error_propagation_degradation_study_afroxlmr.log` - 5-fold grouped
CV, AfroXLM-RoBERTa, evaluated on text corrupted at WER 10/20/30/40/50%
(mixed + substitution/deletion/insertion ablations) using the real
full-scale Whisper error model. Clean baseline F1 0.546 (matches Objective
4's originally reported 0.550 +/- 0.030). Insertion errors most harmful
(F1 down to 0.389 by WER 50%), deletion least harmful (stays at
0.516-0.550). See `notes/tshivenda-error-propagation.md` for the full
write-up, real corrupted-text examples, and the practical reliability
threshold this implies.
