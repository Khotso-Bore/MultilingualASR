# Conclusions - Tshivenda Track

Start here. This is the short version of everything in `notes/`, answering
the proposal's Main Research Question and Sub-questions 1-5 directly, with
links to the detailed evidence rather than repeating it. If you want the
real reference-vs-hypothesis examples, the checkpoint-by-checkpoint training
progressions, or the raw run logs, follow the links - they're not
duplicated here on purpose, so this stays readable in one sitting.

## Main Research Question

*How can data augmentation and fine-tuning techniques improve ASR
performance for Setswana, Sepedi, and Tshivenda, and how do the resulting
transcription errors propagate into downstream misinformation
classification inaccuracies?*

**For Tshivenda**: fine-tuning takes Whisper from an unusable zero-shot
WER of 1.108 down to 0.103 at full scale - a real, large improvement.
Targeted augmentation (SpecAugment + speed perturbation) adds a further,
measurable gain on top, more so for weaker models than strong ones. A
two-stage fine-tuning strategy, properly isolated from confounds, also
helps. And the resulting transcription quality is good enough to keep a
downstream misinformation classifier reliable, as long as ASR error rate
stays below roughly 25-30% WER - a threshold this project both establishes
and confirms Whisper's Tshivenda transcriptions comfortably (NCHLT) or
narrowly (ANV) sit inside.

This is a complete, evidenced answer for Tshivenda. It is not an answer for
Setswana or Sepedi - see "What's still open" below.

## Sub-question 1: baseline WER/CER for fine-tuned Wav2Vec2 and Whisper

**Answer**: Whisper is the clearly stronger architecture. Full-scale
results (60,087 combined NCHLT+ANV clips):

| Model | NCHLT WER | NCHLT CER | ANV WER | ANV CER |
|---|---|---|---|---|
| Whisper Large v3, zero-shot | 1.108 | 0.763 | 1.072 | 0.501 |
| Wav2Vec2 XLS-R-300M, full-scale | 0.252 | 0.056 | 0.457 | 0.106 |
| **Whisper, full-scale** | **0.103** | **0.032** | **0.256** | **0.108** |

Getting here required trying **9 different ASR architectures**, not just
the 2 named in the proposal - most (6 of 9) collapse into predicting a
single repeated character regardless of audio content, a real and
reproducible failure mode across independent checkpoints (AfriHuBERT, MMS,
w2v-BERT, data2vec-audio, XLSR-53, SSA-HuBERT), not one unlucky checkpoint.
Only Wav2Vec2 XLS-R-300M and UniSpeech train cleanly among the non-Whisper
options.

Evidence: `notes/pilot-ven-results.md` ("Whisper full-scale" and
"Wav2Vec2 XLS-R-300M full-scale" sections, plus every "model attempt"
section for the collapse investigation) - real reference-vs-hypothesis
examples for every model, working or collapsed. Raw logs: `results/logs/README.md`.

## Sub-question 2: does augmentation improve transcription accuracy?

**Answer: yes, consistently, but the size of the gain shrinks as the
model/data gets stronger, and it does not substitute for real data.**

| Setup | NCHLT WER | Relative improvement |
|---|---|---|
| Wav2Vec2, no augmentation (5k clips) | 0.332 (pilot) / 0.614 (5k fresh) | - |
| Wav2Vec2, + augmentation (5k->15k) | 0.269 | **56%** vs. the matched 5k-clip baseline |
| Whisper, no augmentation (5k clips) | 0.265 | - |
| Whisper, + augmentation (5k->15k) | 0.217 | 18% |
| Whisper, + augmentation at matched volume (20k->60k) | 0.187 | improves further, still behind... |
| Whisper, full-scale real data (60k, no augmentation) | 0.103 | ...genuinely more real data |

Evidence: `notes/pilot-ven-results.md` ("Objective 2" section and its
"Follow-up run" subsection) - includes the real memory bug (silent OOM
kill) hit and fixed while running the matched-volume test, and
checkpoint-by-checkpoint progressions showing *when* augmented models
actually improve during training.

## Sub-question 3: two-stage vs. single-stage fine-tuning

**Answer: two-stage genuinely helps, once measured fairly.**

| Setup | NCHLT WER |
|---|---|
| Single-stage, full fine-tune (the strongest baseline) | 0.265 |
| Single-stage, LoRA (same capacity as the two-stage run below) | 0.384 |
| **Two-stage, LoRA** (combined-domain Stage 1 -> NCHLT Stage 2) | **0.321** |

The first comparison (two-stage LoRA vs. single-stage full fine-tune) made
two-stage look like a loss - but that mixed two variables (staging *and*
how many parameters get updated) at once. Running the missing control
(single-stage with the identical LoRA setup) shows two-stage beats
single-stage by 16% relative WER once capacity is held constant - agreeing
with the literature (Teryan et al., 2026) rather than contradicting it.
LoRA itself still trails full fine-tuning regardless of staging - both
facts are true and don't conflict.

Two-stage LoRA also shows real resistance to catastrophic forgetting:
Stage 2 trains on NCHLT only, yet ANV performance stays well above
zero-shot and above the single-stage-LoRA control that never saw ANV at all.

Evidence: `notes/pilot-ven-results.md` ("Objective 3" section, documented
as Run 1 then Run 2 rather than a single rewritten conclusion, since Run
1's own reasoning is what motivated Run 2's control).

## Sub-question 4: AfroXLM-RoBERTa classification performance vs. text-only baselines

**Answer**: AfroXLM-RoBERTa clearly beats the comparison baseline -
0.562 accuracy / 0.550 macro F1 (5-fold grouped CV) vs. XLM-RoBERTa's
exactly-chance 0.500/0.333, confirming the literature's claim (MphayaNER)
that Africa-specific pretraining matters specifically for Tshivenda.

**Important caveat, not fixable**: the real Mukwevho et al. (2024)
misinformation dataset this was supposed to use is permanently
unrecoverable. This result is on a synthetic proxy built from real
Vukuzenzele government news, with synthetic "fake" counterparts generated
by content-level distortion (not ASR-style noise - that's a separate
study, see Sub-question 5) - the proposal's own text-topic-classification
fallback pattern (already used for Setswana/Sepedi), applied here as the
closest available substitute. No comparison against Mukwevho's original
GRU baseline is possible.

Evidence: `notes/tshivenda-classifier-proxy.md`.

## Sub-question 5: how does ASR error rate affect downstream classification, and which error types matter most?

**Answer**: not all errors are equally harmful. **Insertion errors hurt
classification the most** (clean F1 0.546 -> 0.389 by WER 50%, a 29%
relative drop); **deletion errors barely matter** (F1 stays at
0.516-0.550 across the entire range, essentially flat); substitution sits
in between. Under the realistic mixed-error condition (using the real
measured Whisper error model, which is itself insertion-heavy at 29% of
errors), F1 crosses below ~0.50 around **WER 25-30%** - that's the
practical reliability threshold this project establishes.

Whisper's actual full-scale Tshivenda ASR sits comfortably inside that
safe zone on NCHLT (WER 0.103) but close to the edge on the harder ANV
spontaneous-speech domain (WER 0.256) - a genuine, specific deployment
caveat, not just an abstract number.

Evidence: `notes/tshivenda-error-propagation.md` - includes real corrupted-
transcript examples at each WER level (e.g. a government minister's
surname being overwritten by filler text by WER 0.5), not just the
aggregate F1 curve.

## What's still open

- **Objective 7** (cross-language feasibility/transferability) needs
  Setswana and Sepedi results from the rest of the team - this track only
  answers it for Tshivenda, and can't complete it alone.
- **XEUS** (ESPnet) - the one architecture candidate with *confirmed*
  native Tshivenda coverage - was identified but never attempted, due to
  integration cost (non-mainline fork, CUDA required, no path to the
  cheap local-pilot-first check used everywhere else). Logged as a
  deliberate no, not a failure.
- **A same-capacity LoRA rank sweep** (16/32, not just 8) - the two-stage
  LoRA result showed a word-boundary-merge artifact that looks like a
  rank-8 capacity limit; untested whether a higher rank closes more of the
  gap to full fine-tuning.
- **A true full-scale augmented run** (180k effective examples, not the
  20k->60k matched-volume compromise) - would take an estimated 30+ hours;
  the matched-volume result is a scaled-down but real proxy for this
  question, not a substitute for the definitive number.
- **Statistical confidence on the classifier** - Objective 4's numbers are
  one 5-fold run on 179 source articles; more seeds would turn the single
  point estimate into a proper mean +/- std, the same way the report
  already does for other results.
- The repo-restructure pass (`src/asr/` grouping 10 files under one flat
  directory) is planned but not done - purely cosmetic, doesn't affect any
  result.

## How to reproduce every headline number

All commands assume the repo root and an activated `.venv`
(`pip install -r requirements.txt`). `PYTORCH_ENABLE_MPS_FALLBACK=1` is
needed for the CTC-loss models (Wav2Vec2 family) on Apple Silicon.

| Result | Command |
|---|---|
| Whisper full-scale (WER 0.103) | `python src/asr/pilot_finetune_whisper_mps_ven.py --resume-from results/whisper-ven-pilot-v2/final --include-anv --epochs 1 --learning-rate 5e-5 --train-clips 100000 --eval-clips 500` |
| Wav2Vec2 full-scale (WER 0.252) | `PYTORCH_ENABLE_MPS_FALLBACK=1 python src/asr/pilot_finetune_wav2vec2_mps_ven.py --resume-from results/wav2vec2-ven-pilot-v2/final --include-anv --epochs 1 --learning-rate 5e-5 --train-clips 100000 --eval-clips 500` |
| Wav2Vec2 + augmentation (WER 0.269) | `PYTORCH_ENABLE_MPS_FALLBACK=1 python src/asr/pilot_finetune_wav2vec2_mps_ven.py --train-clips 5000 --eval-clips 500 --epochs 3 --augment --spec-augment` |
| Whisper + augmentation (WER 0.217) | `python src/asr/pilot_finetune_whisper_mps_ven.py --train-clips 5000 --eval-clips 500 --epochs 3 --augment --spec-augment` |
| Whisper + augmentation, matched volume (WER 0.187) | `python src/asr/pilot_finetune_whisper_mps_ven.py --train-clips 20000 --eval-clips 500 --epochs 1 --include-anv --augment --spec-augment` |
| Two-stage LoRA Stage 1 (WER 0.517) | `python src/asr/pilot_finetune_whisper_mps_ven.py --train-clips 5000 --eval-clips 500 --epochs 3 --include-anv --lora --learning-rate 3e-4` |
| Two-stage LoRA Stage 2 (WER 0.360 -> 0.321 standardized) | `python src/asr/pilot_finetune_whisper_mps_ven.py --train-clips 5000 --eval-clips 500 --epochs 3 --lora-adapter-from <stage1_dir>/final --learning-rate 3e-4` |
| Single-stage LoRA control (WER 0.384) | `python src/asr/pilot_finetune_whisper_mps_ven.py --train-clips 5000 --eval-clips 500 --epochs 3 --lora --learning-rate 3e-4` |
| Standardized eval on any Whisper/full checkpoint | `python src/asr/zero_shot_baseline_ven.py --model <checkpoint_dir> --limit 200 --seed 42 --save-predictions <out_dir>` |
| Standardized eval on any Wav2Vec2/CTC checkpoint | `python src/asr/evaluate_wav2vec2_ven.py --checkpoint <checkpoint_dir> --limit 200 --seed 42 --save-predictions <out_dir>` |
| Classifier, AfroXLM-RoBERTa (0.562 acc / 0.550 F1) | `python src/classification/train_classifier_ven.py --model Davlan/afro-xlmr-base --folds 5 --epochs 15 --learning-rate 1e-3 --freeze-base` |
| Error-propagation degradation study | `python src/error_propagation/run_degradation_study_ven.py --error-model results/preds_full/final_nchlt_test.csv results/preds_full/final_anv_dev_test.csv` |

LoRA adapters must be merged before standardized eval - see the "merge and
verify generation config" snippet in `notes/pilot-ven-results.md`
("Objective 3" section) for the exact steps and the bug that first version
of this step hit.
