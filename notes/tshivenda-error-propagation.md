# Tshivenda Error-Propagation Study (proposal §4.6, Objectives 5-6)

Answers Sub-question 5: how does ASR transcription error rate (WER) affect
the misinformation classifier's F1/accuracy, and which error type is most
harmful.

## Method

`src/error_propagation/run_degradation_study_ven.py`, reusing the exact
grouped 5-fold CV split from `src/classification/train_classifier_ven.py`
(`Davlan/afro-xlmr-base`, `--freeze-base --learning-rate 1e-3 --epochs 15`,
same proven config as the Objective 4 result). For each fold: train once on
clean text, then evaluate that same trained model on the held-out fold's
text corrupted at target WER 10/20/30/40/50%, for "mixed" mode plus each
ablation (substitution-only, deletion-only, insertion-only). Corruption uses
the real error model measured from the full-scale Whisper predictions
(`results/preds_full/final_nchlt_test.csv` + `final_anv_dev_test.csv`, S:D:I
= 54.4 : 16.5 : 29.0 - see `notes/pilot-ven-results.md`), not the corruption
engine's generic default, so "mixed" mode reflects this project's actual
measured ASR error shape.

Smoke-tested first (`--smoke-test`, 2 folds / 1 epoch) to catch bugs before
committing to the full run - that pass flatlined at exactly WER-independent
0.333 macro F1 for fold 2, the same "predicts one class regardless of input"
signature already seen for XLM-RoBERTa's collapse in the Objective 4 study;
expected from 1 epoch of a frozen linear probe, not a bug in the
corruption/eval wiring (confirmed once the full run below produced real,
varying, WER-sensitive numbers).

## What the corrupted text actually looks like

Not just numbers - real examples from `dataset/vukuzenzele/misinfo_proxy_ven.csv`,
corrupted with the real full-scale Whisper error model, "mixed" mode, seed 42:

**Example 1** (original): `U ya nga Minisṱa vha Muhasho wa Mutakalo, Dokotela
Vho Joe Phaahla, nga ḽa 25 Lambamai Afrika Tshipembe ḽo rekhoda u engedzea
nga 137% ha tshivhalo tshiswa tsha zwiwo zwa u kavhiwa nga vairasi ya COVI...`

- WER 0.1: `...Dokotela Vho Joe Phaahla, ḽa 25 Lambamai Afrika Tshipembe ḽo
  rekhoda u 60 engedzea nga 137%...` - a dropped word ("nga"), one inserted
  number ("60"), otherwise intact.
- WER 0.3: `...Dokutela Vho wavho, Joe Phaahla, ngauri ḽa Lambamii Afrika
  guda Tshipembe lo rekhoda u engedzua dzi 137%...` - "Dokotela"->"Dokutela",
  an inserted "wavho", "engedzea"->"engedzua dzi" - still readable, meaning
  drifting.
- WER 0.5: `...Dokutela Vho wavho, Joe Iṅwe ngauri vhuḓifari nḓivhadzo
  Lambomai Afrika Tshipembu guda lo u engodzea ngauri...` - "Phaahla" (the
  minister's actual surname) is gone, replaced by unrelated inserted words
  ("Iṅwe", "vhuḓifari", "nḓivhadzo") - a named entity central to the
  article's claim has been corrupted away.

**Example 3** (original): `dzo ya khothe, vha Maanḓalanga a Vhutshu-tshisi
ha Lushaka (NPA) vha na tshumelo yo khe-theaho ya u thusa vhone...`

- WER 0.5: `...vha Maanḓalanga avha Vhutshu-tshisi ha Lushika (NPA) vha na
  Iṅwe chumelo khayo ke-theaho yo u wone...` - the organisation acronym
  "(NPA)" survives (numbers/acronyms are comparatively robust to this word-
  level corruption model), but surrounding context words are increasingly
  replaced by high-frequency corpus filler ("Iṅwe", "khayo") as WER climbs.

The pattern across all three source examples: at WER 0.1 the text stays
essentially readable with minor slips; by WER 0.3 word choice starts
drifting in ways that could plausibly flip meaning; by WER 0.5 specific
named entities (a minister's surname, in Example 1) get overwritten by
generic corpus-frequent words - exactly the failure mode that would make a
misinformation classifier's job harder, since names/entities are often what
carries the claim.

## Results (5-fold grouped CV, mean macro F1)

Clean-text baseline per fold: 0.578, 0.524, 0.559, 0.514, 0.555 (mean
**0.546**) - matches Objective 4's originally reported 0.550 +/- 0.030
almost exactly, confirming this reuses the same trained-model behavior, not
a different/weaker setup.

| Mode | wer=0.0 | wer=0.1 | wer=0.2 | wer=0.3 | wer=0.4 | wer=0.5 |
|---|---|---|---|---|---|---|
| **mixed** (real measured error shape) | 0.546 | 0.528 | 0.535 | 0.500 | 0.451 | 0.451 |
| substitution only | 0.546 | 0.545 | 0.515 | 0.529 | 0.515 | 0.487 |
| deletion only | 0.546 | 0.543 | 0.537 | 0.529 | 0.550 | 0.516 |
| insertion only | 0.546 | 0.542 | 0.483 | 0.454 | 0.443 | 0.389 |

## Findings

- **Insertion errors are the most harmful error type**, by a clear margin -
  F1 falls from 0.546 to 0.389 by WER 50% (a 29% relative drop), and the
  decline is the steepest and most consistent of the three ablations.
- **Deletion errors are, surprisingly, the least harmful** - F1 stays within
  0.516-0.550 across the entire WER range, essentially flat. Dropping words
  appears to cost this classifier far less signal than inserting wrong ones.
- **Substitution sits in between**, a moderate, fairly steady decline to 0.487.
- **Mixed mode (the realistic condition) tracks closer to the harmful end**
  than a simple average of the three ablations would suggest (0.451 at
  WER 50% vs. e.g. deletion's 0.516) - consistent with the real measured
  error model being insertion-heavy (29% of errors, far above the
  corruption engine's generic 15% default), so the realistic degradation is
  worse than the deletion-dominated pilot-v2-era error model would have
  predicted.
- **Practical reliability threshold (Objective 6)**: under the mixed/
  realistic condition, F1 crosses below ~0.50 (from a clean 0.546) around
  WER 25-30%. Whisper's full-scale ASR is well inside the safe zone on
  NCHLT (WER 0.103) but sits close to this threshold on ANV's harder
  spontaneous-speech domain (WER 0.256) - worth flagging as a real
  deployment caveat, not just a headline WER number.

## Caveats

- Small dataset (179 source articles / 358 rows, same proxy limitation as
  Objective 4 - see `notes/tshivenda-classifier-proxy.md`), so per-fold
  variance is real; read the table as a shape/trend, not a precise curve.
- Error-model corruption is applied at the word level to already-clean
  Vukuzenzele text, not real ASR output on this specific classifier's text -
  same synthetic-corruption limitation the proposal itself names in scope
  (§2.3).
- Only one classifier (AfroXLM-RoBERTa) has been run through this study so
  far; XLM-RoBERTa collapsed to chance even on clean text (Objective 4), so
  running it through the degradation study too would not add a meaningful
  comparison - per Seani's model-selection guidance, not worth sinking
  further time into an already-ruled-out underperformer.

## How to regenerate

```bash
python src/error_propagation/run_degradation_study_ven.py --smoke-test   # sanity check, ~2 min
python src/error_propagation/run_degradation_study_ven.py \
    --error-model results/preds_full/final_nchlt_test.csv results/preds_full/final_anv_dev_test.csv
```

Log used for this writeup: `/tmp/degradation_study_afroxlmr.log` (not
committed - rerun the command above to regenerate).
