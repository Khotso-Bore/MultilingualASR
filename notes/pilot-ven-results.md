# Tshivenda MPS Pilot Fine-Tune - Results

## Pilot v2 update

v2: resumed from v1's weights, added ANV clips <= 10s to the mix (7,325 mixed
train clips after the length cap), 5 more epochs, lr 5e-5. 6.5 h on the M4.
Validation WER fell every epoch: 60.4 -> 53.5 -> 51.1 -> 49.0 -> 48.7%.

Fixed 200-clip test samples (seed 42), directly comparable across rows:

| Model | NCHLT WER | NCHLT CER | ANV WER | ANV CER |
|---|---|---|---|---|
| Whisper Large v3 zero-shot | 1.108 | 0.763 | 1.072 | 0.501 |
| Pilot v1 (5k NCHLT, 3 ep) | 0.614 | 0.151 | 0.851 | 0.257 |
| **Pilot v2 (+ANV, +5 ep, resumed)** | **0.332** | **0.074** | **0.537** | **0.127** |

- NCHLT: 33% WER / 7.4% CER - short read-speech clips are now frequently
  transcribed exactly.
- ANV gap closed substantially (85% -> 54% WER) after including spontaneous
  speech in training.
- **Diacritics emerged**: v1 emitted zero; v2 emits ḓ ḽ ṅ ṱ at roughly
  reference-level frequency (NCHLT: 58 in refs, 67 in hyps). Exception: ṋ is
  still never emitted (0 of 84 across both sets) - likely being confused with
  ṅ/n; worth checking after Stage 1.
- Re-measured error model: S:D:I = 64.4 : 32.7 : 2.9 (v1: 49.5 : 49.3 : 1.3) -
  as the model improves, deletions fall and substitutions dominate, close to
  the corruption engine's 60:25:15 default. Top confusions: nṋe->ne, unga->nga.
- Still a laptop pilot: 12% of available data, 10s clip cap. Stage 1 on GPU
  remains the real run.


Pilot: Wav2Vec2 XLS-R-300M, 5,000 NCHLT train clips (<= 10 s), 3 epochs, on an
M4 MacBook (MPS). NOT the real Stage 1 (60k clips, 10 epochs, GPU) - a proof
that fine-tuning moves the needle, run while GPU access is pending.
Training: 100 min, loss 38 -> 8.7, no divergence.

## WER/CER vs zero-shot (200 clips per corpus, seed 42, same normalisation)

| Model | NCHLT test WER | NCHLT CER | ANV dev_test WER | ANV CER |
|---|---|---|---|---|
| Whisper Large v3 zero-shot | 1.108 | 0.763 | 1.072 | 0.501 |
| Pilot Wav2Vec2 (5k clips, 3 ep) | **0.614** | **0.151** | **0.851** | **0.257** |

- NCHLT CER 15%: the model is clearly learning Tshivenda orthography
  (e.g. "ya u sumbedzwa tshirunzi na" -> "ya u sumbedzwa tshirundzi na").
- ANV lags (out-of-domain for this pilot: trained on NCHLT read speech only,
  ANV is scripted/unscripted spontaneous speech with longer clips).

## Diacritics finding (honest correction)

The pilot emits ZERO diacritics (ḓ ḽ ṅ ṋ ṱ): 757 occurrences in references,
0 in predictions - it writes the plain-letter equivalent (nṋe -> ne,
ṱuṱuwedzwa -> totowedzwa). The vocabulary supports them; 5k clips / 3 epochs
is not enough for the model to learn these rare characters. Whether they
emerge is a key thing to check after the real Stage 1 run.

## Measured error model (first real fine-tuned-ASR error statistics for Tshivenda)

From 400 saved ref/hyp pairs (results/preds_pilot/, gitignored):

- S:D:I ratio = **49.5 : 49.3 : 1.3** (vs the corruption engine's assumed
  60:25:15 default) - real errors are far more deletion-heavy, insertions
  nearly absent. Use via corrupt_transcripts.py --error-model.
- Top confusion pairs are linguistically plausible: nṋe->ne, zwine->zine,
  hu<->u, wana->wa (diacritic loss and cluster simplification dominate).

## Caveats

- Pilot only: small subset, short-clip cap, single corpus, few epochs.
- Deletion-heavy ratio partly reflects degenerate merged-word output on long
  ANV clips; re-measure after Stage 1.
- Regenerate predictions: src/evaluate_wav2vec2.py --checkpoint
  results/wav2vec2-ven-pilot/final --save-predictions results/preds_pilot
